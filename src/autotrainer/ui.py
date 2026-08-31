"""Built-in Web UI server for autotrainer.

Provides a zero-dependency local web dashboard (`autotrainer ui`) using
Python's standard `http.server`. Serves real-time charts of loss curves,
custom styled multi-user workspace dropdown, dynamic multi-path log directories,
multi-run comparisons, AI health triage doctor, hardware efficiency gauge,
file locations, run renaming, enlarged dual chart modal, and 1-click standalone HTML report export with download confirmation popup.

The two documents it serves - the dashboard and the standalone export report -
live in ``autotrainer/templates/*.html`` rather than in string literals here,
so an editor can lint and highlight them as the HTML/CSS/JS they are. This
module holds only the server: routing, auth, the REST API, and the run-data
reads the templates render. See :func:`_template` and :func:`_render`.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import secrets
import shutil
import socketserver
import time
import webbrowser
from functools import cache
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


def _detect_system_user() -> str:
    for env_key in ("AUTOTRAINER_USER", "SLURM_JOB_USER", "USER", "USERNAME", "LOGNAME"):
        val = os.getenv(env_key)
        if val:
            return val.strip()
    try:
        return getpass.getuser().strip()
    except Exception:
        return "default"


_RUN_MARKERS = ("run.json", "metrics.csv", "metrics.jsonl")


def _looks_like_logs_dir(path: Path) -> bool:
    """True when ``path`` holds at least one autotrainer run directory.

    Gate for adding a source at runtime. Without it any absolute path was
    accepted, so the dashboard doubled as a filesystem browser for whoever
    could reach it. Only the immediate children are inspected - a run lives
    directly under a logs root - so this stays cheap on large trees.
    """
    try:
        for child in path.iterdir():
            if child.is_dir() and any((child / marker).exists() for marker in _RUN_MARKERS):
                return True
    except OSError:
        return False
    return False


def _get_live_hardware() -> dict[str, Any]:
    """Return live host CPU, RAM, and GPU memory telemetry."""
    hw: dict[str, Any] = {
        "timestamp": round(time.time(), 2),
        "gpu_available": False,
        "gpus": [],
        "cpu_pct": 0.0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "ram_pct": 0.0,
    }
    try:
        import psutil

        hw["cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
        vm = psutil.virtual_memory()
        hw["ram_used_gb"] = round((vm.total - vm.available) / (1024**3), 2)
        hw["ram_total_gb"] = round(vm.total / (1024**3), 2)
        hw["ram_pct"] = round(vm.percent, 1)
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            hw["gpu_available"] = True
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated_mb = torch.cuda.memory_allocated(i) / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved(i) / (1024 * 1024)
                total_mb = props.total_memory / (1024 * 1024)
                hw["gpus"].append(
                    {
                        "id": i,
                        "name": props.name,
                        "allocated_mb": round(allocated_mb, 1),
                        "reserved_mb": round(reserved_mb, 1),
                        "total_mb": round(total_mb, 1),
                        "allocated_gb": round(allocated_mb / 1024, 2),
                        "total_gb": round(total_mb / 1024, 2),
                        "util_pct": round((allocated_mb / total_mb) * 100, 1)
                        if total_mb > 0
                        else 0.0,
                    }
                )
    except Exception:
        pass

    return hw


@cache
def _template(name: str) -> str:
    """Read a UI template from ``autotrainer/templates/``, once per process.

    The dashboard and the export report are HTML/CSS/JS documents, so they
    live in .html files where an editor can lint and highlight them, rather
    than as multi-thousand-line string literals in this module. Cached
    because the dashboard template is read on every page load.
    """
    return (resources.files(__package__) / "templates" / name).read_text(encoding="utf-8")


def _render(template: str, **values: str) -> str:
    """Fill ``__AT_NAME__`` slots in a template.

    Plain substitution rather than str.format or string.Template: the
    templates are full of CSS braces and JavaScript ``${...}`` literals, and
    both of those would need every one of them escaped, which would stop the
    files being valid CSS and JS.
    """
    for key, value in values.items():
        template = template.replace(f"__AT_{key.upper()}__", value)
    return template


class AutotrainerUIHandler(BaseHTTPRequestHandler):
    """HTTP Request handler for serving the Autotrainer Web UI and REST API.

    The API mutates the filesystem - it renames, archives and deletes run
    directories - so two guards sit in front of every request:

    * :meth:`_safe_run_dir` resolves a ``run_id`` and refuses anything that
      escapes the configured log roots. Without it ``run_id`` was pasted
      straight into ``root / run_id``, so ``../..`` walked anywhere on disk
      and ``DELETE /api/runs/../<path>`` reached ``shutil.rmtree``.
    * :meth:`_authorized` requires the session token unless the server was
      started with ``token=None``. Paired with the loopback default bind in
      :func:`run_ui_server`, that keeps a dashboard on a shared login node
      from being someone else's remote-control panel.
    """

    logs_dirs: list[Path] = [Path("logs")]
    # Set by run_ui_server(). None disables the check entirely (--no-token).
    auth_token: str | None = None

    # -- security ---------------------------------------------------------

    def _safe_run_dir(self, run_id: str) -> Path | None:
        """Resolve ``run_id`` to a directory inside a configured log root.

        A run id names one directory directly under a root, so anything with
        a path separator, a drive letter, or a ``..`` component is invalid by
        construction. Resolution happens before the containment check so a
        symlink pointing outside is caught too. Returns None on any failure -
        callers turn that into a uniform 404 rather than leaking whether the
        path existed.
        """
        if not run_id or run_id in (".", ".."):
            return None
        if any(sep in run_id for sep in ("/", "\\", "\x00")):
            return None

        for root in self.logs_dirs:
            try:
                root_resolved = root.resolve()
                candidate = (root_resolved / run_id).resolve()
            except OSError:
                continue
            if candidate == root_resolved or not candidate.is_relative_to(root_resolved):
                continue
            if candidate.is_dir():
                return candidate
        return None

    def _authorized(self) -> bool:
        """True when the request carries the session token.

        Accepted from the ``token`` query parameter (how the printed URL
        arrives), the ``autotrainer_token`` cookie (set on that first page
        load, so the dashboard's own fetch() calls carry it), or an
        ``X-Autotrainer-Token`` header for scripted access.
        """
        if not self.auth_token:
            return True

        query = parse_qs(urlparse(self.path).query)
        tokens = query.get("token") or []
        supplied: str | None = tokens[0] if tokens else None
        if supplied is None:
            supplied = self.headers.get("X-Autotrainer-Token")
        if supplied is None:
            cookie_header = self.headers.get("Cookie", "")
            jar = SimpleCookie()
            with contextlib.suppress(Exception):
                jar.load(cookie_header)
            if "autotrainer_token" in jar:
                supplied = jar["autotrainer_token"].value

        if not supplied:
            return False
        # Constant-time: a plain == leaks the token prefix through timing.
        return secrets.compare_digest(supplied, self.auth_token)

    def _same_origin(self) -> bool:
        """Reject cross-site state-changing requests.

        The token cookie is SameSite=Strict, so a browser will not attach it
        to a cross-site POST in the first place; this is the belt to that
        suspenders, and also covers clients that ignore SameSite.
        """
        origin = self.headers.get("Origin")
        if not origin:
            return True  # non-browser client; the token is the gate there
        host = self.headers.get("Host", "")
        return urlparse(origin).netloc == host

    def _guard(self, *, mutating: bool = False) -> bool:
        """Run the request guards, responding with the error if one fails."""
        if not self._authorized():
            self._respond_error(
                HTTPStatus.UNAUTHORIZED,
                "Missing or invalid token. Use the URL printed by `autotrainer ui`.",
            )
            return False
        if mutating and not self._same_origin():
            self._respond_error(HTTPStatus.FORBIDDEN, "Cross-origin request refused")
            return False
        return True

    @property
    def route(self) -> str:
        """The request path with any query string removed.

        Routing has to match on this rather than ``self.path``: the token
        arrives as ``?token=...``, which would otherwise turn every endpoint
        into a 404.
        """
        return urlparse(self.path).path

    def do_GET(self) -> None:
        if not self._guard():
            return

        route = self.route
        if route in ("/", "/index.html"):
            # First load carries the token in the URL; hand it back as a
            # SameSite cookie so the dashboard's fetch() calls are
            # authenticated without threading it through every JS call site.
            self._respond_html(_template("dashboard.html"), set_token_cookie=True)
        elif route == "/api/hardware":
            self._handle_api_hardware()
        elif route == "/api/sources":
            self._handle_api_sources()
        elif route == "/api/runs":
            self._handle_api_runs()
        elif route.startswith("/api/runs/") and (
            route.endswith("/export") or route.endswith("/export/html")
        ):
            self._handle_export_html(route.strip("/").split("/")[2], as_attachment=True)
        elif route.startswith("/api/runs/") and (
            route.endswith("/export/markdown") or route.endswith("/export/md")
        ):
            self._handle_export_markdown(route.strip("/").split("/")[2])
        elif route.startswith("/api/runs/") and route.endswith("/export/csv"):
            self._handle_export_csv(route.strip("/").split("/")[2])
        elif route.startswith("/api/runs/") and route.endswith("/export/json"):
            self._handle_export_json(route.strip("/").split("/")[2])
        elif route.startswith("/api/runs/") and (
            route.endswith("/report") or route.endswith("/preview")
        ):
            self._handle_export_html(route.strip("/").split("/")[2], as_attachment=False)
        elif route.startswith("/api/runs/"):
            self._handle_api_run_detail(route[len("/api/runs/") :])
        else:
            self._respond_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_POST(self) -> None:
        if not self._guard(mutating=True):
            return

        if self.route == "/api/sources":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                raw_path = data.get("path", "").strip()
                if not raw_path:
                    self._respond_error(HTTPStatus.BAD_REQUEST, "Missing path")
                    return

                new_dir = Path(raw_path).resolve()
                if not new_dir.is_dir():
                    self._respond_error(HTTPStatus.BAD_REQUEST, "Not a directory")
                    return
                if not _looks_like_logs_dir(new_dir):
                    # Any absolute path used to be accepted here, which made
                    # the dashboard a filesystem browser: POST /api/sources
                    # with a home directory and every subfolder was listed as
                    # a "run". Require it to actually contain runs.
                    self._respond_error(
                        HTTPStatus.BAD_REQUEST,
                        "No autotrainer runs found in that directory. A logs "
                        "directory contains run folders with run.json, "
                        "metrics.csv, or metrics.jsonl.",
                    )
                    return
                if new_dir not in [d.resolve() for d in self.logs_dirs]:
                    self.logs_dirs.append(new_dir)

                self._respond_json({"success": True, "path": str(new_dir)})
            except Exception as e:
                self._respond_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

        elif self.route == "/api/runs/rename":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                old_id = data.get("old_run_id")
                new_id = data.get("new_run_id")
                if not old_id or not new_id:
                    self._respond_error(HTTPStatus.BAD_REQUEST, "Missing old_run_id or new_run_id")
                    return

                clean_new_id = "".join(c for c in new_id if c.isalnum() or c in ("_", "-")).strip()
                if not clean_new_id:
                    self._respond_error(HTTPStatus.BAD_REQUEST, "Invalid run_id")
                    return

                found_dir = self._safe_run_dir(old_id)
                if not found_dir:
                    self._respond_error(HTTPStatus.NOT_FOUND, f"Run {old_id} not found")
                    return

                new_path = found_dir.parent / clean_new_id
                if new_path.exists() and found_dir != new_path:
                    self._respond_error(HTTPStatus.CONFLICT, f"Run {clean_new_id} already exists")
                    return

                if found_dir != new_path:
                    found_dir.rename(new_path)

                meta_file = new_path / "run.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r+", encoding="utf-8") as f:
                            meta = json.load(f)
                            meta["run_id"] = clean_new_id
                            f.seek(0)
                            json.dump(meta, f, indent=2)
                            f.truncate()
                    except Exception:
                        pass

                self._respond_json({"success": True, "run_id": clean_new_id})
            except Exception as e:
                self._respond_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

        elif self.route.startswith("/api/runs/") and self.route.endswith("/tags"):
            run_id = self.route[len("/api/runs/") : -len("/tags")].strip("/")
            self._handle_api_tags(run_id)

        elif self.route.startswith("/api/runs/") and self.route.endswith("/favorite"):
            run_id = self.route[len("/api/runs/") : -len("/favorite")].strip("/")
            self._handle_api_favorite(run_id)

        elif self.route.startswith("/api/runs/") and self.route.endswith("/notes"):
            run_id = self.route[len("/api/runs/") : -len("/notes")].strip("/")
            self._handle_api_notes(run_id)

        elif self.route.startswith("/api/runs/") and self.route.endswith("/archive"):
            run_id = self.route[len("/api/runs/") : -len("/archive")].strip("/")
            self._handle_api_archive(run_id)

        else:
            self._respond_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_DELETE(self) -> None:
        if not self._guard(mutating=True):
            return

        if self.route == "/api/sources":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                raw_path = data.get("path", "").strip()
                to_remove = Path(raw_path).resolve()
                self.logs_dirs = [d for d in self.logs_dirs if d.resolve() != to_remove]
                self._respond_json({"success": True})
            except Exception as e:
                self._respond_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
        elif self.route.startswith("/api/runs/"):
            run_id = self.route[len("/api/runs/") :].strip("/")
            self._handle_api_delete_run(run_id)
        else:
            self._respond_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _handle_api_hardware(self) -> None:
        self._respond_json(_get_live_hardware())

    def _handle_api_tags(self, run_id: str) -> None:
        run_dir = self._safe_run_dir(run_id)
        if not run_dir:
            self._respond_error(HTTPStatus.NOT_FOUND, f"Run {run_id} not found")
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            data = json.loads(body)
            action = data.get("action", "add")
            tag = str(data.get("tag", "")).strip().lower()
            meta_file = run_dir / "run.json"
            meta = {}
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
            tags = meta.get("tags", [])
            if action == "add" and tag and tag not in tags:
                tags.append(tag)
            elif action == "remove" and tag in tags:
                tags.remove(tag)
            elif "tags" in data:
                tags = [str(t).strip().lower() for t in data["tags"] if str(t).strip()]
            meta["tags"] = tags
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            self._respond_json({"success": True, "tags": tags})
        except Exception as e:
            self._respond_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def _handle_api_favorite(self, run_id: str) -> None:
        run_dir = self._safe_run_dir(run_id)
        if not run_dir:
            self._respond_error(HTTPStatus.NOT_FOUND, f"Run {run_id} not found")
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            data = json.loads(body) if body else {}
            meta_file = run_dir / "run.json"
            meta = {}
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
            current_fav = meta.get("favorite", False)
            new_fav = data.get("favorite", not current_fav)
            meta["favorite"] = bool(new_fav)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            self._respond_json({"success": True, "favorite": meta["favorite"]})
        except Exception as e:
            self._respond_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def _handle_api_notes(self, run_id: str) -> None:
        run_dir = self._safe_run_dir(run_id)
        if not run_dir:
            self._respond_error(HTTPStatus.NOT_FOUND, f"Run {run_id} not found")
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            data = json.loads(body)
            notes = str(data.get("notes", ""))
            meta_file = run_dir / "run.json"
            meta = {}
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
            meta["notes"] = notes
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            with contextlib.suppress(Exception):
                (run_dir / "notes.md").write_text(notes, encoding="utf-8")
            self._respond_json({"success": True, "notes": notes})
        except Exception as e:
            self._respond_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def _handle_api_archive(self, run_id: str) -> None:
        run_dir = self._safe_run_dir(run_id)
        if not run_dir:
            self._respond_error(HTTPStatus.NOT_FOUND, f"Run {run_id} not found")
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            data = json.loads(body) if body else {}
            meta_file = run_dir / "run.json"
            meta = {}
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
            current_archived = meta.get("archived", False)
            new_archived = data.get("archived", not current_archived)
            meta["archived"] = bool(new_archived)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            self._respond_json({"success": True, "archived": meta["archived"]})
        except Exception as e:
            self._respond_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def _handle_api_delete_run(self, run_id: str) -> None:
        run_dir = self._safe_run_dir(run_id)
        if not run_dir:
            self._respond_error(HTTPStatus.NOT_FOUND, f"Run {run_id} not found")
            return
        try:
            shutil.rmtree(run_dir)
            self._respond_json({"success": True, "deleted_run_id": run_id})
        except Exception as e:
            self._respond_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def _respond_html(self, content: str, *, set_token_cookie: bool = False) -> None:
        encoded = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if set_token_cookie and self.auth_token:
            # HttpOnly so page scripts can't read it; SameSite=Strict so a
            # browser never attaches it to a cross-site request, which is what
            # stops another tab from driving the delete endpoint.
            self.send_header(
                "Set-Cookie",
                f"autotrainer_token={self.auth_token}; Path=/; HttpOnly; SameSite=Strict",
            )
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _respond_json(self, data: Any) -> None:
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _respond_error(self, status: HTTPStatus, message: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode("utf-8"))

    def _handle_api_sources(self) -> None:
        sources_info: list[dict[str, Any]] = []
        for idx, d in enumerate(self.logs_dirs):
            resolved = d.resolve()
            exists = resolved.exists() and resolved.is_dir()
            runs_count = 0
            if exists:
                with contextlib.suppress(Exception):
                    runs_count = sum(1 for item in resolved.iterdir() if item.is_dir())
            sources_info.append(
                {
                    "path": str(resolved),
                    "exists": exists,
                    "runs_count": runs_count,
                    "is_default": idx == 0,
                }
            )
        self._respond_json(sources_info)

    def _handle_api_runs(self) -> None:
        default_user = _detect_system_user()
        runs: list[dict[str, Any]] = []
        seen_run_ids = set()

        for d in self.logs_dirs:
            if d.exists() and d.is_dir():
                for item in sorted(d.iterdir(), reverse=True):
                    if item.is_dir() and item.name not in seen_run_ids:
                        seen_run_ids.add(item.name)
                        meta_file = item / "run.json"
                        meta: dict[str, Any] = {
                            "run_id": item.name,
                            "user": default_user,
                            "source_dir": str(d.resolve()),
                        }
                        if meta_file.exists():
                            try:
                                with open(meta_file, encoding="utf-8") as f:
                                    loaded = json.load(f)
                                    meta.update(loaded)
                            except Exception:
                                pass
                        notes_file = item / "notes.md"
                        if notes_file.exists() and not meta.get("notes"):
                            with contextlib.suppress(Exception):
                                meta["notes"] = notes_file.read_text(encoding="utf-8")
                        meta.setdefault("tags", [])
                        meta.setdefault("favorite", False)
                        meta.setdefault("notes", "")
                        meta.setdefault("archived", False)
                        runs.append(meta)
        self._respond_json(runs)

    def _get_run_data(self, run_id: str) -> dict[str, Any] | None:
        # Same containment check as every other entry point - this method had
        # its own inlined copy of the unchecked lookup, so hardening only
        # _find_run_dir would have left the read path wide open.
        run_dir = self._safe_run_dir(run_id)
        if not run_dir:
            return None

        default_user = _detect_system_user()
        metadata: dict[str, Any] = {"user": default_user}
        meta_file = run_dir / "run.json"
        if meta_file.exists():
            try:
                with open(meta_file, encoding="utf-8") as f:
                    metadata = json.load(f)
                    if "user" not in metadata:
                        metadata["user"] = default_user
            except Exception:
                pass

        notes_file = run_dir / "notes.md"
        if notes_file.exists() and not metadata.get("notes"):
            with contextlib.suppress(Exception):
                metadata["notes"] = notes_file.read_text(encoding="utf-8")

        metadata.setdefault("tags", [])
        metadata.setdefault("favorite", False)
        metadata.setdefault("notes", "")
        metadata.setdefault("archived", False)

        metrics: list[dict[str, Any]] = []
        jsonl_file = run_dir / "metrics.jsonl"
        if jsonl_file.exists():
            try:
                with open(jsonl_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if record.get("type") in ("epoch", "step"):
                                item = record.get("metrics", {})
                                if "epoch" in record:
                                    item["epoch"] = record["epoch"]
                                if "step" in record:
                                    item["step"] = record["step"]
                                metrics.append(item)
            except Exception:
                pass

        csv_file = run_dir / "metrics.csv"

        paths = {
            "logs_dir": str(run_dir.resolve()),
            "metrics_csv": str(csv_file.resolve()) if csv_file.exists() else "N/A",
            "metrics_jsonl": str(jsonl_file.resolve()) if jsonl_file.exists() else "N/A",
            "saved_model": str(Path("model.pt").resolve()) if Path("model.pt").exists() else "N/A",
        }

        return {
            "run_id": run_id,
            "metadata": metadata,
            "metrics": metrics,
            "paths": paths,
        }

    def _handle_api_run_detail(self, run_id: str) -> None:
        data = self._get_run_data(run_id)
        if data is None:
            self._respond_error(HTTPStatus.NOT_FOUND, f"Run {run_id} not found")
            return
        self._respond_json(data)

    def _handle_export_html(self, run_id: str, as_attachment: bool = True) -> None:
        data = self._get_run_data(run_id)
        if data is None:
            self._respond_error(HTTPStatus.NOT_FOUND, f"Run {run_id} not found")
            return

        json_str = json.dumps(data)
        user_name = data.get("metadata", {}).get("user", _detect_system_user())
        start_time = data.get("metadata", {}).get("start_datetime", "N/A")

        standalone_html = _render(
            _template("report.html"),
            run_id=run_id,
            user_name=user_name,
            start_time=start_time,
            json_str=json_str,
        )

        # Also save persistently into the run directory on disk
        # Containment-checked like every other run_id use: this writes a
        # file, so an unchecked join here was an arbitrary-write primitive.
        run_dir = self._safe_run_dir(run_id)
        if run_dir is not None:
            with contextlib.suppress(Exception):
                (run_dir / "report.html").write_text(standalone_html, encoding="utf-8")

        encoded = standalone_html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if as_attachment:
            self.send_header(
                "Content-Disposition", f'attachment; filename="autotrainer_report_{run_id}.html"'
            )
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_export_markdown(self, run_id: str) -> None:
        data = self._get_run_data(run_id)
        if data is None:
            self._respond_error(HTTPStatus.NOT_FOUND, f"Run {run_id} not found")
            return

        meta = data.get("metadata", {})
        summary = meta.get("summary", {})
        user_name = meta.get("user", _detect_system_user())
        start_time = meta.get("start_datetime", "N/A")
        triage_list = summary.get("triage_diagnostics", [])

        # Compute losses
        metrics = data.get("metrics", [])
        train_losses = [
            m.get("train_loss") or m.get("loss")
            for m in metrics
            if (m.get("train_loss") or m.get("loss")) is not None
        ]
        val_losses = [m.get("val_loss") for m in metrics if m.get("val_loss") is not None]

        first_loss = summary.get("init_loss", train_losses[0] if train_losses else "N/A")
        latest_train_loss = train_losses[-1] if train_losses else summary.get("final_loss", "N/A")
        latest_val_loss = val_losses[-1] if val_losses else summary.get("val_loss", "N/A")

        first_loss_str = (
            f"{first_loss:.4f}" if isinstance(first_loss, (int, float)) else str(first_loss)
        )
        train_loss_str = (
            f"{latest_train_loss:.4f}"
            if isinstance(latest_train_loss, (int, float))
            else str(latest_train_loss)
        )
        val_loss_str = (
            f"{latest_val_loss:.4f}"
            if isinstance(latest_val_loss, (int, float))
            else str(latest_val_loss)
        )

        md = [
            f"# ⚡ Autotrainer Run Summary: `{run_id}`",
            f"> **Author**: `👤 {user_name}` | **Started**: `{start_time}` | **Status**: `Completed`",
            "",
            "## 📊 Key Telemetry",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **First Loss** | `{first_loss_str}` |",
            f"| **Training Loss** | `{train_loss_str}` |",
            f"| **Validation Loss** | `{val_loss_str}` |",
            f"| **Epochs** | `{summary.get('epochs', len(data.get('metrics', [])))}` |",
            "",
            "## 🧠 AI Health Triage Diagnosis",
        ]

        if triage_list:
            for item in triage_list:
                clean_item = item.replace("[autotrainer] triage: ", "")
                md.append(f"- ⚠️ {clean_item}")
        else:
            md.append(
                "- ✅ **All Systems Healthy**: No numerical anomalies, vanishing gradients, or dataloader starvation detected."
            )

        md.extend(
            [
                "",
                "## ⚙️ Hyperparameters & Configuration",
                "| Parameter | Value |",
                "| :--- | :--- |",
            ]
        )

        combined = {
            **meta.get("params", {}),
            **{k: v for k, v in summary.items() if k != "triage_diagnostics"},
        }
        for k, v in combined.items():
            md.append(f"| `{k}` | `{v}` |")

        md_content = "\n".join(md) + "\n"

        # Also save to run directory
        # Containment-checked like every other run_id use: this writes a
        # file, so an unchecked join here was an arbitrary-write primitive.
        run_dir = self._safe_run_dir(run_id)
        if run_dir is not None:
            with contextlib.suppress(Exception):
                (run_dir / "report.md").write_text(md_content, encoding="utf-8")

        encoded = md_content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header(
            "Content-Disposition", f'attachment; filename="autotrainer_summary_{run_id}.md"'
        )
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_export_csv(self, run_id: str) -> None:
        data = self._get_run_data(run_id)
        if data is None:
            self._respond_error(HTTPStatus.NOT_FOUND, f"Run {run_id} not found")
            return

        csv_content = ""
        csv_path = data.get("paths", {}).get("metrics_csv")
        if csv_path and csv_path != "N/A" and Path(csv_path).exists():
            try:
                csv_content = Path(csv_path).read_text(encoding="utf-8")
            except Exception:
                csv_content = ""

        if not csv_content:
            metrics = data.get("metrics", [])
            if metrics:
                import csv as py_csv
                import io

                output = io.StringIO()
                all_keys = list(dict.fromkeys(k for m in metrics for k in m))
                writer = py_csv.DictWriter(output, fieldnames=all_keys)
                writer.writeheader()
                for m in metrics:
                    writer.writerow(m)
                csv_content = output.getvalue()
            else:
                csv_content = "epoch,step,train_loss,val_loss,val_acc\n"

        encoded = csv_content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="metrics_{run_id}.csv"')
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_export_json(self, run_id: str) -> None:
        data = self._get_run_data(run_id)
        if data is None:
            self._respond_error(HTTPStatus.NOT_FOUND, f"Run {run_id} not found")
            return

        json_str = json.dumps(data, indent=2)
        encoded = json_str.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="run_data_{run_id}.json"')
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        # Silence default stderr logging to keep console clean
        pass


def run_ui_server(
    logs_dir: str | Path | list[str | Path] | None = None,
    logs_dirs: list[str | Path] | None = None,
    port: int = 8501,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    token: str | None = "",
) -> None:
    """Launch the autotrainer Web UI server over one or more logs directories.

    Args:
        logs_dir: a single logs directory, or a list of them.
        logs_dirs: explicit list form; takes precedence over ``logs_dir``.
        port: TCP port to serve on.
        open_browser: open the dashboard automatically once bound.
        host: interface to bind. **Defaults to loopback.** The API renames,
            archives and deletes run directories, so binding it to
            ``0.0.0.0`` publishes that to everyone who can route to the box -
            on a shared SLURM login node, the whole cluster. Pass an explicit
            address to widen it; a warning is printed when it isn't loopback.
        token: session token required on every request. ``""`` (default)
            generates one and prints it as part of the URL; ``None`` disables
            authentication entirely, which is only reasonable on a machine
            where you trust every local user.
    """
    paths: list[Path] = []

    if logs_dirs:
        for p in logs_dirs:
            paths.append(Path(p).resolve())
    elif logs_dir:
        if isinstance(logs_dir, (list, tuple)):
            for p in logs_dir:
                paths.append(Path(p).resolve())
        else:
            paths.append(Path(logs_dir).resolve())
    else:
        paths.append(Path("logs").resolve())

    AutotrainerUIHandler.logs_dirs = paths

    resolved_token = secrets.token_urlsafe(32) if token == "" else token
    AutotrainerUIHandler.auth_token = resolved_token

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer((host, port), AutotrainerUIHandler)

    display_host = "localhost" if host in ("127.0.0.1", "localhost", "::1") else host
    url = f"http://{display_host}:{port}"
    if resolved_token:
        url = f"{url}/?token={resolved_token}"

    print(f"[autotrainer] Web UI running at {url} (monitoring {len(paths)} directories)")
    if host not in ("127.0.0.1", "localhost", "::1"):
        # Deleting and renaming runs is part of this API, so a non-loopback
        # bind hands that to anyone who can reach the port. Say so plainly
        # rather than letting it be discovered.
        print(
            f"[autotrainer] WARNING: bound to {host}, not loopback - the run "
            "rename/archive/delete endpoints are reachable from the network. "
            + (
                "The session token above is the only thing gating them."
                if resolved_token
                else "Authentication is DISABLED (token=None). Anyone who can "
                "reach this port can delete your runs."
            )
        )
    if not resolved_token:
        print("[autotrainer] WARNING: authentication disabled (token=None).")

    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Autotrainer Web UI server...")
        server.server_close()
