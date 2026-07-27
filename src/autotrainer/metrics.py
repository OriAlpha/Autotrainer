"""Selection metrics: what "best" means when ``tune()``/``fit()`` pick a recipe.

By default autotrainer selects on validation LOSS. That is the right default -
it is defined for every task and needs nothing from the user - but it is a
proxy, and on classification the proxy and the goal come apart in a way that
costs real quality. A regularized recipe (label smoothing, augmentation)
produces a *less confident* model, so validation cross-entropy can bottom out
and start climbing from overconfidence while validation accuracy is still
improving. Selecting the recipe, pruning trials, and early-stopping on loss
then hands back a model that is not the most accurate one the run ever saw.

``metric=`` makes the target explicit. Built-ins, all pure torch so sklearn
stays an optional dependency:

    ``loss``      minimize   the training criterion, always scored unsmoothed
    ``accuracy``  maximize   fraction correct (argmax, or a 0-threshold on a
                             single BCE-style logit)
    ``f1``        maximize   macro-averaged over the classes present in the
                             targets - the one to reach for on imbalanced data,
                             where accuracy flatters a majority-class predictor
    ``auc``       maximize   ROC-AUC; binary, or macro one-vs-rest on the
                             softmax probabilities
    ``r2``        maximize   coefficient of determination, for regression

Anything else: pass ``callable(model, loader) -> float``. It is assumed to be
a score (higher is better); pass ``direction="minimize"`` if it is an error.

This mirrors the sklearn-estimator path, which has scored on accuracy/R^2
through ``scoring=`` since 0.10 - the torch path was the inconsistent one.
"""

from __future__ import annotations

from typing import Any

MINIMIZE = "minimize"
MAXIMIZE = "maximize"

# Built-in metric name -> the direction that means "better".
_DIRECTIONS = {
    "loss": MINIMIZE,
    "accuracy": MAXIMIZE,
    "f1": MAXIMIZE,
    "auc": MAXIMIZE,
    "r2": MAXIMIZE,
}


def resolve(metric: Any, direction: str = "auto") -> tuple[Any, str]:
    """Validate ``metric`` and settle the direction to optimize it in.

    Returns ``(metric, direction)`` with ``metric`` normalized (``None`` ->
    ``"loss"``) and ``direction`` resolved to ``"minimize"``/``"maximize"``.
    Raises on an unknown name rather than silently falling back to loss - a
    typo'd metric should not quietly change what the search optimizes.
    """
    if direction not in ("auto", MINIMIZE, MAXIMIZE):
        raise ValueError(
            f"direction must be 'auto', {MINIMIZE!r} or {MAXIMIZE!r}, got {direction!r}"
        )
    if metric is None:
        metric = "loss"
    if callable(metric):
        # A custom scorer is a score unless the caller says otherwise.
        return metric, (MAXIMIZE if direction == "auto" else direction)
    if not isinstance(metric, str) or metric not in _DIRECTIONS:
        raise ValueError(
            f"Unknown metric {metric!r}. Use one of {sorted(_DIRECTIONS)}, or pass a "
            "callable(model, loader) -> float (with direction= if lower is better)."
        )
    return metric, (_DIRECTIONS[metric] if direction == "auto" else direction)


def name_of(metric: Any) -> str:
    """Short label for log lines and study user-attrs."""
    if callable(metric):
        return getattr(metric, "__name__", "metric")
    return str(metric)


def worst(direction: str) -> float:
    """The starting value that any real score beats."""
    return float("-inf") if direction == MAXIMIZE else float("inf")


def is_better(new: float, current: float, direction: str, min_delta: float = 0.0) -> bool:
    """Direction-aware improvement test.

    ``min_delta`` is always a magnitude - the amount of movement that counts as
    real - so the same early-stopping setting means the same thing whether the
    run is minimizing a loss or maximizing an accuracy.
    """
    if direction == MAXIMIZE:
        return new > current + min_delta
    return new < current - min_delta


def score(model: Any, loader: Any, metric: Any, device: Any) -> float:
    """Score ``model`` over ``loader`` under a non-loss ``metric``.

    The loss path stays in ``tuning._evaluate`` (it needs the criterion, which
    metrics don't); this covers the built-in names and custom callables.
    """
    import torch

    from .utils import robust_forward, split_xy, to_device

    if callable(metric):
        return float(metric(model, loader))

    acc = _ACCUMULATORS[metric]()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            xb, yb = split_xy(batch)
            if yb is None:
                raise ValueError(
                    f"metric={metric!r} needs targets, but the validation batches "
                    "carry inputs only. Use metric='loss', or pass a "
                    "callable(model, loader) -> float."
                )
            out = robust_forward(model, to_device(xb, device))
            acc.update(out, to_device(yb, device))
    return float(acc.result())


# --- prediction/target normalization -------------------------------------


def _class_pred(out: Any) -> Any:
    """Predicted class indices from raw model output.

    A single score per sample is read as a BCE-style logit (positive means
    class 1); anything wider is an argmax over dim 1, which is the channel
    dimension for both ``(N, C)`` logits and ``(N, C, H, W)`` segmentation maps.
    """
    out = out.detach()
    if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
        return (out.reshape(-1) > 0).long()
    return out.argmax(dim=1)


def _as_labels(y: Any, pred: Any) -> Any:
    """Targets as non-negative integer class indices shaped like ``pred``."""
    import torch

    y = y.detach()
    y = y.round().long() if torch.is_floating_point(y) else y.long()
    if y.shape != pred.shape:
        if y.numel() != pred.numel():
            raise ValueError(
                f"targets of shape {tuple(y.shape)} don't line up with predictions "
                f"of shape {tuple(pred.shape)}. Class metrics need one label per "
                "prediction - use metric='loss' or a custom callable for this task."
            )
        y = y.reshape(pred.shape)
    if y.numel() and int(y.min()) < 0:
        raise ValueError(
            "class metrics need non-negative integer class targets; got negative "
            "values. Use metric='loss' (or 'r2' for regression)."
        )
    return y


def _grow_add(store: Any, vec: Any) -> Any:
    """Sum two per-class count vectors that may cover different class counts."""
    import torch

    if store is None:
        return vec.clone()
    if vec.numel() > store.numel():
        store = torch.cat([store, store.new_zeros(vec.numel() - store.numel())])
    elif vec.numel() < store.numel():
        vec = torch.cat([vec, vec.new_zeros(store.numel() - vec.numel())])
    return store + vec


# --- accumulators ---------------------------------------------------------
#
# Accuracy/F1/R^2 stream: they keep counts, not predictions, so a large
# validation set costs nothing extra. AUC is rank-based and genuinely needs
# every score, so it is the only one that materializes the predictions.


class _Accuracy:
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, out: Any, y: Any) -> None:
        pred = _class_pred(out)
        y = _as_labels(y, pred)
        self.correct += int((pred == y).sum())
        self.total += y.numel()

    def result(self) -> float:
        if not self.total:
            raise ValueError("accuracy: the validation loader yielded no samples")
        return self.correct / self.total


class _F1:
    """Macro F1 over the classes that actually occur in the targets.

    Averaging over present classes only (rather than over every column the
    model can emit) keeps the number meaningful when the head is wider than
    the label set - an absent class would otherwise contribute a free 0.
    """

    def __init__(self) -> None:
        self.tp: Any = None
        self.fp: Any = None
        self.fn: Any = None

    def update(self, out: Any, y: Any) -> None:
        import torch

        pred = _class_pred(out)
        y = _as_labels(y, pred)
        pred, y = pred.reshape(-1), y.reshape(-1)
        n = int(max(int(pred.max()), int(y.max()))) + 1 if y.numel() else 0
        if not n:
            return
        tp = torch.bincount(pred[pred == y], minlength=n).cpu()
        self.tp = _grow_add(self.tp, tp)
        self.fp = _grow_add(self.fp, torch.bincount(pred, minlength=n).cpu() - tp)
        self.fn = _grow_add(self.fn, torch.bincount(y, minlength=n).cpu() - tp)

    def result(self) -> float:
        if self.tp is None:
            raise ValueError("f1: the validation loader yielded no samples")
        tp, fp, fn = self.tp.double(), self.fp.double(), self.fn.double()
        present = (tp + fn) > 0
        if not bool(present.any()):
            raise ValueError("f1: no classes present in the validation targets")
        tp, fp, fn = tp[present], fp[present], fn[present]
        return float((2 * tp / (2 * tp + fp + fn)).mean())


class _Auc:
    def __init__(self) -> None:
        self.scores: list[Any] = []
        self.labels: list[Any] = []

    def update(self, out: Any, y: Any) -> None:
        import torch

        s = out.detach().float().cpu()
        if s.ndim > 2:
            raise ValueError(
                f"auc: expected (N,) or (N, C) model outputs, got {tuple(s.shape)}. "
                "Use metric='accuracy' or a custom callable for this task."
            )
        if s.ndim == 2 and s.shape[1] == 1:
            s = s.reshape(-1)
        lab = y.detach()
        lab = lab.round().long() if torch.is_floating_point(lab) else lab.long()
        lab = lab.reshape(-1).cpu()
        if lab.numel() != s.shape[0]:
            raise ValueError(
                f"auc: {lab.numel()} targets for {s.shape[0]} predictions in one batch"
            )
        self.scores.append(s)
        self.labels.append(lab)

    def result(self) -> float:
        import torch

        if not self.scores:
            raise ValueError("auc: the validation loader yielded no samples")
        scores = torch.cat(self.scores)
        labels = torch.cat(self.labels)
        if scores.ndim == 1:
            if int(labels.min()) < 0 or int(labels.max()) > 1:
                raise ValueError(
                    "auc: a single output score means binary classification, so "
                    "targets must be 0/1; got labels outside that range."
                )
            pos = labels == 1
            n_pos = int(pos.sum())
            if n_pos == 0 or n_pos == labels.numel():
                raise ValueError(
                    "auc is undefined: the validation set contains only one class. "
                    "Use metric='accuracy' or supply a validation set with both."
                )
            return _binary_auc(scores.double(), pos)

        # Multiclass: macro one-vs-rest over the classes that have both
        # positives and negatives (a class with neither has no defined AUC).
        #
        # Rank on the SOFTMAX probability, not the raw logit. Softmax is not a
        # monotone transform of one logit column - its denominator varies per
        # sample - so the two give different rankings and different AUCs. The
        # probability is the right one: "how confident is the model in class c
        # relative to its alternatives" is what one-vs-rest is asking, and it
        # is what every reference implementation (sklearn's ovr included)
        # scores. Models here emit logits, as CrossEntropyLoss requires.
        probs = scores.double().softmax(dim=1)
        aucs = [
            _binary_auc(probs[:, c], labels == c)
            for c in range(scores.shape[1])
            if 0 < int((labels == c).sum()) < labels.numel()
        ]
        if not aucs:
            raise ValueError(
                "auc is undefined: no class in the validation set has both "
                "positive and negative examples."
            )
        return sum(aucs) / len(aucs)


class _R2:
    def __init__(self) -> None:
        self.n = 0
        self.sy = 0.0
        self.sy2 = 0.0
        self.ssr = 0.0

    def update(self, out: Any, y: Any) -> None:
        p = out.detach().double().reshape(-1)
        t = y.detach().double().reshape(-1)
        if p.numel() != t.numel():
            raise ValueError(f"r2: {t.numel()} targets for {p.numel()} predictions in one batch")
        self.n += t.numel()
        self.sy += float(t.sum())
        self.sy2 += float((t * t).sum())
        self.ssr += float(((t - p) ** 2).sum())

    def result(self) -> float:
        if not self.n:
            raise ValueError("r2: the validation loader yielded no samples")
        ss_tot = self.sy2 - self.sy**2 / self.n
        if ss_tot <= 0:  # constant targets: R^2 undefined, mirror sklearn's convention
            return 1.0 if self.ssr == 0 else 0.0
        return 1.0 - self.ssr / ss_tot


_ACCUMULATORS: dict[str, Any] = {
    "accuracy": _Accuracy,
    "f1": _F1,
    "auc": _Auc,
    "r2": _R2,
}


def _binary_auc(scores: Any, pos: Any) -> float:
    """ROC-AUC via the Mann-Whitney U statistic, with ties broken by average
    rank (the standard convention - ignoring ties inflates the score whenever
    a model emits identical logits, which saturated models do)."""
    n_pos = int(pos.sum())
    n_neg = int(pos.numel()) - n_pos
    ranks = _average_ranks(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(x: Any) -> Any:
    import torch

    n = int(x.numel())
    order = torch.argsort(x)
    uniq, inverse, counts = torch.unique(x[order], return_inverse=True, return_counts=True)
    positions = torch.arange(1, n + 1, dtype=torch.float64)
    sums = torch.zeros(uniq.numel(), dtype=torch.float64).scatter_add_(0, inverse, positions)
    out = torch.empty(n, dtype=torch.float64)
    out[order] = (sums / counts.double())[inverse]
    return out
