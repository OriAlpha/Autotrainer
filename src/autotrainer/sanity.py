"""Data sanity checks: the failures that aren't the framework's fault.

autotrainer can distribute a run perfectly, search a wide recipe, and hand
back the best model it found - and the whole thing can still be worthless
because the data was wrong. Those failures are quiet: un-normalized inputs
look like a bad learning rate, class imbalance looks like a model that
"works" at 95% accuracy, and a train/val split that overlaps looks like an
excellent validation score. Each one costs an allocation to discover the
hard way.

The information needed to catch them is already on hand - ``auto()`` and
``tune()`` both peek at batches to infer the loss - so these checks are
nearly free. They run on a SAMPLE of the data (whatever the loss inference
already gathered, plus a bounded scan for the overlap test), which is why
every message says what it looked at. They only ever warn: nothing here
changes the loss, the recipe, or the data.

Each function returns a list of message strings rather than printing, so the
caller decides whether and how to surface them.
"""

from __future__ import annotations

from typing import Any

# Largest:smallest class ratio that counts as imbalance worth mentioning.
_IMBALANCE_RATIO = 10.0
# How far |mean| or std may drift before inputs look un-standardized.
_SCALE_LIMIT = 10.0
# Rows to sample from each loader for the train/val overlap test.
_OVERLAP_SAMPLE = 512
# Below this many features, byte-identical rows happen by chance (think a
# handful of categorical columns), so value-based overlap means nothing.
_MIN_FEATURES_FOR_OVERLAP = 4

_PREFIX = "data check"


def report(x: Any, targets: Any, loss_name: str) -> list[str]:
    """Check one sample of inputs and targets. Returns warnings, or ``[]``."""
    return _check_inputs(x) + _check_targets(targets, loss_name)


def _check_inputs(x: Any) -> list[str]:
    import torch

    if not torch.is_tensor(x) or not torch.is_floating_point(x) or x.numel() == 0:
        return []

    finite = torch.isfinite(x)
    if not bool(finite.all()):
        n_bad = int((~finite).sum())
        return [
            f"{_PREFIX}: {n_bad} non-finite value(s) (NaN/Inf) in the sampled "
            "inputs. Every loss computed from them will be NaN, so no recipe "
            "can rescue this - fix the data first."
        ]

    lo, hi, mean, std = float(x.min()), float(x.max()), float(x.mean()), float(x.std())
    if x.ndim == 4 and hi > 1.5 and lo >= -0.01 and hi <= 255.5:
        return [
            f"{_PREFIX}: inputs look like raw 0-255 pixels (range "
            f"{lo:.0f}..{hi:.0f}). Divide by 255 and normalize - un-scaled "
            "inputs make the useful learning-rate band very narrow, so the "
            "search spends its trials fighting the data."
        ]
    if abs(mean) > _SCALE_LIMIT or std > _SCALE_LIMIT:
        return [
            f"{_PREFIX}: inputs are far from standardized (mean={mean:.3g}, "
            f"std={std:.3g}). Standardizing them is usually worth more than "
            "anything the recipe search can find."
        ]
    if std < 1e-6:
        return [
            f"{_PREFIX}: the sampled inputs are essentially constant "
            f"(std={std:.3g}) - there may be nothing in them to learn from."
        ]
    return []


def _check_targets(targets: Any, loss_name: str) -> list[str]:
    import torch

    if not torch.is_tensor(targets) or targets.numel() == 0:
        return []

    if torch.is_floating_point(targets) and not bool(torch.isfinite(targets).all()):
        n_bad = int((~torch.isfinite(targets)).sum())
        return [f"{_PREFIX}: {n_bad} non-finite value(s) (NaN/Inf) in the sampled targets."]

    if loss_name in ("cross_entropy", "bce"):
        return _check_classes(targets)

    std = float(targets.double().std()) if targets.numel() > 1 else 0.0
    if std < 1e-9:
        return [
            f"{_PREFIX}: every sampled target is the same value - a model that "
            "predicts that constant will score perfectly and learn nothing."
        ]
    return []


def _check_classes(targets: Any) -> list[str]:
    import torch

    # One label per sample, or nothing to say: multi-label targets (N, C) are
    # a different question than class balance, so leave them alone.
    if targets.numel() != targets.shape[0]:
        return []
    labels = targets.reshape(-1)
    if torch.is_floating_point(labels):
        if not bool((labels == labels.round()).all()):
            return []
        labels = labels.round()
    labels = labels.long()
    if int(labels.min()) < 0:
        return []

    counts = torch.bincount(labels)
    present = counts[counts > 0]
    n = int(labels.numel())
    if present.numel() < 2:
        return [
            f"{_PREFIX}: all {n} sampled targets are the same class. If that "
            "holds for the whole set, there is nothing to classify; if it "
            "doesn't, your loader may not be shuffling."
        ]

    biggest, smallest = int(present.max()), int(present.min())
    if biggest / smallest < _IMBALANCE_RATIO:
        return []
    share = biggest / n
    return [
        f"{_PREFIX}: class imbalance {biggest / smallest:.0f}:1 in the {n} "
        f"targets sampled - the largest class is {share:.0%} of them, so a "
        f"model that only ever predicts it scores {share:.0%} accuracy. "
        "Consider metric='f1' so the search doesn't reward that, and class "
        "weights in your loss."
    ]


def overlap(train_loader: Any, val_loader: Any, sample: int = _OVERLAP_SAMPLE) -> list[str]:
    """Look for validation samples that were also trained on.

    This is the check worth the most: an overlapping split doesn't fail, it
    reports an excellent validation score - and then the recipe search spends
    every trial optimizing against that flattered number.

    Uses exact indices when both loaders are ``Subset``s of one dataset, and
    otherwise falls back to comparing a bounded sample of rows byte-for-byte.
    """
    train_ds = getattr(train_loader, "dataset", None)
    val_ds = getattr(val_loader, "dataset", None)
    if train_ds is None or val_ds is None:
        return []

    from torch.utils.data import IterableDataset, Subset

    # Streaming datasets can't be re-read cheaply (or at all), and a check must
    # never consume the data it is checking.
    if isinstance(train_ds, IterableDataset) or isinstance(val_ds, IterableDataset):
        return []

    if (
        isinstance(train_ds, Subset)
        and isinstance(val_ds, Subset)
        and train_ds.dataset is val_ds.dataset
    ):
        shared = set(map(int, train_ds.indices)) & set(map(int, val_ds.indices))
        if not shared:
            return []
        return [
            f"{_PREFIX}: train and validation share {len(shared)} of the "
            f"validation set's {len(val_ds.indices)} samples (same indices of "
            "the same dataset). The val score - and everything the search "
            "picks from it - is optimistic by however much that leaks."
        ]

    if train_ds is val_ds:
        return [
            f"{_PREFIX}: the train and validation loaders wrap the SAME dataset "
            "object, so every validation sample is also trained on. The val "
            "score measures memorization, not generalization."
        ]

    return _overlap_by_value(train_loader, val_loader, sample)


def _overlap_by_value(train_loader: Any, val_loader: Any, sample: int) -> list[str]:
    val_rows = _sample_rows(val_loader, sample)
    if not val_rows:
        return []
    train_rows = _sample_rows(train_loader, sample)
    if not train_rows:
        return []
    hits = len(set(val_rows) & set(train_rows))
    if hits < 2:  # one coincidence is not evidence
        return []
    return [
        f"{_PREFIX}: {hits} of the {len(val_rows)} validation rows sampled are "
        f"identical to rows among {len(train_rows)} sampled from train. Both "
        "are samples, so the real overlap is likely larger - check the split "
        "before trusting the val score."
    ]


def _sample_rows(loader: Any, limit: int) -> list[bytes] | None:
    """Up to ``limit`` input rows as opaque bytes, or ``None`` if unusable."""
    import torch

    from .utils import split_xy

    rows: list[bytes] = []
    for batch in loader:
        x, _ = split_xy(batch)
        if not torch.is_tensor(x) or not torch.is_floating_point(x) or x.ndim < 2:
            return None
        if x[0].numel() < _MIN_FEATURES_FOR_OVERLAP:
            return None
        for row in x.detach().cpu().float():
            rows.append(row.contiguous().numpy().tobytes())
            if len(rows) >= limit:
                return rows
    return rows
