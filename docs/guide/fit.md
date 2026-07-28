# One-call training: `fit()`

`fit()` is the whole pipeline in one call — give it a model and data, get back
the best model it can produce on your hardware:

```python
model, params, study = autotrainer.fit(model, train_loader, val_loader, trials=30)
```

1. **Tune.** An ASHA (successive-halving) search over the training recipe — lr,
   weight decay, optimizer, batch size, LR schedule + warmup, gradient clipping,
   training length, (for classification) label smoothing, and (for CNNs)
   augmentation strength — on short trials, with the default space chosen from
   your model and its inferred loss. Most candidates get a small budget and only
   survivors are promoted, so the wide space stays cheap. Launched distributed,
   trials are split across all ranks via a shared journal-file study — one trial
   per process, every GPU busy during the search.
2. **Train.** The winning recipe is retrained from your model's original init
   through `prepare()` — which auto-distributes it across every GPU/node — with
   the winning schedule, mixed precision, and early stopping on the val score.
   The best epoch's weights are returned.

If you want the search without the final training run, that's `tune()`, which
returns the same `(model, params, study)` shape.

## Select on the metric you care about

Both phases select on validation loss by default. Loss is a proxy, and on
classification it drifts from the goal exactly where it matters: a regularized
recipe is less confident, so val cross-entropy can bottom out and start climbing
from overconfidence while val accuracy is still improving — and a loss-driven
`patience` stops there and hands back a less-accurate model. Name the number
instead:

```python
model, params, study = autotrainer.fit(model, train, val, metric="accuracy")
```

`"accuracy"`, `"f1"` (macro — use it on imbalanced data), `"auc"`, `"r2"`, or
your own `callable(model, loader) -> float` (add `direction="minimize"` if lower
is better). It drives the search, the ASHA pruning, early stopping, and
best-epoch selection. Training always uses the loss; this changes only what runs
are *scored* by.

Pass `test_loader=` to also get an honest held-out score the search never saw
(printed, and stored on `study.user_attrs["test_score"]`) — the number to trust
once a wide search has been optimizing against your val set.

## Data checks, before the compute

`auto()` and `tune()` already peek at your batches to infer the loss, so they
also check them for the problems that look like a bad recipe and aren't:

```
[autotrainer] tune: data check: class imbalance 19:1 in the 200 targets sampled -
  the largest class is 95% of them, so a model that only ever predicts it scores
  95% accuracy. Consider metric='f1' so the search doesn't reward that, and class
  weights in your loss.
[autotrainer] tune: data check: train and validation share 30 of the validation
  set's 50 samples (same indices of the same dataset). The val score - and
  everything the search picks from it - is optimistic by however much that leaks.
```

Also caught: un-normalized or raw 0–255 inputs, NaN/Inf in inputs or targets,
and constant inputs or targets. They run *before* the LR range test and the
first trial, so you hear about it before the allocation is spent, not after.

Warnings only — nothing is changed for you. `sanity=False` turns them off.

## Surviving preemption

Pass `checkpoint="fit.ckpt"` to make it preemption-safe: the full training state
is saved every epoch, and rerunning the same script resumes where it died
(skipping the search) — ideal for requeued SLURM jobs. It also arms the signal
watcher, so with

```bash
#SBATCH --signal=B:USR1@120
```

a preempted job stops at the next epoch boundary *after* its checkpoint is
written, instead of losing the epoch it was in the middle of.

The search is journaled to `fit.ckpt.study`, so a job preempted during phase 1
resumes with its completed trials rather than searching again. Delete both files
to start fresh.

## Augmentation

For CNNs the search covers an `aug_strength` scalar over flip + cutout. The same
transform is available directly if you want it in your own loop:

```python
xb = autotrainer.augment_batch(xb, strength=0.5)
```

## Other frameworks

`tune()` also handles sklearn-API estimators — scikit-learn, XGBoost, LightGBM —
with curated default spaces. Pass `(X, y)` tuples instead of loaders, and score
with `scoring=` rather than `metric=`:

```python
best_est, params, study = autotrainer.tune(XGBClassifier(), (X, y), (X_val, y_val))
```

See [`examples/`](../../examples/) for runnable scripts per framework.
