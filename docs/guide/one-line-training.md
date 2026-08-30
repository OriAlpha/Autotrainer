# One-line training: `train()` and `auto()`

`prepare()` keeps your loop and fixes the hardware around it. These two go a
step further and write the recipe as well — inferred once from the model and
the data, printed, and overridable. Neither of them searches: every decision
is a single deterministic read of your batches.

## `train()` — no loop at all

```python
import autotrainer

model = autotrainer.train(model, loader, epochs=5, save_path="model.pt")
```

Infers the loss, optimizer, learning rate and schedule, distributes through
`prepare()`, runs the epochs, prints the summary, and saves. PyTorch, Keras,
scikit-learn, XGBoost and LightGBM all route through the same call:

```python
autotrainer.train(sklearn_estimator, X, y)          # estimators take arrays
autotrainer.train(params_dict, dmatrix, epochs=200)  # native XGBoost API
```

`epochs` means passes over the data for PyTorch and Keras, and
`num_boost_round` on the native-XGBoost path.

Anything you pass explicitly wins over the inference:

```python
autotrainer.train(model, loader, epochs=5, lr=3e-4, loss_fn=my_loss, optimizer="sgd")
```

(`patience=` is the Keras early-stopping knob and applies to that path only.)

## `auto()` — the recipe, then your own loop

When you want the inferred recipe but not the loop:

```python
model, loader, opt, loss_fn, sched = autotrainer.auto(model, loader, epochs=10)

for epoch in range(10):
    for xb, yb in loader:
        ...your loop, with the objects auto() handed back...
```

Every choice is printed with its reasoning, because a silently wrong loss
function trains fine and produces garbage:

```
[autotrainer] auto: loss=cross_entropy (integer targets, 10 classes)
[autotrainer] auto: optimizer=adamw (not a CNN), weight_decay=0.01
[autotrainer] auto: lr=3.16e-04 (LR range test)
[autotrainer] auto: schedule=warmup(78 steps)+cosine over 1560 steps
```

Override any of them by keyword: `loss=`, `optimizer=`, `lr=`, `schedule=`.

## Data checks, before the compute

`auto()` and `train()` already peek at your batches to infer the loss, so they
also check them for the problems that look like a bad recipe and aren't:

```
[autotrainer] auto: data check: class imbalance 19:1 in the 200 targets
  sampled - the largest class is 95% of them, so a model that only ever
  predicts it scores 95% accuracy. Consider class weights in your loss.
```

Also caught: un-normalized or raw 0–255 inputs, NaN/Inf in inputs or targets,
and constant inputs or targets. They run *before* the LR range test and the
first epoch, so you hear about it before the allocation is spent, not after.

Warnings only — nothing is changed for you. `auto(..., sanity=False)`
turns them off.

## Next

- [GPU optimization](gpu-optimization.md) — what `prepare(optimize=True)` does
  underneath both of these.
- [Scaling up](scaling.md) — running either one across GPUs or SLURM nodes.
