# Training-loop helpers

The small things people forget *inside* the loop. None of these touch your lr,
loss, schedule, or optimizer choice — they are bookkeeping, not recipe.

All of them are no-ops or safe passthroughs when you aren't distributed or
aren't on a GPU, so the same script runs on a laptop and on a cluster.

## `set_epoch`, `zero_grad`, `eval_mode`

```python
import autotrainer

for epoch in range(epochs):
    autotrainer.set_epoch(loader, epoch)      # fresh shuffle each epoch under DDP
    model.train()
    for xb, yb in loader:
        with autotrainer.autocast_context():
            loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        autotrainer.zero_grad(opt)            # set_to_none=True, saves memory

    # eval_mode restores the prior train/eval state - kills the classic
    # "forgot to flip back to train() after validation" silent bug.
    with autotrainer.eval_mode(model):
        val_loss = evaluate(model, val_loader)
```

`set_epoch` matters more than it looks: without it a `DistributedSampler`
hands every epoch the same order, so your shuffling quietly stops shuffling.
It is a no-op when you aren't distributed, so it costs nothing to always call.

**When `prepare()` installed the sampler, this is already handled.** Passing a
loader to `prepare()` under DDP returns one that advances the epoch on each
pass, so the call above is optional there — kept in the example because it is
still correct, still free, and still required for the cases below. Note the
automatic counter *wins*: it overwrites whatever epoch you set, counting passes
from zero, so an explicit number (resuming at epoch 40, say) does not survive.

You do need to call it yourself when:

- you built the `DistributedSampler` and passed it in — `prepare()` leaves
  loaders you sharded yourself alone, so nothing advances the epoch for you;
- you iterate a sharded loader outside the training pass (a manual eval loop
  over the train split) and want a specific order;
- you need a *particular* epoch number rather than "a different one each pass".

`train_mode` is the mirror of `eval_mode` for the same restore-the-prior-state
reason.

## Gradient accumulation

When the effective batch you want is larger than the one that fits, `accumulate`
scales the *step count*, **not** the lr:

```python
# Effective batch = 4 micro-batches; opt steps once per 4 backwards.
with autotrainer.accumulate(opt, steps=4, scaler=scaler) as acc:
    for micro_xb, micro_yb in micro_batches:
        with autotrainer.autocast_context():
            loss = loss_fn(model(micro_xb), micro_yb) / 4
        acc.backward(loss)
```

## Batch size

Two different tools, depending on whether you already have a loop.

**During `prepare()`** — grow the batch size until OOM, then back off one step:

```python
model, loader, opt = autotrainer.prepare(
    model, loader, opt, auto_bs=True, loss_fn=loss_fn
)
```

Pass `loss_fn` for an accurate forward+backward measurement; without it the
sweep is forward-only and therefore conservative. **lr and schedule are not
changed** — pair it with `accumulate()` if you want the step to reflect the new
effective batch.

**Standalone** — when you have your own one-step function and just want the
number:

```python
best = autotrainer.find_batch_size(model, my_one_step_fn)   # -> int
```

`find_batch_size(model, sample_batch_fn, start=2, max_bs=4096)` runs the same
sweep without touching your loader.

## Finding a learning rate

`find_lr` runs an LR range test on a throwaway copy of your model, so your real
weights and optimizer state are untouched:

```python
lr = autotrainer.find_lr(model, loader, loss_fn)   # -> float
```

It reports a suggestion; what you do with it is yours. If you'd rather have the
whole recipe searched instead of just the lr, that's [`fit()`](fit.md).

## Post-training summary (`finish`, `log_epoch`, `step`, `SummaryTracker`)

At the end of your training script, call `autotrainer.finish()` to print a comprehensive training summary box (hardware topology, VRAM usage, optimizer, batch size, throughput, duration, and health diagnostic) managed by `SummaryTracker` and cleanly close distributed process groups across all supported frameworks.

You can also record epoch metrics or step losses directly into the active summary tracker using `autotrainer.log_epoch` or `autotrainer.step`:

```python
# Log epoch metrics or step losses into the summary tracker
autotrainer.log_epoch(train_loss=0.25, val_loss=0.30)
autotrainer.step(loss=0.25)

# Print summary box & clean up distributed process groups
autotrainer.finish(checkpoint="best_model.pt")
```

## Next

- [Monitors](monitors.md) — is it fast? is it healthy?
- [One-call training](fit.md) — skip the loop entirely.
