# How autotrainer compares


Most tools in this space ask you to hand over your training loop, your launcher,
or both. Autotrainer's bet is that you shouldn't have to give up either to use
your hardware properly.

| Tool | What it gives you | What you change to adopt it |
|---|---|---|
| `torch.distributed` + `torchrun` | The primitives | Write the DDP wrap, the sampler, the launcher flags — every time |
| HF Accelerate | Device/precision/distribution abstraction | Restructure around `accelerator.*`; configure and use `accelerate launch` |
| PyTorch Lightning | A full training framework | Move your code into a `LightningModule` + `Trainer` |
| Ray Train | Distributed execution | Adopt the Ray runtime and wrap your function in it |
| **autotrainer** | **All of the above paths, on your existing loop** | **One line: `prepare(model, loader, opt)`** |

Concretely, four things are unusual here:

1. **Your loop stays yours.** `prepare()` returns the same three objects you
   passed in. There is no base class to inherit, no trainer object to configure,
   and no callback system to learn. Deleting the import leaves a working script.
2. **Your hyperparameters are never touched silently.** lr, loss, schedule, and
   optimizer choice are yours unless you explicitly call `auto()` or `train()`
   to have them inferred. Everything `optimize=True` does — TF32,
   `cudnn.benchmark`, `channels_last`, workers, AMP — is throughput, not
   recipe. And it prints every change it makes, so nothing is a surprise.
3. **No launcher to configure.** `python train.py` spawns one worker per GPU on
   its own. There is no config file to generate and no separate launch command,
   and under SLURM the auto-spawn correctly stands down so `srun autotrainer
   run` behaves. `autotrainer doctor` checks the setup first — the CPU budget
   and the `num_workers` it implies, `--ntasks-per-node` against your GPU
   count, whether scratch is on NFS, and whether the rendezvous port is free.
4. **It tells you why the run is bad.** `ThroughputMonitor` reports MFU,
   `BottleneckMonitor` says whether the loader is starving the GPU, and
   `TrainingMonitor` names the numerical failure (NaN loss, divergence, fp16
   overflow, vanishing grads) in plain language with the fix. Getting a
   straight answer to "why is this slow / broken" is usually the actual job.

It is also framework-plural: PyTorch, TensorFlow/Keras, scikit-learn, XGBoost,
and LightGBM go through one API, where most of the tools above are PyTorch-only.

## The core philosophy: Hardware vs. Math

Autotrainer maintains a strict separation between **hardware execution throughput** and **mathematical modeling recipe**:

| Dimension | What it Includes | How Autotrainer Treats It |
|---|---|---|
| **⚡ Hardware Throughput** | AMP (bf16/fp16), TF32, channels_last, Fused Optimizer Kernels, DataLoader Workers, Pin Memory, cuDNN Benchmark, DDP Spawning | **100% Automatic & Safe** — `prepare()` enables these out of the box because they accelerate compute execution without altering your loss function or model convergence. |
| **🧠 Algorithmic / Math** | Gradient Clipping, Learning Rate, Optimizer Choice, Weight Decay, Loss Scaling | **Non-Invasive by Default** — Autotrainer never silently modifies your mathematical hyperparameters. Instead, the **AI Training Doctor** (`TrainingMonitor`) inspects live tensor dynamics and calculates concrete remediation steps. |

## Reach for something else when

- **You're already happy on Lightning, Accelerate, or Ray.** Autotrainer doesn't
  integrate with them — it's an alternative to that layer, not an addition. If
  their abstractions already fit your work, switching buys you little.
- **You need multi-node beyond SLURM**, or a scheduler-agnostic cluster
  abstraction. Ray covers ground autotrainer doesn't.
- **You need heavy cloud-hosted model registries.** Autotrainer provides a local
  zero-dependency Web UI (`autotrainer ui`) and local file trackers (CSV, JSONL, run metadata), but does not run external hosted cloud services.
- **You need hyperparameter or architecture search.** Deliberately out of
  scope — the recipe and the model are yours. Reach for Optuna or Ray Tune,
  and use `prepare()` inside the objective.
- **You can't take pre-1.0 churn.** The public API has been frozen since 0.10,
  but this is 0.x and multi-node SLURM validation is still the open item before
  1.0 (see [Roadmap](roadmap.md)).

