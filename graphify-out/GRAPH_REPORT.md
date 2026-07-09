# Graph Report - .  (2026-07-09)

## Corpus Check
- Corpus is ~6,103 words - fits in a single context window. You may not need a graph.

## Summary
- 189 nodes · 286 edges · 20 communities (11 shown, 9 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 20 edges (avg confidence: 0.8)
- Token cost: 33,203 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]

## God Nodes (most connected - your core abstractions)
1. `detect()` - 14 edges
2. `auto()` - 10 edges
3. `_infer_loss()` - 9 edges
4. `find_lr()` - 9 edges
5. `tune()` - 9 edges
6. `to_device()` - 9 edges
7. `robust_forward()` - 9 edges
8. `run_doctor()` - 8 edges
9. `_make_optimizer()` - 7 edges
10. `boost_params()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `autotrainer.auto()` --conceptually_related_to--> `autocast_context() mixed precision`  [INFERRED]
  README.md → CHANGELOG.md
- `Backend contribution pattern` --rationale_for--> `Backend dispatcher / routing`  [INFERRED]
  CONTRIBUTING.md → CHANGELOG.md
- `autotrainer.prepare()` --conceptually_related_to--> `PyTorch backend (DDP + DistributedSampler)`  [INFERRED]
  README.md → CHANGELOG.md
- `autotrainer.auto()` --conceptually_related_to--> `autotrainer.find_lr() (LR range test)`  [EXTRACTED]
  README.md → CHANGELOG.md
- `autotrainer.auto()` --conceptually_related_to--> `Loss function inference (CrossEntropy/BCE/MSE/Huber)`  [EXTRACTED]
  README.md → CHANGELOG.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Framework backends routed by the dispatcher** — changelog_pytorch_backend, changelog_sklearn_backend, changelog_xgboost_lightgbm_backend, changelog_tf_scope, changelog_dispatcher [INFERRED 0.85]
- **auto() training-recipe inference flow** — readme_auto, changelog_find_lr, changelog_loss_inference, changelog_mixed_precision [INFERRED 0.85]

## Communities (20 total, 9 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (33): device, auto(), find_lr(), _infer_loss(), _looks_like_cnn(), _make_loss(), _make_optimizer(), _param_groups() (+25 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (22): main(), CLI: `autotrainer run train.py [args...]` and `autotrainer info`., detect(), Environment, _gpu_count(), Hardware and cluster environment detection.  Detection hierarchy:     1. SLURM e, Count GPUs without importing torch (works pre-install too)., First hostname in the SLURM node list is the rendezvous master. (+14 more)

### Community 2 - "Community 2"
Cohesion: 0.14
Nodes (19): _is_sklearn_estimator(), _is_torch_module(), prepare(), autotrainer: automatic distributed training.  Usage inside a training script (on, Framework dispatcher: route to the right backend by model type., autocast_context(), barrier(), GradScaler() (+11 more)

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (20): CI publish job (trusted PyPI publishing), CI test job (pytest matrix), Backend dispatcher / routing, autotrainer.find_lr() (LR range test), Loss function inference (CrossEntropy/BCE/MSE/Huber), autocast_context() mixed precision, Optuna hyperparameter search (TPE + median pruning), PyTorch backend (DDP + DistributedSampler) (+12 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (7): clean_env(), Test suite. Run with: pytest tests/ -v, Strip SLURM/rank vars so tests don't leak into each other., TestBoostParams, TestCLI, TestDispatcher, TestUtils

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (13): build_tf_config(), TensorFlow/Keras backend.  TF requires models to be *created inside* a strategy, Build the TF_CONFIG dict for MultiWorkerMirroredStrategy from SLURM vars.      O, Return the right tf.distribute strategy scope for this environment., TF splits the *global* batch across replicas; scale accordingly., scale_batch_size(), scope(), _slurm_hostnames() (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.21
Nodes (12): boost_params(), prepare(), Gradient boosting backend (XGBoost, LightGBM).  Tree libraries have no batch siz, Configure thread count on an XGBoost/LightGBM estimator (in place)., Return a params dict with the right thread key set, for native APIs.      Exampl, _warn_if_multinode(), _available_cpus(), prepare() (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.33
Nodes (6): _dist_info(), find_batch_size(), prepare(), PyTorch backend.  Called from user code via `autotrainer.prepare(...)`. Reads th, Make (model, dataloader, optimizer) distribution-ready.      Single device: retu, Double batch size until OOM, then back off one step.      sample_batch_fn(bs) mu

### Community 8 - "Community 8"
Cohesion: 0.29
Nodes (3): Recursively slice batch data structures to the first n samples., slice_batch(), TestPytorchEnhancements

## Knowledge Gaps
- **9 isolated node(s):** `autotrainer`, `autotrainer run CLI launcher`, `SLURM multi-node support`, `Optuna hyperparameter search (TPE + median pruning)`, `autocast_context() mixed precision` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TestPytorchEnhancements` connect `Community 8` to `Community 2`, `Community 4`?**
  _High betweenness centrality (0.262) - this node is a cross-community bridge._
- **Why does `to_device()` connect `Community 0` to `Community 8`, `Community 2`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `detect()` (e.g. with `.test_single_mode_default()` and `.test_slurm_cpu_only_falls_back_to_ntasks()`) actually correct?**
  _`detect()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Smart defaults: loss, optimizer, LR, and schedule inferred automatically. Run: a`, `Minimal example: a tiny model on random data, launched via `autotrainer run`.`, `Hyperparameter search: find the best training recipe for YOUR model. Run: python` to the rest of the system?**
  _64 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.0915915915915916 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.12183908045977011 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._