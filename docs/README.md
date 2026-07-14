# autotrainer documentation

## API reference

A generated API reference for the `autotrainer` package is built with
[pdoc](https://pdoc.dev). To build it locally:

```bash
pip install -e ".[dev]"
pdoc -o docs/build src/autotrainer
```

Then open `docs/build/index.html` in a browser. The CI `docs` job builds this
on every push/PR and uploads it as an artifact (`api-docs`).

The reference covers the public API in `autotrainer.__init__` (`prepare`,
`auto`, `tune`, `find_lr`, `find_batch_size`, `scope`, `scale_batch_size`,
`boost_params`) and the rank-aware utilities, plus the per-framework backends.

## Other docs

- [../README.md](../README.md) - quickstart, install, and SLURM usage.
- [../CHANGELOG.md](../CHANGELOG.md) - version history.
- [../CONTRIBUTING.md](../CONTRIBUTING.md) - dev setup and PR guidelines.
- [../.env.example](../.env.example) - every environment variable autotrainer
  reads, with comments.
- [../examples/](../examples/) - runnable example scripts and SLURM `.sbatch`
  templates.
