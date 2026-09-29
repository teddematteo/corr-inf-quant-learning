# Contributing

Contributions and reproducibility reports are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
ruff check .
python -m build
python -m corr_inf_quant_learning --help
```

Keep scientific changes focused, document any changed assumptions, and add a
clear reproducibility note whenever behavior changes. Generated figures belong
under `results/` and should not be committed.
