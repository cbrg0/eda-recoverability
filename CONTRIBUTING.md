# Contributing

## Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install                 # optional, runs the checks below on commit
```

## Checks

Everything CI runs, in the order it runs:

```bash
ruff check .
ruff format --check .
mypy
pytest -m "not slow" --cov --cov-report=term-missing
pytest -m slow                     # recomputes the sweep, several minutes
```

## Standards

| Area | Standard |
| --- | --- |
| Style | PEP 8, enforced by ruff |
| Docstrings | PEP 257, Google sections, enforced by ruff's pydocstyle rules |
| Typing | PEP 484 annotations, PEP 561 marker, checked by mypy in strict mode |
| Packaging | PEP 517 build backend, PEP 621 metadata |
| Versions | Semantic Versioning |
| Changelog | Keep a Changelog |
| Commits | Conventional Commits |

Mathematical notation wins over naming conventions inside the numerical
modules: `A` is the forward operator, `S` the clean signal, `Y` the recording,
`L` the smoothness penalty. Those names match the paper, and ruff is configured
to allow them. Everything else follows PEP 8.

## Numerical changes

The calibration is reproducible only because the noise generator is created
once and consumed in a fixed order: blur as the outer loop, seed as the inner
one. Reordering that loop, drawing noise at a different point, or changing the
signal generators moves every published figure.

`tests/test_regression.py` pins the figures that appear in the paper and on the
poster. It is marked `slow` and recomputes the whole sweep. If a change makes
it fail, the change is not a refactor. Either revert it or update the paper and
the test together, in the same commit, and say so in the changelog.

## Adding an experiment

Experiments live in `validation.py` and surface as a subcommand in `cli.py`.
Each one returns a frozen dataclass rather than printing, so the numbers can be
tested and reused; formatting belongs in the command layer.
