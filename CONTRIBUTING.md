# Contributing

## Dev Setup

```bash
git clone https://github.com/parasite-benchmark/parasite.git
cd parasite
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
pytest tests/ -v                    # all tests
pytest tests/ -v --cov=mbb         # with coverage
pytest tests/test_scoring.py -v    # single module
```

## Code Style

- **Linter/formatter:** ruff (select: E, F, I, N, W, UP, B, C90, S, A, RUF)
- **Type checker:** mypy with `disallow_untyped_defs`
- **Docstrings:** NumPy-style for all public functions and classes
- **Line length:** 100 characters

Run checks locally before pushing:

```bash
ruff check src/ tests/ examples/
ruff format --check src/ tests/ examples/
mypy src/
```

## Adding a Model Adapter

1. Create `src/mbb/models/yourprovider.py`.
2. Subclass `ModelAdapter` and implement `complete()` and `complete_json()`.
3. Register a factory function in `ADAPTER_REGISTRY` (in `models/__init__.py`).
4. If auto-detection is needed, update `create_adapter()` logic.

```python
from mbb.models._base import ModelAdapter

class YourAdapter(ModelAdapter):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        # initialize your client

    async def complete(self, messages, temperature=0.0, max_tokens=2048):
        # return plain text
        ...

    async def complete_json(self, messages, temperature=0.0, max_tokens=2048):
        # return parsed dict
        ...
```

## Adding a Task

1. Create a YAML file in `data/v2.1/<category>/` following the task file schema (see `docs/configuration.md`).
2. Include at least 10 `standard` variants plus canary/adversarial variants.
3. Update `EXPECTED_COUNTS` in `src/mbb/v2/spec.py` to reflect the new task count.
4. Verify: `parasite list tasks` should show your new task.

## PR Process

1. Create a feature branch from `main`.
2. Make your changes.
3. Ensure all checks pass:
   ```bash
   ruff check src/ tests/ examples/
   ruff format --check src/ tests/ examples/
   mypy src/
   pytest tests/ -v
   pre-commit run --all-files
   ```
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Open a pull request with a clear description.
