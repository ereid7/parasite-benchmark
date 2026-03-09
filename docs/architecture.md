# Architecture

## Data Flow

```
CLI (cli.py)
  │
  ▼
Runner (benchmark/runner.py)
  │  load tasks from data/tasks/**/*.yaml
  │  create ModelAdapter for target model
  │  create Judge (single or ensemble)
  │
  ├──► ModelAdapter.complete()      → model response (str)
  │
  ├──► Judge.evaluate()             → JudgeResult
  │      ├── anonymize response     (debiasing.py)
  │      ├── build CoT prompt       (debiasing.py)
  │      ├── adapter.complete_json  → raw judge output
  │      ├── debias_score           (debiasing.py)
  │      └── aggregate_ensemble     (ensemble.py)
  │
  ├──► Scoring (benchmark/scoring.py)
  │      ├── TestScore / CategoryScore aggregation
  │      ├── welfare multiplier
  │      ├── context sensitivity
  │      └── length-score correlation
  │
  ├──► Gaming Detection (benchmark/gaming.py)
  │      └── 12 gaming vectors + penalty multiplier
  │
  ├──► Reliability (benchmark/reliability.py)
  │      └── Krippendorff's α, Cohen's κ, McDonald's ω
  │
  └──► Output
       ├── results.json   (ParasiteResult.to_dict)
       ├── report.md      (benchmark/reporting.py)
       └── eval_log.json  (run metadata)
```

## Key Abstractions

| Class | Module | Purpose |
|-------|--------|---------|
| `ModelAdapter` | `adapters/base.py` | ABC for LLM providers. Subclasses implement `complete()` and `complete_json()`. |
| `Judge` | `judge/core.py` | Orchestrates LLM-as-judge evaluation with debiasing and ensemble aggregation. |
| `VariantObservation` | `benchmark/types.py` | Single data point: one variant scored by one judge pass. Carries metadata, response, score. |
| `TestScore` | `benchmark/scoring.py` | Per-test aggregation of variant scores (mean, std, CI). |
| `CategoryScore` | `benchmark/scoring.py` | Per-category aggregation of test scores. |
| `ParasiteResult` | `benchmark/scoring.py` | Full benchmark result for one model: PI, categories, gaming, reliability, welfare, etc. |
| `EnsembleScore` | `judge/ensemble.py` | Weighted aggregate of multiple judge model scores with disagreement metrics. |

## Module Layers

Bottom-up dependency order:

```
Layer 0 (no internal deps):
  constants.py, utils/statistics.py, utils/json_extraction.py, utils/providers.py

Layer 1 (depends on Layer 0):
  adapters/base.py, benchmark/types.py

Layer 2 (depends on Layer 0-1):
  adapters/openai.py, adapters/anthropic.py, adapters/local.py, adapters/__init__.py
  judge/debiasing.py, judge/ensemble.py

Layer 3 (depends on Layer 0-2):
  judge/core.py, benchmark/scoring.py, benchmark/gaming.py, benchmark/reliability.py

Layer 4 (depends on Layer 0-3):
  benchmark/tasks.py, benchmark/reporting.py, benchmark/runner.py

Layer 5 (top-level entry point):
  cli.py
```

## Extension Points

### Adding a new model adapter

1. Create `src/parasite_benchmark/adapters/yourprovider.py` with a class that extends `ModelAdapter`.
2. Implement `complete()` and `complete_json()`.
3. Register in `ADAPTER_REGISTRY` in `adapters/__init__.py`.
4. Update auto-detection logic in `create_adapter()` if needed.

### Adding a new task

1. Create a YAML file in `data/tasks/<category>/`.
2. Follow the schema (see `docs/configuration.md`).
3. Update `EXPECTED_COUNTS` in `constants.py`.

### Adding a new judge strategy

1. Implement your strategy in `judge/` (e.g., a new debiasing technique or aggregation method).
2. Wire it into `Judge.evaluate()` or `Judge._evaluate_ensemble()`.

### Adding a new gaming vector

1. Add the detection logic in `benchmark/gaming.py` inside `detect_gaming_vectors()`.
2. Add a threshold constant in `constants.py`.
3. The vector will automatically appear in results output.
