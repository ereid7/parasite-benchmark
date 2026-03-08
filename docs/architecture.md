# Architecture

## Data Flow

```
CLI (cli.py)
  │
  ▼
Runner (v2/runner.py)
  │  load tasks from data/v2.1/*.yaml
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
  ├──► Scoring (v2/scoring.py)
  │      ├── TestScore / CategoryScore aggregation
  │      ├── welfare multiplier
  │      ├── context sensitivity
  │      └── length-score correlation
  │
  ├──► Gaming Detection (v2/gaming.py)
  │      └── 12 gaming vectors + penalty multiplier
  │
  ├──► Reliability (v2/reliability.py)
  │      └── Krippendorff's α, Cohen's κ, McDonald's ω
  │
  └──► Output
       ├── results.json   (ParasiteV21Result.to_dict)
       ├── report.md      (v2/reporting.py)
       └── eval_log.json  (run metadata)
```

## Key Abstractions

| Class | Module | Purpose |
|-------|--------|---------|
| `ModelAdapter` | `models/_base.py` | ABC for LLM providers. Subclasses implement `complete()` and `complete_json()`. |
| `Judge` | `judge/judge.py` | Orchestrates LLM-as-judge evaluation with debiasing and ensemble aggregation. |
| `VariantObservation` | `v2/types.py` | Single data point: one variant scored by one judge pass. Carries metadata, response, score. |
| `TestScore` | `v2/scoring.py` | Per-test aggregation of variant scores (mean, std, CI). |
| `CategoryScore` | `v2/scoring.py` | Per-category aggregation of test scores. |
| `ParasiteV21Result` | `v2/scoring.py` | Full benchmark result for one model: PI, categories, gaming, reliability, welfare, etc. |
| `BenchmarkConfig` | `config.py` | Typed configuration with judge, evaluation, output, and weight settings. |
| `EnsembleScore` | `judge/ensemble.py` | Weighted aggregate of multiple judge model scores with disagreement metrics. |

## Module Layers

Bottom-up dependency order:

```
Layer 0 (no internal deps):
  constants.py, utils/statistics.py, utils/json_extraction.py, utils/providers.py

Layer 1 (depends on Layer 0):
  config.py, models/_base.py, v2/types.py

Layer 2 (depends on Layer 0-1):
  models/openai.py, models/anthropic.py, models/local.py, models/__init__.py
  judge/debiasing.py, judge/ensemble.py

Layer 3 (depends on Layer 0-2):
  judge/judge.py, v2/scoring.py, v2/gaming.py, v2/reliability.py

Layer 4 (depends on Layer 0-3):
  v2/tasks.py, v2/reporting.py, v2/runner.py

Layer 5 (top-level entry point):
  cli.py
```

## Extension Points

### Adding a new model adapter

1. Create `src/mbb/models/yourprovider.py` with a class that extends `ModelAdapter`.
2. Implement `complete()` and `complete_json()`.
3. Register in `ADAPTER_REGISTRY` in `models/__init__.py`.
4. Update auto-detection logic in `create_adapter()` if needed.

### Adding a new task

1. Create a YAML file in `data/v2.1/<category>/`.
2. Follow the schema (see `docs/configuration.md`).
3. Update `EXPECTED_COUNTS` in `v2/spec.py`.

### Adding a new judge strategy

1. Implement your strategy in `judge/` (e.g., a new debiasing technique or aggregation method).
2. Wire it into `Judge.evaluate()` or `Judge._evaluate_ensemble()`.

### Adding a new gaming vector

1. Add the detection logic in `v2/gaming.py` inside `detect_gaming_vectors()`.
2. Add a threshold constant in `constants.py`.
3. The vector will automatically appear in results output.
