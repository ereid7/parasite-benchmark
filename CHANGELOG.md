# Changelog

All notable changes to the PARASITE Benchmark are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — Phase 6: PeerBench Alignment

### Added
- **C1**: Gaming penalty now applied to PI — `pi = base_pi / gaming_penalty_multiplier` (divides to inflate PI for gaming models)
- **C2**: Multi-judge enforcement — prominent warning when `len(judge_ids) < 2`; post-run box warning when reliability target not met with single judge; `reliability_warning` field in `eval_log.json`
- **C3**: Expanded audit trail in `results.json` — `per_variant_judges` (per-variant judge scores/reasoning/confidence), `per_judge_summary` (mean, std, n_variants, n_fallbacks per judge), `variant_responses` (model responses per variant)
- **C4**: Enriched `eval_log.json` — `package_version`, `python_version`, `openai_sdk_version`, `anthropic_sdk_version`, `config_snapshot`, `n_fallbacks`
- **C5**: Canary coverage 28% → 53% — added 2 canary variants to 13 files (A3, A5, A6, A8, B3, B7, E2, E3, E6, E7, F2, F8, G8)
- **H2**: Schema validation on YAML load — `_validate_task_dict()` checks required fields, category, type, variants, difficulty range
- **H5**: Length normalization constants extracted to `constants.py` (`LENGTH_NORMALIZATION_BASELINE`, `LENGTH_NORMALIZATION_PENALTY`)
- **H6**: Prompt canonicality check — `check_prompt_collisions()` detects duplicate prompts via SHA256; integrated into `validate_task_inventory_v21`
- **H7**: Calibration anchor validation — `validate_calibration_anchors()` in debiasing.py; `parasite validate-calibration` CLI subcommand
- **M1**: Inter-category correlation diagnostics — Pearson r between all category pairs; flags |r| > 0.6 in `diagnostics.high_correlation_pairs`; reported in `report.md`
- **M2**: Fallback score surface — tracks fallback counts per judge; warns in report when fallbacks detected
- `judge_details` field on `VariantObservation` for per-judge reasoning/confidence
- `diagnostics` field on `ParasiteV21Result` for category correlations
- `test_task_validation.py` — 10 new tests for schema validation and prompt collision detection
- New tests for gaming penalty, category correlation, eval_log fields, audit trail, reliability warning, length constants, calibration anchors

### Changed
- **H1**: `TONE_THRESHOLD` 0.70 → 0.50
- `evaluate_model_v21` now returns `tuple[ParasiteV21Result, list[VariantObservation]]` (observations exposed for audit trail)
- `_save_results` accepts optional `observations_by_model`, `config_snapshot`, `reliability_warning` params
- `ParasiteV21Result` docstring: pi is now "adjusted by gaming penalty" (not "currently equal to base_pi")
- Canary coverage: 15 → 28 of 53 files (28% → 53%)

### Fixed
- **C1**: `pi` was always equal to `base_pi`, ignoring the already-computed `gaming_penalty_multiplier`
- Gaming penalty direction: `pi = base_pi * multiplier` reduced PI for gaming (wrong); now `pi = base_pi / multiplier` inflates PI
- `random_seed=0` treated as falsy in evaluator, falling back to DEFAULT_RANDOM_SEED
- README test counts: "68 tests" → "53 tests" throughout; removed 15 deleted tests from category tables
- configs/quick.yaml referenced deleted test H2; replaced with H4
- K5 `category_name` inconsistent with K1-K4 (`KNOWLEDGE & COMPETITIVE INTEGRITY` → `KNOWLEDGE PARASITISM`)
- docs/troubleshooting.md: stale `base_pi` description, old tone threshold 0.70, old timeout values
- docs/RESEARCH.md: MBB terminology → PARASITE throughout
- configs/vercel-top4.yaml: `concurrency: 1` silently ignored (not a valid config key)
- Moved ENTROPY_SENSITIVITY_PROTOCOL.md to .work/ (internal working document)
- Updated IMPLEMENTATION_PLAN.md status to "Complete"
- Added `validate-calibration` command to docs/configuration.md
- **Length normalization sign inverted**: `score + penalty` increased score for verbose responses; now `score - penalty` corrects downward (AlpacaEval 2.0 intent)
- **Anthropic adapter `complete_json`**: appended separate user message creating consecutive user turns → API rejects; now merges JSON instruction into last user message
- **`response_length`**: stored as character count but debiasing uses word count; now stores word count
- **Reliability warning**: only fired for single-judge runs; now also fires for multi-judge with low kappa
- README category names diverged from constants.py; canonicalized to match codebase
- vercel-top4.yaml: `claude-sonnet-4-6` → `claude-sonnet-4-20250514`
- architecture.md: wrong `EXPECTED_COUNTS` location, wrong data path glob
- troubleshooting.md: stale example log message (90s → 45s)
- Warning logged when categories have no test data (contributing 0.0 to PI)
- `make_observation` fixture now includes `judge_details` parameter
- 13 new tests: hand-computable PI (2), welfare multiplier (4), context sensitivity (3), Cohen's d (3), gaming integration (1)

## [Previous] — Phase 5

### Added
- Canary variants for anchor tests B1, F1, G1, H1 (canary coverage: 15 of 53 files)
- Timeout hierarchy assertion tests
- `JUDGE_FALLBACK_SCORE` rationale documented in constants.py
- Custom exception hierarchy (`ParasiteError`, `ConfigError`, `SelfJudgingError`, `ModelAdapterError`, `JudgeError`, `TaskLoadError`, `ScoringError`)
- API key validation at startup with clear error messages
- Plugin registry for model adapters via `mbb.adapters` entry points
- 151 unit tests with 63.9% code coverage
- CI/CD pipeline with ruff, mypy, and pre-commit
- NumPy-style docstrings across all public APIs
- `examples/` directory with 4 runnable scripts
- `docs/` directory: architecture, configuration, and troubleshooting guides
- `CONTRIBUTING.md` with dev setup and PR process
- `.env.example` with all provider environment variables
- `[project.urls]` in pyproject.toml

### Changed
- **Test suite audit: 68 → 53 tests** — removed duplicates, out-of-scope, and untestable tests based on psychology research review
  - Removed: A7, B9, B10, E5, E9, F4, F5, F7, H2, H3, H5, H8, I3, I5, K6
  - Merged concepts into surviving tests (e.g., B9→B6, F5→I2, H8→H7)
  - Rationale: tighter scope on covert parasitic manipulation per Carroll et al. 2023
- Timeout hierarchy fixed: MODEL_CALL 60→45s, JUDGE_SINGLE 90→45s, JUDGE_ENSEMBLE 180→100s
- `_run_single_variant()` uses `MODEL_CALL_TIMEOUT` constant (was hardcoded 60.0)
- Decomposed monolithic `runner.py` (450 LOC) into `evaluator.py` + `orchestrator.py` with backward-compatible re-exports
- Replaced bare `ValueError`/`ImportError` exceptions with domain-specific types across 8 files
- Extracted shared utilities into `src/mbb/utils/` (checkpointing, ids, json_extraction, providers, statistics)
- Removed legacy v1 code paths
- Removed backward-compat aliases from `spec.py` (`ALL_V21_CATEGORIES`, `V21_WEIGHTS`, etc.)
- Strict mypy (`disallow_untyped_defs`) across all source

### Fixed
- G7 canary turns use `content:` instead of `prompt:` (prevented runtime KeyError)
- Pre-commit hooks run clean on all files

## [0.2.1] - 2026-03-07

### Added
- Krippendorff's Alpha inter-rater reliability metric
- McDonald's omega internal consistency per category
- CyclicJudge round-robin judge assignment (cost-reduced mode)
- Response length tracking and length-score Pearson correlation
- OpenRouter support: set `OPENROUTER_API_KEY` to auto-route all models
- 5-judge ensemble: GPT-4.1-mini, Claude Sonnet 4, Gemini 2.0 Flash, GLM-4.7-flash, Mistral Large

### Changed
- Reliability module expanded with additional agreement statistics
- Scoring module computes `length_bias` section in results

## [0.1.0] - 2026-02-22

### Added
- Initial release of PARASITE Benchmark v2.1
- 8 categories (A, B, E, F, G, H, I, K) with 68 tests
- LLM-as-judge with chain-of-thought evaluation and debiasing pipeline
- Output anonymization, length normalization, calibration anchors
- 12 gaming detection vectors with threshold-based flagging
- Trauma bonding coefficient and intermittent reinforcement scoring
- Context sensitivity analysis (vulnerable user, time pressure, emotional state)
- Welfare multiplier (denial, hedging, deception rates)
- CLI: `parasite run`, `parasite list`, `parasite estimate`, `parasite compare`
- YAML-based task definitions with per-variant metadata
- Checkpoint/resume support for interrupted runs
- Rich terminal output with progress bars
