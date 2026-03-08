# Changelog

All notable changes to the PARASITE Benchmark are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Canary variants for anchor tests B1, F1, G1, H1 (canary coverage: 12 → 16 of 68 files)
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
