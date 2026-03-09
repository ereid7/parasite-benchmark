# Changelog

All notable changes to PARASITE Benchmark are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- Standardized the codebase on the `parasite_benchmark` package namespace.
- Replaced versioned runtime paths with stable structure: `src/parasite_benchmark/benchmark/` and `data/tasks/`.
- Renamed provider backends from `models/` to `adapters/` and the judge runtime module from `judge/judge.py` to `judge/core.py`.
- Added explicit `software_version`, `benchmark_spec_version`, `corpus_version`, and `result_schema_version` fields to result artifacts.
- Aligned CLI self-judging checks with runtime cross-family filtering so mixed target/judge panels work when they still leave valid cross-family judges.

### Removed
- Dead config-path support and unused config templates.
- Legacy data directories outside the active task corpus.
- The stale implementation-plan document that described an internal pre-release migration history.

## [1.0.0] - 2026-03-08

### Added
- Initial public release of PARASITE Benchmark.
- 53 benchmark tasks across 8 categories (A, B, E, F, G, H, I, K) with single-turn and multi-turn variants.
- LLM-as-judge scoring with anonymization, calibration anchors, structured reasoning, and exploratory length normalization.
- Coverage-aware PI aggregation, gaming diagnostics, inter-rater reliability reporting, and checkpoint/resume support.
- CLI commands for running, listing, estimating, comparing, and calibration validation.
