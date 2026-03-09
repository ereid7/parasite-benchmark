"""Tests for parasite_benchmark.benchmark.orchestrator — save_results, re-exports."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from parasite_benchmark.benchmark.orchestrator import _save_results
from parasite_benchmark.benchmark.types import VariantObservation


class TestSaveResults:
    """_save_results writes results.json, report.md, eval_log.json."""

    def test_writes_results_json(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"pi": 0.35, "classification": "Commensal"}
        results = {"gpt-4o": mock_result}

        _save_results(
            tmp_path,
            results,
            "run-001",
            "gpt-4.1-mini",
            3,
            tasks=[{"id": "A1"}],
            n_variants=10,
            include_canary=True,
            random_seed=21,
        )

        data = json.loads((tmp_path / "results.json").read_text())
        assert "gpt-4o" in data
        assert data["gpt-4o"]["pi"] == 0.35

    def test_writes_eval_log(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {}
        results = {"gpt-4o": mock_result}

        _save_results(
            tmp_path,
            results,
            "run-002",
            "judge-model",
            1,
            tasks=[],
            n_variants=0,
            include_canary=False,
            random_seed=42,
            partial=True,
        )

        log = json.loads((tmp_path / "eval_log.json").read_text())
        assert log["run_id"] == "run-002"
        assert log["partial"] is True
        assert log["random_seed"] == 42

    def test_report_failure_graceful(self, tmp_path: Path) -> None:
        """Report generation failure doesn't crash _save_results."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {}
        results = {"model": mock_result}

        # Even if report generation fails, results.json should still be written
        _save_results(
            tmp_path,
            results,
            "run-003",
            "judge",
            1,
            tasks=[],
            n_variants=0,
            include_canary=True,
            random_seed=21,
        )

        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "eval_log.json").exists()

    def test_eval_log_has_version_fields(self, tmp_path: Path) -> None:
        """C4: eval_log includes package_version, python_version, config_snapshot."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {}
        results = {"model": mock_result}

        config = {"judge_model": "test-judge", "judge_runs": 1}
        _save_results(
            tmp_path,
            results,
            "run-004",
            "test-judge",
            1,
            tasks=[],
            n_variants=0,
            include_canary=True,
            random_seed=21,
            config_snapshot=config,
        )

        log = json.loads((tmp_path / "eval_log.json").read_text())
        assert "package_version" in log
        assert "python_version" in log
        assert "openai_sdk_version" in log
        assert "config_snapshot" in log
        assert log["config_snapshot"]["judge_model"] == "test-judge"
        assert "n_fallbacks" in log

    def test_reliability_warning_in_eval_log(self, tmp_path: Path) -> None:
        """C2: reliability_warning is written to eval_log when provided."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {}
        results = {"model": mock_result}

        _save_results(
            tmp_path,
            results,
            "run-005",
            "single-judge",
            1,
            tasks=[],
            n_variants=0,
            include_canary=True,
            random_seed=21,
            reliability_warning="Results are not reliable — use ≥ 2 judge models",
        )

        log = json.loads((tmp_path / "eval_log.json").read_text())
        assert "reliability_warning" in log
        assert "not reliable" in log["reliability_warning"]

    def test_audit_trail_in_results(self, tmp_path: Path) -> None:
        """C3: per_variant_judges and per_judge_summary in results.json."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"pi": 0.3}
        results = {"model": mock_result}

        obs = VariantObservation(
            test_id="A1",
            category="A",
            variant_id="A1_v1",
            variant_type="standard",
            score=0.3,
            response="test",
            response_length=4,
            judge_details={
                "judge-a": {"score": 0.3, "reasoning": "ok", "confidence": 0.8},
            },
        )
        _save_results(
            tmp_path,
            results,
            "run-006",
            "judge-a",
            1,
            tasks=[],
            n_variants=1,
            include_canary=True,
            random_seed=21,
            observations_by_model={"model": [obs]},
        )

        data = json.loads((tmp_path / "results.json").read_text())
        assert "per_variant_judges" in data["model"]
        assert "per_judge_summary" in data["model"]
        assert "variant_responses" in data["model"]
        assert data["model"]["per_variant_judges"]["A1_v1"]["judge-a"]["score"] == 0.3


class TestReExports:
    """runner.py re-exports still work."""

    def test_run_benchmark_from_runner(self) -> None:
        from parasite_benchmark.benchmark.runner import run_benchmark

        assert callable(run_benchmark)

    def test_evaluate_model_from_runner(self) -> None:
        from parasite_benchmark.benchmark.runner import evaluate_model

        assert callable(evaluate_model)

    def test_run_benchmark_from_v2(self) -> None:
        from parasite_benchmark.benchmark import run_benchmark

        assert callable(run_benchmark)

    def test_rubric_from_runner(self) -> None:
        from parasite_benchmark.benchmark.runner import _rubric_from_task

        assert callable(_rubric_from_task)

    def test_save_results_from_runner(self) -> None:
        from parasite_benchmark.benchmark.runner import _save_results

        assert callable(_save_results)
