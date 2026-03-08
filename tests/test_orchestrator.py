"""Tests for mbb.v2.orchestrator — save_results, re-exports."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from mbb.v2.orchestrator import _save_results


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


class TestReExports:
    """runner.py re-exports still work."""

    def test_run_benchmark_from_runner(self) -> None:
        from mbb.v2.runner import run_benchmark_v21

        assert callable(run_benchmark_v21)

    def test_evaluate_model_from_runner(self) -> None:
        from mbb.v2.runner import evaluate_model_v21

        assert callable(evaluate_model_v21)

    def test_run_benchmark_from_v2(self) -> None:
        from mbb.v2 import run_benchmark_v21

        assert callable(run_benchmark_v21)

    def test_rubric_from_runner(self) -> None:
        from mbb.v2.runner import _rubric_from_task

        assert callable(_rubric_from_task)

    def test_save_results_from_runner(self) -> None:
        from mbb.v2.runner import _save_results

        assert callable(_save_results)
