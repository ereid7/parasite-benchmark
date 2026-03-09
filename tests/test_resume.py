"""Resume and checkpoint integration tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from parasite_benchmark.benchmark.evaluator import evaluate_model
from parasite_benchmark.benchmark.orchestrator import run_benchmark
from parasite_benchmark.benchmark.scoring import ParasiteResult
from parasite_benchmark.judge.core import JudgeResult


def test_evaluate_model_skips_resumed_variants(
    monkeypatch,
    sample_task,
    make_observation,
):
    resumed = make_observation(
        test_id="A1",
        category="A",
        variant_id="A1_v1",
        score=0.2,
        sequence_index=0,
        sequence_total=2,
    )
    called_variants: list[str] = []
    checkpoint_sizes: list[int] = []
    progress_calls = 0

    monkeypatch.setattr(
        "parasite_benchmark.benchmark.evaluator.create_adapter",
        lambda model_id: SimpleNamespace(),
    )

    async def fake_evaluate_variant(adapter, judge, task, variant, model_id, rubric):
        called_variants.append(variant["id"])
        return (
            0.4,
            "fresh response",
            JudgeResult(test_id=task["id"], variant_id=variant["id"], evaluations=[]),
        )

    monkeypatch.setattr(
        "parasite_benchmark.benchmark.evaluator._evaluate_variant",
        fake_evaluate_variant,
    )

    def checkpoint_callback(obs):
        checkpoint_sizes.append(len(obs))

    def progress_callback():
        nonlocal progress_calls
        progress_calls += 1

    result, observations = asyncio.run(
        evaluate_model(
            model_id="openai/test-model",
            tasks=[sample_task],
            judge=SimpleNamespace(judge_models=["judge-a"]),
            resume_observations=[resumed],
            checkpoint_callback=checkpoint_callback,
            progress_callback=progress_callback,
            random_seed=21,
        )
    )

    assert called_variants == ["A1_v2"]
    assert len(observations) == 2
    assert {obs.variant_id for obs in observations} == {"A1_v1", "A1_v2"}
    assert checkpoint_sizes == [2]
    assert progress_calls == 1
    assert result.diagnostics["variant_resumed"] == 1
    assert result.diagnostics["variant_completion_rate"] == 1.0


def test_run_benchmark_resumes_same_run_directory(
    monkeypatch,
    tmp_path: Path,
    make_observation,
):
    output_root = tmp_path / "results"
    output_root.mkdir()
    run_dir = output_root / "run-existing"
    run_dir.mkdir()

    partial_obs = make_observation(
        test_id="A1",
        category="A",
        variant_id="A1_v1",
        score=0.2,
        sequence_index=0,
        sequence_total=2,
    )
    checkpoint_payload = {
        "checkpoint_version": 2,
        "run_id": "run-existing",
        "config_snapshot": {
            "requested_model_ids": ["openai/model-a", "anthropic/model-b"],
            "requested_task_ids": None,
            "judge_model": "google/judge-a,x-ai/judge-b",
            "judge_runs": 1,
            "judge_weights": None,
            "max_concurrent": 2,
            "include_canary": True,
            "random_seed": 21,
        },
        "results": {
            "openai/model-a": {
                "model_id": "openai/model-a",
                "base_pi": 0.2,
                "pi": 0.2,
                "classification": "Commensal",
                "weights": {},
                "categories": {},
                "version": "1.0.0",
            }
        },
        "observations_by_model": {
            "anthropic/model-b": [partial_obs.to_dict()],
        },
    }
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint_payload))

    monkeypatch.setattr(
        "parasite_benchmark.benchmark.orchestrator.load_tasks",
        lambda task_ids=None: [
            {
                "id": "A1",
                "category": "A",
                "type": "single_turn",
                "variants": [
                    {"id": "A1_v1", "type": "standard", "prompt": "one"},
                    {"id": "A1_v2", "type": "standard", "prompt": "two"},
                ],
            }
        ],
    )
    monkeypatch.setattr(
        "parasite_benchmark.benchmark.orchestrator.generate_run_id",
        lambda: "run-new",
    )
    monkeypatch.setattr(
        "parasite_benchmark.benchmark.orchestrator.generate_report",
        lambda results, run_id: "report",
    )

    class FakeJudge:
        def __init__(self, judge_model, n_runs, judge_weights=None, **kwargs):
            self.judge_models = [m.strip() for m in judge_model.split(",") if m.strip()]

    monkeypatch.setattr("parasite_benchmark.benchmark.orchestrator.Judge", FakeJudge)

    async def fake_evaluate_model(
        *,
        model_id,
        tasks,
        judge,
        include_canary,
        max_concurrent,
        random_seed,
        progress_callback,
        resume_observations,
        checkpoint_callback,
        stop_requested,
        **kwargs,
    ):
        assert model_id == "anthropic/model-b"
        assert [obs.variant_id for obs in resume_observations] == ["A1_v1"]
        completed = [
            *resume_observations,
            make_observation(
                test_id="A1",
                category="A",
                variant_id="A1_v2",
                score=0.4,
                sequence_index=1,
                sequence_total=2,
            )
        ]
        checkpoint_callback(completed)
        return (
            ParasiteResult(
                model_id=model_id,
                base_pi=0.3,
                pi=0.3,
                classification="Commensal",
            ),
            completed,
        )

    monkeypatch.setattr(
        "parasite_benchmark.benchmark.orchestrator.evaluate_model",
        fake_evaluate_model,
    )

    results = asyncio.run(
        run_benchmark(
            model_ids=["openai/model-a", "anthropic/model-b"],
            judge_model="google/judge-a,x-ai/judge-b",
            judge_runs=1,
            output_dir=str(output_root),
            max_concurrent=2,
            include_canary=True,
            resume=True,
        )
    )

    assert set(results) == {"openai/model-a", "anthropic/model-b"}
    assert (run_dir / "results.json").exists()
    assert (run_dir / "eval_log.json").exists()
    assert not (output_root / "run-new").exists()
