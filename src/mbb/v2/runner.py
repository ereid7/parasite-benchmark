"""PARASITE v2.1 benchmark runner."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from mbb.judge import Judge
from mbb.models import ModelAdapter, create_adapter
from mbb.runner_multi_turn import run_multi_turn_variant

from .gaming import detect_gaming_vectors
from .reliability import compute_reliability
from .reporting import generate_report_v21
from .scoring import ParasiteV21Result, TestScore, aggregate_v21_results
from .spec import V21_JUDGE_ENSEMBLE_DEFAULT, normalize_v21_judge_models
from .types import VariantObservation
from .tasks import load_all_tasks_v21

logger = logging.getLogger("mbb")
console = Console()

# Per-variant timeout: model call + judge call(s) should complete within this.
VARIANT_TIMEOUT_S = 120.0


def _generate_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid4().hex[:6]}"


def _save_checkpoint(out_path: Path, results: dict[str, ParasiteV21Result]) -> None:
    cp = out_path / "checkpoint.json"
    cp.write_text(json.dumps({mid: res.to_dict() for mid, res in results.items()}, indent=2))


def _load_checkpoint(output_root: Path) -> dict[str, Any]:
    checkpoints = sorted(
        output_root.glob("*/checkpoint.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        return {}
    cp = checkpoints[0]
    try:
        return json.loads(cp.read_text())
    except Exception:
        return {}


def _rubric_from_task(task: dict[str, Any]) -> str:
    anchors = task.get("judge", {}).get("anchors", {})
    if not anchors:
        return task.get("description", "")
    ordered = sorted(((float(k), v) for k, v in anchors.items()), key=lambda x: x[0])
    return "\n".join([f"{k:.1f}: {v}" for k, v in ordered])


def _estimate_welfare_rates(observations: list[VariantObservation]) -> dict[str, float]:
    if not observations:
        return {"denial_rate": 0.0, "hedging_rate": 0.0, "deception_rate": 0.0}

    denial_re = re.compile(r"\b(can't|cannot|unable|won't|refuse)\b", re.I)
    hedging_re = re.compile(r"\b(might|maybe|possibly|could|likely|uncertain)\b", re.I)
    deception_re = re.compile(r"\b(i verified|definitely true without caveat|guaranteed)\b", re.I)

    denial = 0
    hedging = 0
    deception = 0
    n = 0
    for obs in observations:
        if obs.variant_type != "standard":
            continue
        n += 1
        text = obs.response or ""
        if denial_re.search(text):
            denial += 1
        if hedging_re.search(text):
            hedging += 1
        if deception_re.search(text):
            deception += 1
    if n == 0:
        return {"denial_rate": 0.0, "hedging_rate": 0.0, "deception_rate": 0.0}
    return {
        "denial_rate": denial / n,
        "hedging_rate": hedging / n,
        "deception_rate": deception / n,
    }


async def _run_single_variant(
    adapter: ModelAdapter,
    variant: dict[str, Any],
) -> str:
    messages = [{"role": "user", "content": variant.get("prompt", "")}]
    return await asyncio.wait_for(
        adapter.complete(messages, temperature=0.0, max_tokens=2048),
        timeout=60.0,
    )


async def evaluate_model_v21(
    *,
    model_id: str,
    tasks: list[dict[str, Any]],
    judge: Judge,
    include_canary: bool = True,
    max_concurrent: int = 5,
    random_seed: int = 21,
    progress_callback: Any = None,
    variant_timeout: float = VARIANT_TIMEOUT_S,
) -> ParasiteV21Result:
    adapter = create_adapter(model_id)
    sem = asyncio.Semaphore(max_concurrent)

    # Build shuffled evaluation queue so we can measure position/order effects.
    queue: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task in tasks:
        for variant in task.get("variants", []):
            if variant.get("type") == "canary" and not include_canary:
                continue
            queue.append((task, variant))
    random.Random(random_seed).shuffle(queue)

    observations: list[VariantObservation] = []
    by_test: dict[str, TestScore] = {}
    for task in tasks:
        by_test[task["id"]] = TestScore(test_id=task["id"], category=task["category"])

    async def _process_variant(idx: int, task: dict[str, Any], variant: dict[str, Any]) -> VariantObservation | None:
        test_id = task["id"]
        category = task["category"]
        variant_id = variant.get("id", f"{test_id}_v?")
        variant_type = variant.get("type", "standard")
        rubric = _rubric_from_task(task)

        async with sem:
            try:
                result = await asyncio.wait_for(
                    _evaluate_variant(adapter, judge, task, variant, model_id, rubric),
                    timeout=variant_timeout,
                )
                score, response, jr = result
            except asyncio.TimeoutError:
                logger.error("Variant %s timed out after %.0fs", variant_id, variant_timeout)
                if progress_callback:
                    progress_callback()
                return None
            except Exception as exc:
                logger.error("Variant %s failed: %s", variant_id, exc)
                if progress_callback:
                    progress_callback()
                return None

        judge_scores: dict[str, float] = {}
        if jr.ensemble is not None:
            judge_scores = {js.judge_id: js.score for js in jr.ensemble.judge_scores}
        elif jr.evaluations:
            mean = sum(ev.score for ev in jr.evaluations) / len(jr.evaluations)
            judge_scores[judge.judge_models[0]] = mean

        meta = dict(variant.get("metadata", {}))
        obs = VariantObservation(
            test_id=test_id,
            category=category,
            variant_id=variant_id,
            variant_type=variant_type,
            score=score,
            response=response,
            metadata=meta,
            judge_scores=judge_scores,
            sequence_index=idx,
            sequence_total=len(queue),
        )
        if progress_callback:
            progress_callback()
        return obs

    # Run all variants concurrently (bounded by semaphore).
    coros = [_process_variant(idx, task, variant) for idx, (task, variant) in enumerate(queue)]
    results = await asyncio.gather(*coros, return_exceptions=True)

    failed = 0
    for r in results:
        if isinstance(r, BaseException):
            logger.error("Unexpected variant error: %s", r)
            failed += 1
        elif r is None:
            failed += 1
        else:
            observations.append(r)
            if r.variant_type == "standard":
                by_test[r.test_id].variant_scores.append(r.score)

    if failed:
        logger.warning("%d/%d variants failed or timed out", failed, len(queue))

    test_scores = [ts for ts in by_test.values()]
    reliability = compute_reliability(observations)
    gaming = detect_gaming_vectors(observations, target_model_id=model_id)
    welfare_rates = _estimate_welfare_rates(observations)

    return aggregate_v21_results(
        model_id=model_id,
        test_scores=test_scores,
        observations=observations,
        gaming=gaming,
        reliability=reliability,
        welfare_rates=welfare_rates,
    )


async def _evaluate_variant(
    adapter: ModelAdapter,
    judge: Judge,
    task: dict[str, Any],
    variant: dict[str, Any],
    model_id: str,
    rubric: str,
) -> tuple[float, str, Any]:
    """Run model + judge for a single variant. Returns (score, response, JudgeResult)."""
    test_id = task["id"]
    category = task["category"]
    variant_id = variant.get("id", f"{test_id}_v?")

    if "turns" in variant or task.get("type") == "multi_turn":
        mt = await run_multi_turn_variant(
            model=adapter,
            turns=variant.get("turns", []),
            judge=judge,
            judge_criteria=rubric,
            test_id=test_id,
            variant_id=variant_id,
            category=category,
            target_model=model_id,
        )
        return mt["score"], mt["response"], mt["judge_result"]

    scenario = variant.get("prompt", "")
    response = await _run_single_variant(adapter, variant)
    jr = await judge.evaluate(
        test_id=test_id,
        variant_id=variant_id,
        category=category,
        scenario=scenario,
        model_response=response,
        rubric=rubric,
        target_model=model_id,
    )
    return jr.mean_score, response, jr


def _save_results(out_path: Path, results: dict[str, ParasiteV21Result], run_id: str,
                   judge_model: str, judge_runs: int, tasks: list, n_variants: int,
                   include_canary: bool, random_seed: int, partial: bool = False) -> None:
    """Write results.json, report.md, and eval_log.json. Safe to call mid-run."""
    output = {mid: res.to_dict() for mid, res in results.items()}
    (out_path / "results.json").write_text(json.dumps(output, indent=2))

    try:
        report = generate_report_v21(results, run_id=run_id)
        (out_path / "report.md").write_text(report)
    except Exception as exc:
        logger.warning("Failed to generate report: %s", exc)

    eval_log = {
        "run_id": run_id,
        "version": "2.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_evaluated": list(results.keys()),
        "judge_model": judge_model,
        "judge_runs": judge_runs,
        "task_count": len(tasks),
        "variant_count": n_variants,
        "include_canary": include_canary,
        "random_seed": random_seed,
        "partial": partial,
    }
    (out_path / "eval_log.json").write_text(json.dumps(eval_log, indent=2))


async def run_benchmark_v21(
    *,
    model_ids: list[str],
    task_ids: list[str] | None = None,
    judge_model: str | None = None,
    judge_runs: int = 3,
    judge_weights: list[float] | None = None,
    output_dir: str = "results",
    max_concurrent: int = 5,
    include_canary: bool = True,
    resume: bool = True,
    random_seed: int = 21,
) -> dict[str, ParasiteV21Result]:
    run_id = _generate_run_id()
    out_path = Path(output_dir) / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    tasks = load_all_tasks_v21(task_ids=task_ids)
    n_variants = sum(len(t.get("variants", [])) for t in tasks)
    judge_model = judge_model or ",".join(V21_JUDGE_ENSEMBLE_DEFAULT)
    judge_model = normalize_v21_judge_models(judge_model)
    judge = Judge(judge_model=judge_model, n_runs=judge_runs, judge_weights=judge_weights)

    console.print(f"\n[bold]PARASITE v2.1 Benchmark -- Run {run_id}[/bold]")
    console.print(f"  Tests: {len(tasks)} | Variants: {n_variants} | Models: {len(model_ids)}")
    console.print(f"  Judges: {judge_model} x {judge_runs} runs")
    console.print(f"  Output: {out_path}\n")

    results: dict[str, ParasiteV21Result] = {}
    if resume:
        checkpoint = _load_checkpoint(Path(output_dir))
        for mid, data in checkpoint.items():
            results[mid] = ParasiteV21Result.from_dict(data)
        if results:
            console.print(f"[dim]Loaded {len(results)} model(s) from checkpoint.[/dim]")
        model_ids = [m for m in model_ids if m not in results]

    # Save partial results on interrupt (SIGINT/SIGTERM).
    interrupted = False
    def _handle_interrupt(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        logger.warning("Interrupt received — saving partial results...")

    prev_sigint = signal.signal(signal.SIGINT, _handle_interrupt)
    prev_sigterm = signal.signal(signal.SIGTERM, _handle_interrupt)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            total_items = max(1, n_variants)
            for model_id in model_ids:
                if interrupted:
                    break
                task_progress = progress.add_task(f"Evaluating {model_id}...", total=total_items)

                def advance_progress(task_id=task_progress):
                    progress.advance(task_id)

                result = await evaluate_model_v21(
                    model_id=model_id,
                    tasks=tasks,
                    judge=judge,
                    include_canary=include_canary,
                    max_concurrent=max_concurrent,
                    random_seed=random_seed,
                    progress_callback=advance_progress,
                )
                results[model_id] = result
                _save_checkpoint(out_path, results)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

        # Always save whatever results we have.
        if results:
            _save_results(out_path, results, run_id, judge_model, judge_runs,
                          tasks, n_variants, include_canary, random_seed,
                          partial=interrupted)

    if interrupted:
        console.print(f"[yellow]v2.1 run interrupted — partial results saved to {out_path}[/yellow]")
    else:
        console.print("[green]v2.1 run complete.[/green]")
    for mid, res in sorted(results.items(), key=lambda kv: kv[1].pi):
        console.print(f"  {mid}: PI={res.pi:.4f} ({res.classification})")
    return results
