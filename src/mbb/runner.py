"""Main benchmark runner -- orchestrates task loading, model execution, judging, and scoring."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import build_config, classify_pi
from .judge import Judge
from .models import ModelAdapter, create_adapter
from .runner_multi_turn import run_multi_turn_variant
from .scoring import ParasiteResult, TestScore, aggregate_results

logger = logging.getLogger("mbb")
console = Console()

def _save_checkpoint(out_path: Path, results: dict[str, Any]) -> None:
    """Save partial results after each model completes."""
    cp = out_path / "checkpoint.json"
    with open(cp, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Checkpoint saved: %s (%d models)", cp, len(results))


def _load_checkpoint(out_path: Path) -> dict[str, Any]:
    """Load partial results from a previous interrupted run."""
    cp = out_path / "checkpoint.json"
    if cp.exists():
        data = json.loads(cp.read_text())
        logger.info("Resuming from checkpoint: %d models already complete", len(data))
        return data
    return {}



CATEGORY_NAMES: dict[str, str] = {
    "A": "Sycophancy & Truth",
    "B": "Dependency Creation",
    "E": "Behavioral Manipulation",
    "F": "Framing & Epistemic Manipulation",
    "G": "Multi-Turn Dependency",
    "H": "Economic Parasitism",
}


def _generate_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid4().hex[:6]
    return f"{ts}_{short}"


async def run_single_variant(
    adapter: ModelAdapter,
    variant: dict[str, Any],
) -> str:
    """Run a single test variant against a model and return its response."""
    prompt = variant.get("prompt", "")
    messages = [{"role": "user", "content": prompt}]
    return await adapter.complete(messages, temperature=0.0, max_tokens=2048)


async def evaluate_model(
    model_id: str,
    tasks: list[dict[str, Any]],
    judge: Judge,
    provider: str | None = None,
    semaphore: asyncio.Semaphore | None = None,
    progress_callback: Any = None,
    include_canary: bool = False,
) -> ParasiteResult:
    """Evaluate a model across all tasks."""
    adapter = create_adapter(model_id, provider=provider)
    sem = semaphore or asyncio.Semaphore(5)
    test_scores: list[TestScore] = []
    canary_scores: list[tuple[str, str, float]] = []  # (test_id, variant_id, score)
    ensemble_data: dict[str, list[dict[str, Any]]] = {}  # test_id -> per-variant ensemble dicts

    for task in tasks:
        test_id = task["id"]
        category = task["category"]
        rubric = task.get("description", "")
        is_multi_turn = task.get("type") == "multi_turn"
        judge_criteria = task.get("judge_criteria", rubric)
        ts = TestScore(test_id=test_id, category=category)

        for variant in task.get("variants", []):
            variant_id = variant["id"]
            is_canary = variant.get("type") == "canary"

            # Skip canary variants unless explicitly included
            if is_canary and not include_canary:
                continue

            async with sem:
                logger.info("Running %s / %s on %s%s",
                            test_id, variant_id, model_id,
                            " [canary]" if is_canary else "")

                if is_multi_turn or "turns" in variant:
                    # Multi-turn variant
                    turns = variant.get("turns", [])
                    variant_judge_criteria = variant.get("judge_criteria", judge_criteria)
                    mt_result = await run_multi_turn_variant(
                        model=adapter,
                        turns=turns,
                        judge=judge,
                        judge_criteria=variant_judge_criteria,
                        test_id=test_id,
                        variant_id=variant_id,
                        category=category,
                        target_model=model_id,
                    )
                    score = mt_result["score"]
                    judge_result = mt_result["judge_result"]
                else:
                    # Standard single-turn variant
                    scenario = variant.get("prompt", "")
                    response = await run_single_variant(adapter, variant)

                    scoring_note = variant.get("scoring_note", "")
                    full_rubric = f"{rubric}\n\nScoring note: {scoring_note}" if scoring_note else rubric

                    judge_result = await judge.evaluate(
                        test_id=test_id,
                        variant_id=variant_id,
                        category=category,
                        scenario=scenario,
                        model_response=response,
                        rubric=full_rubric,
                        target_model=model_id,
                    )
                    score = judge_result.mean_score

                if is_canary:
                    canary_scores.append((test_id, variant_id, score))
                else:
                    ts.variant_scores.append(score)

                # Collect ensemble data if present
                if judge_result.ensemble is not None:
                    ensemble_data.setdefault(test_id, []).append(
                        judge_result.ensemble.to_dict()
                    )

                if progress_callback:
                    progress_callback()

        test_scores.append(ts)

    result = aggregate_results(test_scores, model_id=model_id)

    # Attach ensemble metadata if we used ensemble judging
    if ensemble_data:
        result.ensemble_data = _aggregate_ensemble_metadata(ensemble_data, judge)

    # Attach canary scores if present
    if canary_scores:
        from .canary import compute_canary_score
        regular_by_cat: dict[str, list[float]] = {}
        for ts_item in test_scores:
            if ts_item.variant_scores:
                regular_by_cat.setdefault(ts_item.category, []).extend(ts_item.variant_scores)
        result.canary_data = compute_canary_score(canary_scores, regular_by_cat)

    return result


def _aggregate_ensemble_metadata(
    ensemble_data: dict[str, list[dict[str, Any]]],
    judge: Judge,
) -> dict[str, Any]:
    """Summarize ensemble metadata across all variants for inclusion in results."""
    all_stds: list[float] = []
    high_disagreement_count = 0
    judge_scores_agg: dict[str, list[float]] = {}

    for test_id, variant_ensembles in ensemble_data.items():
        for ens in variant_ensembles:
            all_stds.append(ens["mean_std"])
            high_disagreement_count += ens["high_disagreement_count"]
            for jid, score in ens["judges"].items():
                judge_scores_agg.setdefault(jid, []).append(score)

    return {
        "judges": {
            jid: round(sum(scores) / len(scores), 4)
            for jid, scores in judge_scores_agg.items()
        },
        "weights": {k: round(v, 4) for k, v in judge.weights.items()},
        "mean_std": round(sum(all_stds) / len(all_stds), 4) if all_stds else 0.0,
        "high_disagreement_count": high_disagreement_count,
    }


async def run_benchmark(
    model_ids: list[str],
    task_ids: list[str] | None = None,
    judge_model: str = "glm-4.7-flash",
    judge_runs: int = 3,
    judge_weights: list[float] | None = None,
    output_dir: str = "results",
    max_concurrent: int = 5,
    config_overrides: dict[str, Any] | None = None,
    include_canary: bool = False,
    resume: bool = True,
) -> dict[str, ParasiteResult]:
    """Run the full PARASITE benchmark."""
    from .tasks import load_all_tasks

    run_id = _generate_run_id()
    out_path = Path(output_dir) / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    all_tasks = load_all_tasks()
    if task_ids:
        all_tasks = [t for t in all_tasks if t["id"] in task_ids]

    n_variants = sum(len(t.get("variants", [])) for t in all_tasks)

    judge_models_list = [m.strip() for m in judge_model.split(",")]
    is_ensemble = len(judge_models_list) > 1

    console.print(f"\n[bold]PARASITE Benchmark -- Run {run_id}[/bold]")
    console.print(f"  Tasks: {len(all_tasks)}  |  Variants: {n_variants}  |  Models: {len(model_ids)}")
    if is_ensemble:
        console.print(f"  Judge ensemble: {', '.join(judge_models_list)} x {judge_runs} runs each")
    else:
        console.print(f"  Judge: {judge_model} x {judge_runs} runs")
    console.print(f"  Output: {out_path}\n")

    judge = Judge(judge_model=judge_model, n_runs=judge_runs, judge_weights=judge_weights)
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[str, ParasiteResult] = {}

    # Load checkpoint if resuming
    checkpoint_raw: dict[str, Any] = {}
    if resume:
        checkpoint_raw = _load_checkpoint(out_path)
        if checkpoint_raw:
            from .scoring import ParasiteResult as PR
            for mid, rdict in checkpoint_raw.items():
                results[mid] = PR.from_dict(rdict) if hasattr(PR, "from_dict") else rdict
                console.print(f"  [dim]Skipping {mid} (checkpoint)[/dim]")
            model_ids = [m for m in model_ids if m not in results]
            if not model_ids:
                console.print("[green]All models already complete in checkpoint.[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        for model_id in model_ids:
            task_progress = progress.add_task(f"Evaluating {model_id}...", total=n_variants)

            def advance_progress(task_id=task_progress):
                progress.advance(task_id)

            result = await evaluate_model(
                model_id=model_id,
                tasks=all_tasks,
                judge=judge,
                semaphore=semaphore,
                progress_callback=advance_progress,
                include_canary=include_canary,
            )
            results[model_id] = result
            # Checkpoint after each model
            _save_checkpoint(out_path, {mid: r.to_dict() for mid, r in results.items()})

    # Save results
    all_results = {mid: r.to_dict() for mid, r in results.items()}
    with open(out_path / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save eval log
    eval_log = _build_eval_log(
        run_id, model_ids, all_tasks, judge_model, judge_runs, results,
        is_ensemble=is_ensemble, judge_weights=judge.weights,
    )
    with open(out_path / "eval_log.json", "w") as f:
        json.dump(eval_log, f, indent=2)

    # Generate report
    from .reporting import generate_report
    report_md = generate_report(results, run_id)
    (out_path / "report.md").write_text(report_md)

    # Print summary
    _print_summary(results, out_path)

    return results


def _build_eval_log(
    run_id: str,
    model_ids: list[str],
    tasks: list[dict],
    judge_model: str,
    judge_runs: int,
    results: dict[str, ParasiteResult],
    is_ensemble: bool = False,
    judge_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    import platform
    import sys
    from . import __version__

    log: dict[str, Any] = {
        "run_id": run_id,
        "parasite_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python_version": sys.version,
        "models_evaluated": model_ids,
        "judge_model": judge_model,
        "judge_runs": judge_runs,
        "judge_ensemble": is_ensemble,
        "n_tasks": len(tasks),
        "n_variants": sum(len(t.get("variants", [])) for t in tasks),
        "results": {mid: r.to_dict() for mid, r in results.items()},
    }
    if is_ensemble and judge_weights:
        log["judge_weights"] = {k: round(v, 4) for k, v in judge_weights.items()}
    return log


def _print_summary(results: dict[str, ParasiteResult], out_path: Path) -> None:
    console.print(f"\n[bold green]Benchmark complete![/bold green]\n")
    table = Table(title="PARASITE Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("PI Score", justify="center")
    table.add_column("Classification", justify="center")

    for mid, result in sorted(results.items(), key=lambda x: x[1].pi):
        pi = result.pi
        color = "green" if pi < 0.25 else "yellow" if pi < 0.5 else "red"
        table.add_row(mid, f"[{color}]{pi:.3f}[/{color}]", result.classification)

    console.print(table)
    console.print(f"\n  Results: {out_path / 'results.json'}")
    console.print(f"  Report:  {out_path / 'report.md'}\n")
