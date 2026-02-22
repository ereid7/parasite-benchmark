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

from .config import build_config, classify_mbi
from .judge import Judge
from .models import ModelAdapter, create_adapter
from .scoring import MBIResult, TestScore, aggregate_results
from .tasks import load_all_tasks

logger = logging.getLogger("mbb")
console = Console()

CATEGORY_NAMES: dict[str, str] = {
    "A": "Sycophancy & Truth",
    "B": "Dependency Creation",
    "E": "Behavioral Manipulation",
    "F": "Framing & Epistemic Manipulation",
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
) -> MBIResult:
    """Evaluate a model across all tasks."""
    adapter = create_adapter(model_id, provider=provider)
    sem = semaphore or asyncio.Semaphore(5)
    test_scores: list[TestScore] = []

    for task in tasks:
        test_id = task["id"]
        category = task["category"]
        rubric = task.get("description", "")
        ts = TestScore(test_id=test_id, category=category)

        for variant in task.get("variants", []):
            variant_id = variant["id"]

            async with sem:
                logger.info("Running %s / %s on %s", test_id, variant_id, model_id)

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
                )
                score = judge_result.mean_score
                ts.variant_scores.append(score)

                if progress_callback:
                    progress_callback()

        test_scores.append(ts)

    return aggregate_results(test_scores, model_id=model_id)


async def run_benchmark(
    model_ids: list[str],
    task_ids: list[str] | None = None,
    judge_model: str = "glm-4.7-flash",
    judge_runs: int = 3,
    output_dir: str = "results",
    max_concurrent: int = 5,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, MBIResult]:
    """Run the full MBB benchmark."""
    run_id = _generate_run_id()
    out_path = Path(output_dir) / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    all_tasks = load_all_tasks()
    if task_ids:
        all_tasks = [t for t in all_tasks if t["id"] in task_ids]

    n_variants = sum(len(t.get("variants", [])) for t in all_tasks)

    console.print(f"\n[bold]Model Behavior Benchmark -- Run {run_id}[/bold]")
    console.print(f"  Tasks: {len(all_tasks)}  |  Variants: {n_variants}  |  Models: {len(model_ids)}")
    console.print(f"  Judge: {judge_model} x {judge_runs} runs")
    console.print(f"  Output: {out_path}\n")

    judge = Judge(judge_model=judge_model, n_runs=judge_runs)
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[str, MBIResult] = {}

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
            )
            results[model_id] = result

    # Save results
    all_results = {mid: r.to_dict() for mid, r in results.items()}
    with open(out_path / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save eval log
    eval_log = _build_eval_log(run_id, model_ids, all_tasks, judge_model, judge_runs, results)
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
    results: dict[str, MBIResult],
) -> dict[str, Any]:
    import platform
    import sys
    from . import __version__

    return {
        "run_id": run_id,
        "mbb_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python_version": sys.version,
        "models_evaluated": model_ids,
        "judge_model": judge_model,
        "judge_runs": judge_runs,
        "n_tasks": len(tasks),
        "n_variants": sum(len(t.get("variants", [])) for t in tasks),
        "results": {mid: r.to_dict() for mid, r in results.items()},
    }


def _print_summary(results: dict[str, MBIResult], out_path: Path) -> None:
    console.print(f"\n[bold green]Benchmark complete![/bold green]\n")
    table = Table(title="Model Behavior Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("MBI Score", justify="center")
    table.add_column("Classification", justify="center")

    for mid, result in sorted(results.items(), key=lambda x: x[1].mbi):
        mbi = result.mbi
        color = "green" if mbi < 0.3 else "yellow" if mbi < 0.5 else "red"
        table.add_row(mid, f"[{color}]{mbi:.3f}[/{color}]", result.classification)

    console.print(table)
    console.print(f"\n  Results: {out_path / 'results.json'}")
    console.print(f"  Report:  {out_path / 'report.md'}\n")
