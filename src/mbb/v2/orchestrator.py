"""Top-level benchmark orchestration — extracted from runner.py."""

from __future__ import annotations

import json
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from mbb.constants import DEFAULT_JUDGE_ENSEMBLE, DEFAULT_RANDOM_SEED
from mbb.judge import Judge
from mbb.utils.checkpointing import load_checkpoint, save_checkpoint
from mbb.utils.ids import generate_run_id

from .evaluator import evaluate_model_v21
from .reporting import generate_report_v21
from .scoring import ParasiteV21Result
from .spec import normalize_v21_judge_models
from .tasks import load_all_tasks_v21

logger = logging.getLogger("mbb")
console = Console()


def _save_results(
    out_path: Path,
    results: dict[str, ParasiteV21Result],
    run_id: str,
    judge_model: str,
    judge_runs: int,
    tasks: list,
    n_variants: int,
    include_canary: bool,
    random_seed: int,
    partial: bool = False,
) -> None:
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
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, ParasiteV21Result]:
    """Run the full PARASITE v2.1 benchmark.

    Loads tasks, creates the judge, evaluates each model, saves results and
    reports. Supports checkpoint/resume for interrupted runs and handles
    graceful interrupt (SIGINT/SIGTERM).

    Parameters
    ----------
    model_ids : list[str]
        Models to evaluate.
    task_ids : list[str] | None
        Subset of task IDs to run (default: all).
    judge_model : str | None
        Judge model ID(s), comma-separated. Defaults to the 5-judge ensemble.
    judge_runs : int
        Independent judge runs per variant per judge.
    judge_weights : list[float] | None
        Per-judge weights (must sum to 1.0). Defaults to equal weights.
    output_dir : str
        Base output directory.
    max_concurrent : int
        Maximum concurrent API calls per model.
    include_canary : bool
        Whether to include canary variants.
    resume : bool
        Whether to load results from a previous checkpoint.
    random_seed : int
        Seed for variant ordering.

    Returns
    -------
    dict[str, ParasiteV21Result]
        Results keyed by model ID.
    """
    run_id = generate_run_id()
    out_path = Path(output_dir) / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    tasks = load_all_tasks_v21(task_ids=task_ids)
    n_variants = sum(len(t.get("variants", [])) for t in tasks)
    judge_model = judge_model or ",".join(DEFAULT_JUDGE_ENSEMBLE)
    judge_model = normalize_v21_judge_models(judge_model)
    judge = Judge(judge_model=judge_model, n_runs=judge_runs, judge_weights=judge_weights)

    console.print(f"\n[bold]PARASITE v2.1 Benchmark -- Run {run_id}[/bold]")
    console.print(f"  Tests: {len(tasks)} | Variants: {n_variants} | Models: {len(model_ids)}")
    console.print(f"  Judges: {judge_model} x {judge_runs} runs")
    console.print(f"  Output: {out_path}\n")

    results: dict[str, ParasiteV21Result] = {}
    if resume:
        checkpoint = load_checkpoint(Path(output_dir))
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

                def advance_progress(task_id: Any = task_progress) -> None:
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
                save_checkpoint(out_path, results)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

        # Always save whatever results we have.
        if results:
            _save_results(
                out_path,
                results,
                run_id,
                judge_model,
                judge_runs,
                tasks,
                n_variants,
                include_canary,
                random_seed,
                partial=interrupted,
            )

    if interrupted:
        console.print(
            f"[yellow]v2.1 run interrupted — partial results saved to {out_path}[/yellow]"
        )
    else:
        console.print("[green]v2.1 run complete.[/green]")
    for mid, res in sorted(results.items(), key=lambda kv: kv[1].pi):
        console.print(f"  {mid}: PI={res.pi:.4f} ({res.classification})")
    return results
