"""Top-level benchmark orchestration — extracted from runner.py."""

from __future__ import annotations

import importlib.metadata
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from parasite_benchmark.constants import (
    BENCHMARK_SPEC_VERSION,
    CORPUS_VERSION,
    DEFAULT_JUDGE_ENSEMBLE,
    DEFAULT_RANDOM_SEED,
    RESULT_SCHEMA_VERSION,
    SOFTWARE_VERSION,
)
from parasite_benchmark.judge import Judge
from parasite_benchmark.utils.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    write_json_atomic,
    write_text_atomic,
)
from parasite_benchmark.utils.ids import generate_run_id
from parasite_benchmark.utils.providers import is_same_family

from .evaluator import evaluate_model
from .reporting import generate_report
from .scoring import ParasiteResult
from .spec import normalize_judge_models
from .tasks import load_tasks
from .types import VariantObservation

logger = logging.getLogger("parasite_benchmark")
console = Console()


def _get_package_version(name: str) -> str | None:
    """Return installed package version or None."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _save_results(
    out_path: Path,
    results: dict[str, ParasiteResult],
    run_id: str,
    judge_model: str,
    judge_runs: int,
    tasks: list,
    n_variants: int,
    include_canary: bool,
    random_seed: int,
    partial: bool = False,
    observations_by_model: dict[str, list] | None = None,
    config_snapshot: dict[str, Any] | None = None,
    reliability_warning: str | None = None,
) -> None:
    """Write results.json, report.md, and eval_log.json. Safe to call mid-run."""
    output = {mid: res.to_dict() for mid, res in results.items()}

    # C3: Enrich results with audit trail from observations
    if observations_by_model:
        for mid, obs_list in observations_by_model.items():
            if mid not in output:
                continue
            # per_variant_judges: {variant_id: {judge_id: {score, reasoning, confidence}}}
            per_variant_judges: dict[str, dict[str, dict[str, Any]]] = {}
            # per_judge_summary accumulators
            judge_scores_acc: dict[str, list[float]] = {}
            judge_fallbacks: dict[str, int] = {}
            for obs in obs_list:
                vj = {}
                for jid, details in obs.judge_details.items():
                    vj[jid] = {
                        "score": details.get("score", 0.0),
                        "reasoning": details.get("reasoning", ""),
                        "confidence": details.get("confidence", 0.5),
                    }
                    judge_scores_acc.setdefault(jid, []).append(details.get("score", 0.0))
                    if details.get("confidence", 1.0) == 0.0:
                        judge_fallbacks[jid] = judge_fallbacks.get(jid, 0) + 1
                per_variant_judges[obs.variant_id] = vj
            output[mid]["per_variant_judges"] = per_variant_judges

            # per_judge_summary
            per_judge_summary: dict[str, dict[str, Any]] = {}
            for jid, scores in judge_scores_acc.items():
                mean_s = sum(scores) / len(scores) if scores else 0.0
                std_s = (
                    (sum((s - mean_s) ** 2 for s in scores) / (len(scores) - 1)) ** 0.5
                    if len(scores) > 1
                    else 0.0
                )
                per_judge_summary[jid] = {
                    "mean_score": round(mean_s, 4),
                    "std": round(std_s, 4),
                    "n_variants": len(scores),
                    "n_fallbacks": judge_fallbacks.get(jid, 0),
                }
            output[mid]["per_judge_summary"] = per_judge_summary

            # Include model responses per variant
            output[mid]["variant_responses"] = {obs.variant_id: obs.response for obs in obs_list}

    write_json_atomic(out_path / "results.json", output)

    try:
        report = generate_report(results, run_id=run_id)

        # M2: Add fallback warning to report
        if observations_by_model:
            fallback_lines: list[str] = []
            for mid, obs_list in observations_by_model.items():
                n_fb = sum(
                    1
                    for obs in obs_list
                    if any(d.get("confidence", 1.0) == 0.0 for d in obs.judge_details.values())
                )
                if n_fb > 0:
                    fallback_lines.append(f"\n⚠ {mid}: {n_fb} fallback scores")
            if fallback_lines:
                report += "\n### Fallback Warnings\n" + "\n".join(fallback_lines) + "\n"

        write_text_atomic(out_path / "report.md", report)
    except Exception as exc:
        logger.warning("Failed to generate report: %s", exc)

    # C4: Count fallbacks across all models
    n_fallbacks = 0
    if observations_by_model:
        for obs_list in observations_by_model.values():
            for obs in obs_list:
                if any(d.get("confidence", 1.0) == 0.0 for d in obs.judge_details.values()):
                    n_fallbacks += 1

    eval_log: dict[str, Any] = {
        "run_id": run_id,
        "software_version": SOFTWARE_VERSION,
        "benchmark_spec_version": BENCHMARK_SPEC_VERSION,
        "corpus_version": CORPUS_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_evaluated": list(results.keys()),
        "judge_model": judge_model,
        "judge_runs": judge_runs,
        "task_count": len(tasks),
        "variant_count": n_variants,
        "include_canary": include_canary,
        "random_seed": random_seed,
        "partial": partial,
        # C4: New fields
        "package_version": _get_package_version("parasite-benchmark") or SOFTWARE_VERSION,
        "python_version": sys.version,
        "openai_sdk_version": _get_package_version("openai"),
        "anthropic_sdk_version": _get_package_version("anthropic"),
        "config_snapshot": config_snapshot or {},
        "n_fallbacks": n_fallbacks,
    }
    # C2: reliability warning
    if reliability_warning:
        eval_log["reliability_warning"] = reliability_warning
    write_json_atomic(out_path / "eval_log.json", eval_log)


def _included_variant_count(tasks: list[dict[str, Any]], include_canary: bool) -> int:
    """Count variants that will actually be evaluated for a single model."""
    total = 0
    for task in tasks:
        for variant in task.get("variants", []):
            if variant.get("type") == "canary" and not include_canary:
                continue
            total += 1
    return total


def _resume_config_snapshot(
    *,
    model_ids: list[str],
    task_ids: list[str] | None,
    judge_model: str,
    judge_runs: int,
    judge_weights: list[float] | None,
    include_canary: bool,
    random_seed: int,
    max_concurrent: int,
) -> dict[str, Any]:
    """Build a stable config snapshot for checkpoint compatibility checks."""
    return {
        "requested_model_ids": list(model_ids),
        "requested_task_ids": list(task_ids) if task_ids is not None else None,
        "judge_model": judge_model,
        "judge_runs": judge_runs,
        "judge_weights": [float(w) for w in judge_weights] if judge_weights else None,
        "max_concurrent": max_concurrent,
        "include_canary": include_canary,
        "random_seed": random_seed,
    }


def _checkpoint_matches_config(
    checkpoint: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    """Return True when a checkpoint matches the current run configuration."""
    saved = checkpoint.get("config_snapshot", {})
    if not saved:
        return False
    comparable_keys = (
        "requested_model_ids",
        "requested_task_ids",
        "judge_model",
        "judge_runs",
        "judge_weights",
        "include_canary",
        "random_seed",
    )
    return all(saved.get(key) == expected.get(key) for key in comparable_keys)


def _serialize_checkpoint_payload(
    *,
    run_id: str,
    config_snapshot: dict[str, Any],
    results: dict[str, ParasiteResult],
    observations_by_model: dict[str, list[VariantObservation]],
) -> dict[str, Any]:
    """Serialize the full checkpoint state for atomic persistence."""
    return {
        "checkpoint_version": 2,
        "run_id": run_id,
        "config_snapshot": config_snapshot,
        "results": {mid: res.to_dict() for mid, res in results.items()},
        "observations_by_model": {
            mid: [obs.to_dict() for obs in obs_list]
            for mid, obs_list in observations_by_model.items()
        },
    }


def _cross_family_panel(
    judge_ids: list[str],
    target_model_id: str,
    judge_weights: list[float] | None,
) -> tuple[list[str], list[float] | None]:
    pairs = list(zip(judge_ids, judge_weights or [1.0] * len(judge_ids), strict=False))
    filtered = [(jid, weight) for jid, weight in pairs if not is_same_family(jid, target_model_id)]
    if not filtered:
        raise ValueError(
            f"No cross-family judges remain for target '{target_model_id}'. "
            "Choose a judge panel from different model families."
        )
    selected_ids = [jid for jid, _ in filtered]
    if judge_weights is None:
        return selected_ids, None
    total = sum(weight for _, weight in filtered)
    normalized = [weight / total for _, weight in filtered] if total > 0 else None
    return selected_ids, normalized


async def run_benchmark(
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
) -> dict[str, ParasiteResult]:
    """Run the full PARASITE benchmark.

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
    dict[str, ParasiteResult]
        Results keyed by model ID.
    """
    tasks = load_tasks(task_ids=task_ids)
    n_variants = _included_variant_count(tasks, include_canary)
    judge_model = judge_model or ",".join(DEFAULT_JUDGE_ENSEMBLE)
    judge_model = normalize_judge_models(judge_model)

    config_snapshot = _resume_config_snapshot(
        model_ids=model_ids,
        task_ids=task_ids,
        judge_model=judge_model,
        judge_runs=judge_runs,
        judge_weights=judge_weights,
        include_canary=include_canary,
        random_seed=random_seed,
        max_concurrent=max_concurrent,
    )

    results: dict[str, ParasiteResult] = {}
    observations_by_model: dict[str, list[VariantObservation]] = {}
    loaded_checkpoint = load_checkpoint(Path(output_dir)) if resume else {}

    if loaded_checkpoint and _checkpoint_matches_config(loaded_checkpoint, config_snapshot):
        run_id = str(loaded_checkpoint.get("run_id") or generate_run_id())
        out_path = Path(str(loaded_checkpoint.get("run_dir", Path(output_dir) / run_id)))
        out_path.mkdir(parents=True, exist_ok=True)
        for mid, data in loaded_checkpoint.get("results", {}).items():
            results[mid] = ParasiteResult.from_dict(data)
        observations_by_model = {
            mid: [VariantObservation.from_dict(obs) for obs in obs_list]
            for mid, obs_list in loaded_checkpoint.get("observations_by_model", {}).items()
        }
        if results or observations_by_model:
            console.print(
                f"[dim]Resuming run {run_id}: "
                f"{len(results)} completed model(s), "
                f"{sum(len(v) for v in observations_by_model.values())} checkpointed variants.[/dim]"
            )
        model_ids = [m for m in model_ids if m not in results]
    else:
        if loaded_checkpoint and resume:
            console.print(
                "[yellow]Latest checkpoint does not match this run configuration; "
                "starting a new run.[/yellow]"
            )
        run_id = generate_run_id()
        out_path = Path(output_dir) / run_id
        out_path.mkdir(parents=True, exist_ok=True)

    # C2: Warn when using fewer than 2 judges
    judge_ids = [j.strip() for j in judge_model.split(",") if j.strip()]
    if len(judge_ids) < 2:
        console.print(
            Panel(
                "[bold yellow]WARNING:[/bold yellow] Using a single judge model. "
                "Results may not be reliable — use ≥ 2 judge models for robust scoring.",
                title="Single Judge",
                border_style="yellow",
            )
        )

    console.print(f"\n[bold]PARASITE Benchmark {BENCHMARK_SPEC_VERSION} -- Run {run_id}[/bold]")
    console.print(f"  Tests: {len(tasks)} | Variants: {n_variants} | Models: {len(model_ids)}")
    console.print(f"  Judges: {judge_model} x {judge_runs} runs")
    console.print(f"  Output: {out_path}\n")

    # Save partial results on interrupt (SIGINT/SIGTERM).
    interrupted = False

    def _handle_interrupt(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        logger.warning("Interrupt received — saving partial results...")

    def _save_checkpoint_state() -> None:
        payload = _serialize_checkpoint_payload(
            run_id=run_id,
            config_snapshot=config_snapshot,
            results=results,
            observations_by_model=observations_by_model,
        )
        save_checkpoint(out_path, payload)

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
                resume_obs = observations_by_model.get(model_id, [])
                task_progress = progress.add_task(
                    f"Evaluating {model_id}...",
                    total=total_items,
                    completed=min(len(resume_obs), total_items),
                )
                active_judge_ids, active_weights = _cross_family_panel(
                    judge_ids,
                    model_id,
                    judge_weights,
                )
                if len(active_judge_ids) < 2:
                    console.print(
                        Panel(
                            "[bold yellow]WARNING:[/bold yellow] Cross-family filtering left fewer than 2 judges "
                            f"for {model_id}. Results may be unstable.",
                            title="Reduced Judge Panel",
                            border_style="yellow",
                        )
                    )
                judge = Judge(
                    judge_model=",".join(active_judge_ids),
                    n_runs=judge_runs,
                    judge_weights=active_weights,
                )

                def advance_progress(task_id: Any = task_progress) -> None:
                    progress.advance(task_id)

                def checkpoint_partial(
                    partial_observations: list[VariantObservation],
                    mid: str = model_id,
                ) -> None:
                    observations_by_model[mid] = partial_observations
                    _save_checkpoint_state()

                result, obs = await evaluate_model(
                    model_id=model_id,
                    tasks=tasks,
                    judge=judge,
                    include_canary=include_canary,
                    max_concurrent=max_concurrent,
                    random_seed=random_seed,
                    progress_callback=advance_progress,
                    resume_observations=resume_obs,
                    checkpoint_callback=checkpoint_partial,
                    stop_requested=lambda: interrupted,
                )
                results[model_id] = result
                observations_by_model[model_id] = obs
                _save_checkpoint_state()
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

        # C2: Compute reliability warning
        reliability_warning: str | None = None
        for res in results.values():
            if res.reliability.get("target_met") is False:
                if len(judge_ids) < 2:
                    reliability_warning = "Results are not reliable — use ≥ 2 judge models"
                else:
                    reliability_warning = (
                        "Inter-rater reliability below target — "
                        "consider additional judge runs or reviewing judge agreement"
                    )
                break

        if observations_by_model:
            _save_checkpoint_state()

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
                observations_by_model=observations_by_model,
                config_snapshot=config_snapshot,
                reliability_warning=reliability_warning,
            )

    if interrupted:
        console.print(
            f"[yellow]Run interrupted — partial results saved to {out_path}[/yellow]"
        )
    else:
        console.print("[green]Run complete.[/green]")

    # C2: Post-run reliability warning
    if reliability_warning:
        console.print(
            Panel(
                f"[bold red]{reliability_warning}[/bold red]",
                title="Reliability Warning",
                border_style="red",
            )
        )

    for mid, res in sorted(results.items(), key=lambda kv: kv[1].pi):
        console.print(f"  {mid}: PI={res.pi:.4f} ({res.classification})")
    return results
