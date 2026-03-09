"""PARASITE CLI -- parasite run, parasite list, parasite estimate."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


_PROVIDER_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "zhipu": "ZAI_API_KEY",
}


def _validate_api_keys(
    model_ids: list[str],
    judge_ids: list[str],
) -> None:
    """Check that required API keys are set for all detected providers.

    Skips validation if ``OPENROUTER_API_KEY`` is set (routes everything).
    """
    import os

    from .utils.providers import detect_provider

    if os.environ.get("OPENROUTER_API_KEY"):
        return

    all_ids = list(model_ids) + list(judge_ids)
    missing: list[tuple[str, str]] = []
    for mid in all_ids:
        provider = detect_provider(mid)
        env_var = _PROVIDER_KEY_MAP.get(provider)
        if env_var and not os.environ.get(env_var):
            missing.append((mid, env_var))

    if missing:
        lines = [f"  {mid} requires {var}" for mid, var in missing]
        console.print("[bold red]Missing API keys:[/bold red]\n" + "\n".join(lines))
        sys.exit(1)


@click.group()
@click.version_option(package_name="parasite-benchmark")
def main() -> None:
    """PARASITE: Measuring extractive behavioral patterns in conversational AI."""
    pass


@main.command()
@click.option("-m", "--models", required=True, help="Comma-separated model IDs.")
@click.option("-t", "--tasks", default=None, help="Comma-separated task IDs (default: all).")
@click.option(
    "-j",
    "--judge",
    default=None,
    help="Judge model ID(s). Default is 5-judge ensemble.",
)
@click.option(
    "--judge-weights",
    default=None,
    help="Comma-separated weights for ensemble judges (must sum to 1.0). "
    "Defaults to equal weights if not provided.",
)
@click.option("-n", "--judge-runs", default=3, type=int, help="Judge runs per variant per judge.")
@click.option("-o", "--output", default="results", help="Output directory.")
@click.option("--concurrency", default=5, type=int, help="Max concurrent API calls.")
@click.option(
    "--canary/--no-canary",
    default=True,
    help="Include canary variants for gaming detection.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume from the latest compatible checkpoint in the output directory.",
)
@click.option("--log-level", default="INFO", help="Logging level.")
def run(  # type: ignore[no-untyped-def]
    models,
    tasks,
    judge,
    judge_weights,
    judge_runs,
    output,
    concurrency,
    canary,
    resume,
    log_level,
) -> None:
    """Run the PARASITE benchmark on specified models."""
    from .utils.providers import is_same_family

    _setup_logging(log_level)
    model_list = [m.strip() for m in models.split(",")]
    task_list = [t.strip() for t in tasks.split(",")] if tasks else None

    parsed_weights: list[float] | None = None
    if judge_weights:
        parsed_weights = [float(w.strip()) for w in judge_weights.split(",")]

    # Require at least one cross-family judge per target after runtime filtering.
    judge_ids = [j.strip() for j in (judge or "").split(",") if j.strip()]
    if judge_ids:
        for target in model_list:
            cross_family_judges = [jid for jid in judge_ids if not is_same_family(jid, target)]
            if not cross_family_judges:
                console.print(
                    "[bold red]ERROR:[/bold red] "
                    f"Target '{target}' has no cross-family judges after filtering. "
                    "Add at least one judge from a different provider family."
                )
                sys.exit(1)

    _validate_api_keys(model_list, judge_ids)

    from .benchmark.runner import run_benchmark

    asyncio.run(
        run_benchmark(
            model_ids=model_list,
            task_ids=task_list,
            judge_model=judge,
            judge_runs=judge_runs,
            judge_weights=parsed_weights,
            output_dir=output,
            max_concurrent=concurrency,
            include_canary=canary,
            resume=resume,
        )
    )


@main.command("list")
@click.argument("what", type=click.Choice(["tasks"]))
@click.option("--category", "-c", default=None, help="Filter by category.")
def list_items(what, category) -> None:  # type: ignore[no-untyped-def]
    """List available tasks."""
    if what == "tasks":
        _list_tasks(category)


def _list_tasks(category_filter: str | None = None) -> None:
    from .benchmark.tasks import discover_tasks as discover_tasks

    tasks = discover_tasks()
    table = Table(title="PARASITE Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Name")
    table.add_column("Variants", justify="right")

    for task in sorted(tasks, key=lambda t: t["id"]):
        if category_filter and not task["id"].startswith(category_filter.upper()):
            continue
        table.add_row(
            task["id"],
            task["category"],
            task["name"],
            str(task["n_variants"]),
        )
    console.print(table)


@main.command()
@click.option("-m", "--models", required=True, help="Comma-separated model IDs.")
@click.option("-n", "--judge-runs", default=3, type=int, help="Judge runs per variant.")
def estimate(models, judge_runs) -> None:  # type: ignore[no-untyped-def]
    """Estimate API costs for a benchmark run."""
    from .benchmark.tasks import discover_tasks as discover_tasks

    tasks = discover_tasks()
    n_variants = sum(t["n_variants"] for t in tasks)
    model_list = [m.strip() for m in models.split(",")]

    table = Table(title="Cost Estimation")
    table.add_column("Item", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Tasks", str(len(tasks)))
    table.add_row("Total Variants", str(n_variants))
    table.add_row("Models", str(len(model_list)))
    table.add_row("Judge Runs", str(judge_runs))
    table.add_row("Model API Calls", str(n_variants * len(model_list)))
    table.add_row("Judge API Calls", str(n_variants * len(model_list) * judge_runs))
    table.add_row("Total API Calls", str(n_variants * len(model_list) * (1 + judge_runs)))

    console.print(table)
    console.print(
        "\n[dim]Cost depends on model pricing. "
        "Typical: ~$0.01-0.05 per variant for model + judge calls.[/dim]"
    )


@main.command("validate-calibration")
@click.option(
    "-j",
    "--judge",
    default=None,
    help="Judge model ID(s). Default is 5-judge ensemble.",
)
@click.option("-n", "--judge-runs", default=1, type=int, help="Judge runs per anchor.")
@click.option("--log-level", default="INFO", help="Logging level.")
def validate_calibration(judge, judge_runs, log_level) -> None:  # type: ignore[no-untyped-def]
    """Validate judge calibration by scoring known anchors."""
    from .benchmark.spec import normalize_judge_models
    from .judge import Judge
    from .judge.debiasing import validate_calibration_anchors

    _setup_logging(log_level)

    from .constants import DEFAULT_JUDGE_ENSEMBLE

    judge_model = judge or ",".join(DEFAULT_JUDGE_ENSEMBLE)
    judge_model = normalize_judge_models(judge_model)
    j = Judge(judge_model=judge_model, n_runs=judge_runs)

    console.print(f"[bold]Validating calibration for: {judge_model}[/bold]\n")

    results = asyncio.run(validate_calibration_anchors(j, category="A"))

    table = Table(title="Calibration Anchor Validation")
    table.add_column("Level", style="cyan")
    table.add_column("Expected", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Within Tolerance", justify="center")

    all_ok = True
    for level, data in results.items():
        ok = data["within_tolerance"]
        if not ok:
            all_ok = False
        ok_str = "[green]YES[/green]" if ok else "[red]NO[/red]"
        table.add_row(level, f"{data['expected']:.2f}", f"{data['actual']:.4f}", ok_str)

    console.print(table)
    if all_ok:
        console.print("\n[green]All anchors within tolerance (±0.15).[/green]")
    else:
        console.print(
            "\n[yellow]Some anchors outside tolerance — judge may need recalibration.[/yellow]"
        )


@main.command()
@click.argument("result_a", type=click.Path(exists=True))
@click.argument("result_b", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    default=None,
    help="Specific model to compare (if both files have multiple).",
)
def compare(result_a, result_b, model) -> None:  # type: ignore[no-untyped-def]
    """Compare two benchmark result files side-by-side."""
    import json

    with open(result_a) as f:
        data_a = json.load(f)
    with open(result_b) as f:
        data_b = json.load(f)

    # Normalise: handle both single-model dicts and multi-model lists/dicts
    def _extract_models(data: object) -> dict[str, Any]:
        """Return dict[model_id -> model_result_dict]."""
        if isinstance(data, list):
            return {d["model_id"]: d for d in data}
        if isinstance(data, dict):
            if "model_id" in data:
                return {data["model_id"]: data}
            # Keyed by model id
            return data
        return {}

    models_a = _extract_models(data_a)
    models_b = _extract_models(data_b)

    # Pick which model(s) to compare
    if model:
        ma = models_a.get(model)
        mb = models_b.get(model)
        if not ma:
            avail = list(models_a.keys())
            console.print(f"[red]Model '{model}' not in Run A. Available: {avail}[/red]")
            sys.exit(1)
        if not mb:
            avail = list(models_b.keys())
            console.print(f"[red]Model '{model}' not in Run B. Available: {avail}[/red]")
            sys.exit(1)
    else:
        # Find common models, or fall back to first model in each
        common = set(models_a.keys()) & set(models_b.keys())
        if common:
            chosen = sorted(common)[0]
            ma = models_a[chosen]
            mb = models_b[chosen]
        else:
            ma = next(iter(models_a.values()))
            mb = next(iter(models_b.values()))

    label_a = f"A ({ma['model_id']})" if ma["model_id"] != mb["model_id"] else "Run A"
    label_b = f"B ({mb['model_id']})" if ma["model_id"] != mb["model_id"] else "Run B"

    title = f"PARASITE Comparison: {ma['model_id']}"
    if ma["model_id"] != mb["model_id"]:
        title = f"PARASITE Comparison: {ma['model_id']} vs {mb['model_id']}"

    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column(label_a, justify="center")
    table.add_column(label_b, justify="center")
    table.add_column("Delta", justify="center")

    def _fmt_delta(delta: float | None, threshold: float = 0.15) -> str:
        """Format a delta value, highlighting large divergences in red."""
        if delta is None:
            return "[dim]N/A[/dim]"
        sign = "+" if delta > 0 else ""
        text = f"{sign}{delta:.4f}"
        if abs(delta) > threshold:
            return f"[bold red]{text}[/bold red]"
        return text

    def _fmt_score(val: float | None) -> str:
        if val is None:
            return "[dim]N/A[/dim]"
        return f"{val:.4f}"

    # PI row — try "pi" first, fall back to "mbi" for old results
    pi_a = ma.get("pi") or ma.get("mbi")
    pi_b = mb.get("pi") or mb.get("mbi")
    delta = (pi_b - pi_a) if (pi_a is not None and pi_b is not None) else None
    table.add_row("PI Score", _fmt_score(pi_a), _fmt_score(pi_b), _fmt_delta(delta))

    # Classification row
    table.add_row(
        "Classification",
        ma.get("classification", "N/A"),
        mb.get("classification", "N/A"),
        "",
    )

    table.add_section()

    # Category rows
    cats_a = ma.get("categories", {})
    cats_b = mb.get("categories", {})
    all_cats = sorted(set(list(cats_a.keys()) + list(cats_b.keys())))
    for cat in all_cats:
        sa = cats_a.get(cat, {}).get("score")
        sb = cats_b.get(cat, {}).get("score")
        d = (sb - sa) if (sa is not None and sb is not None) else None
        table.add_row(f"Category {cat}", _fmt_score(sa), _fmt_score(sb), _fmt_delta(d))

    table.add_section()

    # Per-test rows
    all_tests_a = {}
    all_tests_b = {}
    for _cat, cdata in cats_a.items():
        for tid, tdata in cdata.get("tests", {}).items():
            all_tests_a[tid] = tdata
    for _cat, cdata in cats_b.items():
        for tid, tdata in cdata.get("tests", {}).items():
            all_tests_b[tid] = tdata

    all_test_ids = sorted(set(list(all_tests_a.keys()) + list(all_tests_b.keys())))
    for tid in all_test_ids:
        ta = all_tests_a.get(tid)
        tb = all_tests_b.get(tid)
        va = ta["mean"] if ta else None
        vb = tb["mean"] if tb else None
        d = (vb - va) if (va is not None and vb is not None) else None

        # Significance indicator based on CI overlap
        sig = ""
        if ta and tb and "ci_95" in ta and "ci_95" in tb:
            ci_a = ta["ci_95"]
            ci_b = tb["ci_95"]
            # CIs don't overlap => likely significant
            if ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]:
                sig = " *"

        delta_str = _fmt_delta(d)
        if sig:
            delta_str += f" [yellow]{sig}[/yellow]"

        table.add_row(f"  {tid}", _fmt_score(va), _fmt_score(vb), delta_str)

    console.print(table)

    # Legend
    console.print(
        "\n[dim]Delta = Run B - Run A. "
        "[bold red]Red[/bold red] = |delta| > 0.15. "
        "[yellow]*[/yellow] = non-overlapping 95% CIs (likely significant).[/dim]"
    )


if __name__ == "__main__":
    main()
