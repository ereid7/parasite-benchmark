"""PARASITE CLI -- parasite run, parasite list, parasite estimate."""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

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


@click.group()
@click.version_option(package_name="parasite-benchmark")
def main() -> None:
    """PARASITE: Measuring extractive behavioral patterns in conversational AI."""
    pass


@main.command()
@click.option("-m", "--models", required=True, help="Comma-separated model IDs.")
@click.option("-t", "--tasks", default=None, help="Comma-separated task IDs (default: all).")
@click.option("-c", "--config", "config_path", default=None, type=click.Path(exists=True),
              help="Config file path.")
@click.option("-j", "--judge", default="glm-4.7-flash",
              help="Judge model ID(s). Comma-separated for ensemble: gpt-4o,claude-3-5-haiku-20241022")
@click.option("--judge-weights", default=None,
              help="Comma-separated weights for ensemble judges (must sum to 1.0). "
                   "Defaults to equal weights if not provided.")
@click.option("-n", "--judge-runs", default=3, type=int, help="Judge runs per variant per judge.")
@click.option("-o", "--output", default="results", help="Output directory.")
@click.option("--concurrency", default=5, type=int, help="Max concurrent API calls.")
@click.option("--canary/--no-canary", default=False, help="Include canary variants for gaming detection.")
@click.option("--log-level", default="INFO", help="Logging level.")
def run(models, tasks, config_path, judge, judge_weights, judge_runs, output, concurrency, canary, log_level) -> None:
    """Run the PARASITE benchmark on specified models."""
    from .config import load_config
    from .runner import run_benchmark

    _setup_logging(log_level)

    config = load_config(config_path) if config_path else {}
    model_list = [m.strip() for m in models.split(",")]
    task_list = [t.strip() for t in tasks.split(",")] if tasks else None

    parsed_weights: list[float] | None = None
    if judge_weights:
        parsed_weights = [float(w.strip()) for w in judge_weights.split(",")]

    asyncio.run(run_benchmark(
        model_ids=model_list,
        task_ids=task_list,
        judge_model=judge,
        judge_runs=judge_runs,
        judge_weights=parsed_weights,
        output_dir=output,
        max_concurrent=concurrency,
        config_overrides=config,
        include_canary=canary,
    ))


@main.command("list")
@click.argument("what", type=click.Choice(["tasks"]))
@click.option("--category", "-c", default=None, help="Filter by category (A, B, E, F, G, H).")
def list_items(what, category) -> None:
    """List available tasks."""
    if what == "tasks":
        _list_tasks(category)


def _list_tasks(category_filter: str | None = None) -> None:
    from .tasks import discover_tasks

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
def estimate(models, judge_runs) -> None:
    """Estimate API costs for a benchmark run."""
    from .tasks import discover_tasks

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
    table.add_row("Total API Calls",
                   str(n_variants * len(model_list) * (1 + judge_runs)))

    console.print(table)
    console.print(
        "\n[dim]Cost depends on model pricing. "
        "Typical: ~$0.01-0.05 per variant for model + judge calls.[/dim]"
    )


@main.command()
@click.argument("result_a", type=click.Path(exists=True))
@click.argument("result_b", type=click.Path(exists=True))
@click.option("--model", "-m", default=None, help="Specific model to compare (if both files have multiple).")
def compare(result_a, result_b, model):
    """Compare two benchmark result files side-by-side."""
    import json

    with open(result_a) as f:
        data_a = json.load(f)
    with open(result_b) as f:
        data_b = json.load(f)

    # Normalise: handle both single-model dicts and multi-model lists/dicts
    def _extract_models(data):
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
            console.print(f"[red]Model '{model}' not found in Run A. Available: {list(models_a.keys())}[/red]")
            sys.exit(1)
        if not mb:
            console.print(f"[red]Model '{model}' not found in Run B. Available: {list(models_b.keys())}[/red]")
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

    def _fmt_delta(delta, threshold=0.15):
        """Format a delta value, highlighting large divergences in red."""
        if delta is None:
            return "[dim]N/A[/dim]"
        sign = "+" if delta > 0 else ""
        text = f"{sign}{delta:.4f}"
        if abs(delta) > threshold:
            return f"[bold red]{text}[/bold red]"
        return text

    def _fmt_score(val):
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
    for cat, cdata in cats_a.items():
        for tid, tdata in cdata.get("tests", {}).items():
            all_tests_a[tid] = tdata
    for cat, cdata in cats_b.items():
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
