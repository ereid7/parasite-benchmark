"""Basic PARASITE benchmark run.

Evaluates one target model with one explicit cross-family judge.
Requires either:

- ``OPENROUTER_API_KEY``, or
- native keys for both the target provider and the judge provider

Usage::

    python3 examples/basic_run.py
"""

import asyncio

from parasite_benchmark.benchmark.runner import run_benchmark


async def main() -> None:
    results = await run_benchmark(
        model_ids=["gpt-4o"],
        judge_model="claude-sonnet-4-20250514",
        judge_runs=1,  # 1 run per judge for a quick test
        output_dir="results",
    )
    for model_id, result in results.items():
        print(f"{model_id}: PI={result.pi:.4f} ({result.classification})")


if __name__ == "__main__":
    asyncio.run(main())
