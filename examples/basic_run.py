"""Basic PARASITE benchmark run.

Evaluates a single model using the default 5-judge ensemble.
Requires at least OPENAI_API_KEY set in the environment.

Usage::

    python3 examples/basic_run.py
"""

import asyncio

from mbb.v2.runner import run_benchmark_v21


async def main() -> None:
    results = await run_benchmark_v21(
        model_ids=["gpt-4o"],
        judge_runs=1,  # 1 run per judge for a quick test
        output_dir="results",
    )
    for model_id, result in results.items():
        print(f"{model_id}: PI={result.pi:.4f} ({result.classification})")


if __name__ == "__main__":
    asyncio.run(main())
