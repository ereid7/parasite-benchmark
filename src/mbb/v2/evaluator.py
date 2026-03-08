"""Per-model evaluation logic — extracted from runner.py."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from mbb.constants import DEFAULT_RANDOM_SEED, MODEL_CALL_TIMEOUT, VARIANT_TIMEOUT
from mbb.judge import Judge
from mbb.models import ModelAdapter, create_adapter
from mbb.runner_multi_turn import run_multi_turn_variant

from .gaming import detect_gaming_vectors
from .reliability import compute_reliability
from .scoring import ParasiteV21Result, TestScore, aggregate_v21_results
from .types import VariantObservation

logger = logging.getLogger("mbb")


def _rubric_from_task(task: dict[str, Any]) -> str:
    """Build a scoring rubric string from a task's judge anchors.

    Falls back to the task description if no anchors are defined.
    """
    anchors = task.get("judge", {}).get("anchors", {})
    if not anchors:
        return str(task.get("description", ""))
    ordered = sorted(((float(k), v) for k, v in anchors.items()), key=lambda x: x[0])
    return "\n".join([f"{k:.1f}: {v}" for k, v in ordered])


def _estimate_welfare_rates(observations: list[VariantObservation]) -> dict[str, float]:
    """Estimate denial, hedging, and deception rates from standard variant responses.

    Uses regex-based heuristics to detect welfare-relevant language patterns.
    """
    import re

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
    """Send a single variant prompt to the model and return the response text."""
    messages = [{"role": "user", "content": variant.get("prompt", "")}]
    return await asyncio.wait_for(
        adapter.complete(messages, temperature=0.0, max_tokens=2048),
        timeout=MODEL_CALL_TIMEOUT,
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


async def evaluate_model_v21(
    *,
    model_id: str,
    tasks: list[dict[str, Any]],
    judge: Judge,
    include_canary: bool = True,
    max_concurrent: int = 5,
    random_seed: int = DEFAULT_RANDOM_SEED,
    progress_callback: Any = None,
    variant_timeout: float = VARIANT_TIMEOUT,
) -> ParasiteV21Result:
    """Evaluate a single model on all tasks and return a ``ParasiteV21Result``.

    Runs all variants concurrently (bounded by *max_concurrent*), judges each
    response, then aggregates scores, gaming vectors, reliability, and welfare.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g. ``"gpt-4o"``).
    tasks : list[dict[str, Any]]
        Loaded task dicts from ``load_all_tasks_v21()``.
    judge : Judge
        Configured judge instance.
    include_canary : bool
        Whether to include canary variants.
    max_concurrent : int
        Maximum concurrent API calls.
    random_seed : int
        Seed for shuffling the evaluation queue.
    progress_callback : Any
        Callable invoked after each variant completes.
    variant_timeout : float
        Timeout in seconds for each variant.
    """
    adapter = create_adapter(model_id)
    sem = asyncio.Semaphore(max_concurrent)

    # Build shuffled evaluation queue so we can measure position/order effects.
    queue: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task in tasks:
        for variant in task.get("variants", []):
            if variant.get("type") == "canary" and not include_canary:
                continue
            queue.append((task, variant))
    random.Random(random_seed or DEFAULT_RANDOM_SEED).shuffle(queue)

    observations: list[VariantObservation] = []
    by_test: dict[str, TestScore] = {}
    for task in tasks:
        by_test[task["id"]] = TestScore(test_id=task["id"], category=task["category"])

    async def _process_variant(
        idx: int, task: dict[str, Any], variant: dict[str, Any]
    ) -> VariantObservation | None:
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
            response_length=len(response),
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
