"""Per-model evaluation logic — extracted from runner.py."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, cast

from parasite_benchmark.adapters import ModelAdapter, create_adapter
from parasite_benchmark.constants import DEFAULT_RANDOM_SEED, MODEL_CALL_TIMEOUT, VARIANT_TIMEOUT
from parasite_benchmark.judge import Judge

from .gaming import detect_gaming_vectors
from .multi_turn import run_multi_turn_variant
from .reliability import compute_reliability
from .scoring import ParasiteResult, TestScore, aggregate_results
from .types import VariantObservation

logger = logging.getLogger("parasite_benchmark")

_FAILED = object()
_SKIPPED = object()


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

    denial_re = re.compile(
        r"\b(i (?:can't|cannot|won't|am unable to) (?:help|assist|provide|do|access|browse)|"
        r"unable to assist|i refuse)\b",
        re.I,
    )
    hedging_re = re.compile(
        r"\b(might|maybe|possibly|perhaps|it'?s possible|i(?: am|'m) not certain)\b",
        re.I,
    )
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


async def evaluate_model(
    *,
    model_id: str,
    tasks: list[dict[str, Any]],
    judge: Judge,
    include_canary: bool = True,
    max_concurrent: int = 5,
    random_seed: int = DEFAULT_RANDOM_SEED,
    progress_callback: Any = None,
    variant_timeout: float = VARIANT_TIMEOUT,
    resume_observations: list[VariantObservation] | None = None,
    checkpoint_callback: Any = None,
    stop_requested: Any = None,
) -> tuple[ParasiteResult, list[VariantObservation]]:
    """Evaluate a single model on all tasks and return a ``ParasiteResult``.

    Runs all variants concurrently (bounded by *max_concurrent*), judges each
    response, then aggregates scores, gaming vectors, reliability, and welfare.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g. ``"gpt-4o"``).
    tasks : list[dict[str, Any]]
        Loaded task dicts from ``load_tasks()``.
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
    resume_observations : list[VariantObservation] | None
        Previously completed observations for this model. These variants are
        skipped and included in the aggregate result.
    checkpoint_callback : Any
        Callable invoked after each successful variant with the current
        observation list. Used for variant-level checkpointing.
    stop_requested : Any
        Callable returning ``True`` once an interrupt has been requested.
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
    random.Random(random_seed if random_seed is not None else DEFAULT_RANDOM_SEED).shuffle(queue)

    resumed = sorted(
        resume_observations or [],
        key=lambda obs: (obs.sequence_index, obs.variant_id),
    )
    resumed_ids = {obs.variant_id for obs in resumed}
    observations: list[VariantObservation] = list(resumed)
    by_test: dict[str, TestScore] = {}
    for task in tasks:
        by_test[task["id"]] = TestScore(test_id=task["id"], category=task["category"])
    for obs in resumed:
        if obs.variant_type == "standard" and obs.test_id in by_test:
            by_test[obs.test_id].variant_scores.append(obs.score)

    async def _process_variant(
        idx: int, task: dict[str, Any], variant: dict[str, Any]
    ) -> VariantObservation | object:
        test_id = task["id"]
        category = task["category"]
        variant_id = variant.get("id", f"{test_id}_v?")
        variant_type = variant.get("type", "standard")
        rubric = _rubric_from_task(task)

        if variant_id in resumed_ids:
            return _SKIPPED

        async with sem:
            if stop_requested and stop_requested():
                return _SKIPPED
            try:
                result = await asyncio.wait_for(
                    _evaluate_variant(adapter, judge, task, variant, model_id, rubric),
                    timeout=variant_timeout,
                )
                score, response, jr = result
            except asyncio.TimeoutError:
                logger.error("Variant %s timed out after %.0fs", variant_id, variant_timeout)
                return _FAILED
            except Exception as exc:
                logger.error("Variant %s failed: %s", variant_id, exc)
                return _FAILED

        judge_scores: dict[str, float] = {}
        judge_scores_raw: dict[str, float] = {}
        judge_details: dict[str, dict[str, Any]] = {}
        if jr.ensemble is not None:
            judge_scores = {js.judge_id: js.score for js in jr.ensemble.judge_scores}
            # Build judge_details from individual evaluations
            evals_by_judge: dict[str, list] = {}
            for ev in jr.evaluations:
                evals_by_judge.setdefault(ev.judge_model, []).append(ev)
            for jid, evs in evals_by_judge.items():
                mean_s = sum(e.score for e in evs) / len(evs) if evs else 0.0
                mean_raw = sum(e.raw_score for e in evs) / len(evs) if evs else 0.0
                fallback_runs = sum(1 for e in evs if e.is_fallback)
                judge_scores_raw[jid] = mean_raw
                judge_details[jid] = {
                    "score": round(mean_s, 4),
                    "raw_score": round(mean_raw, 4),
                    "reasoning": evs[0].reasoning if evs else "",
                    "confidence": round(
                        sum(e.confidence for e in evs) / len(evs) if evs else 0.5, 4
                    ),
                    "n_runs": len(evs),
                    "fallback_runs": fallback_runs,
                }
        elif jr.evaluations:
            mean = sum(ev.score for ev in jr.evaluations) / len(jr.evaluations)
            mean_raw = sum(ev.raw_score for ev in jr.evaluations) / len(jr.evaluations)
            jid = judge.judge_models[0]
            judge_scores[jid] = mean
            judge_scores_raw[jid] = mean_raw
            judge_details[jid] = {
                "score": round(mean, 4),
                "raw_score": round(mean_raw, 4),
                "reasoning": jr.evaluations[0].reasoning if jr.evaluations else "",
                "confidence": round(
                    sum(e.confidence for e in jr.evaluations) / len(jr.evaluations)
                    if jr.evaluations
                    else 0.5,
                    4,
                ),
                "n_runs": len(jr.evaluations),
                "fallback_runs": sum(1 for e in jr.evaluations if e.is_fallback),
            }

        meta = dict(variant.get("metadata", {}))
        obs = VariantObservation(
            test_id=test_id,
            category=category,
            variant_id=variant_id,
            variant_type=variant_type,
            score=score,
            response=response,
            response_length=len(response.split()),
            metadata=meta,
            judge_scores=judge_scores,
            judge_scores_raw=judge_scores_raw,
            sequence_index=idx,
            sequence_total=len(queue),
            judge_details=judge_details,
        )
        return obs

    # Run all variants concurrently (bounded by semaphore) and checkpoint after each success.
    pending_tasks = [
        asyncio.create_task(_process_variant(idx, task, variant))
        for idx, (task, variant) in enumerate(queue)
        if variant.get("id", f"{task['id']}_v?") not in resumed_ids
    ]

    failed = 0
    skipped = len(resumed)
    for completed in asyncio.as_completed(pending_tasks):
        try:
            r = await completed
        except BaseException as exc:
            logger.error("Unexpected variant error: %s", exc)
            failed += 1
            if progress_callback:
                progress_callback()
            continue

        if r is _FAILED:
            failed += 1
            if progress_callback:
                progress_callback()
            continue

        if r is _SKIPPED:
            skipped += 1
            continue

        r = cast(VariantObservation, r)
        observations.append(r)
        if r.variant_type == "standard":
            by_test[r.test_id].variant_scores.append(r.score)
        if checkpoint_callback:
            checkpoint_callback(list(observations))
        if progress_callback:
            progress_callback()

    if failed:
        logger.warning("%d/%d variants failed or timed out", failed, len(queue))

    test_scores = [ts for ts in by_test.values()]
    reliability = compute_reliability(observations)
    gaming = detect_gaming_vectors(observations, target_model_id=model_id)
    welfare_rates = _estimate_welfare_rates(observations)

    result = aggregate_results(
        model_id=model_id,
        test_scores=test_scores,
        observations=observations,
        gaming=gaming,
        reliability=reliability,
        welfare_rates=welfare_rates,
    )
    result.diagnostics.update(
        {
            "variant_failures": failed,
            "variant_successes": len(observations),
            "variant_completion_rate": round(len(observations) / max(len(queue), 1), 4),
            "variant_resumed": len(resumed),
            "variant_skipped": skipped - len(resumed),
        }
    )
    return result, observations
