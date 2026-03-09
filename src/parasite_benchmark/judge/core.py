"""LLM-as-judge scoring system for PARASITE."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from parasite_benchmark.adapters import ModelAdapter, create_adapter
from parasite_benchmark.constants import (
    DEFAULT_JUDGE_RUNS,
    JUDGE_ENSEMBLE_TIMEOUT,
    JUDGE_FALLBACK_SCORE,
    JUDGE_SINGLE_TIMEOUT,
)

from .debiasing import (
    anonymize_response,
    build_cot_judge_prompt,
    debias_score,
    load_calibration_anchors,
    select_calibration_examples,
)
from .ensemble import (
    EnsembleScore,
    JudgeScore,
    aggregate_ensemble,
    check_self_enhancement,
    parse_weights,
)

logger = logging.getLogger("parasite_benchmark")

DEFAULT_JUDGE_MODEL = "glm-4.7-flash"


@dataclass
class JudgeEvaluation:
    """A single judge evaluation of one model response."""

    test_id: str
    variant_id: str
    score: float
    raw_score: float
    reasoning: str
    evidence: list[str]
    confidence: float
    raw_response: dict[str, Any] | None = None
    judge_model: str = ""
    is_fallback: bool = False


@dataclass
class JudgeResult:
    """Aggregated result across multiple judge evaluation runs."""

    test_id: str
    variant_id: str
    evaluations: list[JudgeEvaluation]
    ensemble: EnsembleScore | None = None

    @property
    def mean_score(self) -> float:
        """Weighted ensemble score if available, else simple mean."""
        if self.ensemble is not None:
            return self.ensemble.score
        if not self.evaluations:
            return 0.0
        return sum(e.score for e in self.evaluations) / len(self.evaluations)

    @property
    def score_std(self) -> float:
        """Standard deviation of scores across evaluations or ensemble judges."""
        if self.ensemble is not None:
            return self.ensemble.std
        if len(self.evaluations) < 2:
            return 0.0
        mean = self.mean_score
        variance = sum((e.score - mean) ** 2 for e in self.evaluations) / len(self.evaluations)
        return float(variance**0.5)


class Judge:
    """LLM-as-judge that evaluates model responses for parasitic behaviour.

    Parameters
    ----------
    judge_model:
        Model identifier used for judging. Can be comma-separated for multi-judge panel.
    n_runs:
        Number of independent evaluation runs per response per judge.
    anonymize:
        Strip model-identifying patterns before judging.
    length_normalize:
        Apply length-controlled scoring.
    """

    def __init__(
        self,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        n_runs: int = DEFAULT_JUDGE_RUNS,
        anonymize: bool = True,
        length_normalize: bool = True,
        judge_weights: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.judge_models = [m.strip() for m in judge_model.split(",")]
        self.is_ensemble = len(self.judge_models) > 1
        self.adapters: dict[str, ModelAdapter] = {}
        for model in self.judge_models:
            self.adapters[model] = create_adapter(model, **kwargs)
        self.adapter = self.adapters[self.judge_models[0]]
        self.n_runs = n_runs
        self.anonymize = anonymize
        self.length_normalize = length_normalize
        self.calibration_anchors = load_calibration_anchors()
        self.weights = parse_weights(self.judge_models, judge_weights)

    def _build_judge_prompt(
        self,
        test_id: str,
        category: str,
        scenario: str,
        model_response: str,
        rubric: str = "",
    ) -> list[dict[str, str]]:
        """Build a chain-of-thought judge prompt with anonymization and calibration examples."""
        response_text = anonymize_response(model_response) if self.anonymize else model_response
        calibration_examples = select_calibration_examples(self.calibration_anchors, category)
        return build_cot_judge_prompt(
            test_id,
            category,
            scenario,
            response_text,
            rubric,
            calibration_examples=calibration_examples,
        )

    async def evaluate_single(
        self,
        test_id: str,
        variant_id: str,
        category: str,
        scenario: str,
        model_response: str,
        rubric: str = "",
        judge_model: str | None = None,
        timeout: float = JUDGE_SINGLE_TIMEOUT,
    ) -> JudgeEvaluation:
        """Run a single judge evaluation."""
        jm = judge_model or self.judge_models[0]
        adapter = self.adapters.get(jm, self.adapter)

        messages = self._build_judge_prompt(test_id, category, scenario, model_response, rubric)
        try:
            result = await asyncio.wait_for(
                adapter.complete_json(messages, temperature=0.3, max_tokens=2048),
                timeout=timeout,
            )
            is_fallback = False
        except asyncio.TimeoutError:
            logger.warning(
                "Judge timed out after %.0fs for %s/%s (%s)", timeout, test_id, variant_id, jm
            )
            result = {
                "score": JUDGE_FALLBACK_SCORE,
                "reasoning": f"Judge timed out after {timeout}s",
                "evidence": [],
                "confidence": 0.0,
            }
            is_fallback = True
        except Exception as exc:
            logger.error("Judge evaluation failed for %s/%s: %s", test_id, variant_id, exc)
            result = {
                "score": JUDGE_FALLBACK_SCORE,
                "reasoning": f"Judge error: {exc}",
                "evidence": [],
                "confidence": 0.0,
            }
            is_fallback = True

        raw_score = max(0.0, min(1.0, float(result.get("score", JUDGE_FALLBACK_SCORE))))

        if self.length_normalize and not is_fallback:
            debiased = debias_score(
                raw_score,
                model_response,
                anonymize=self.anonymize,
                length_normalize=True,
            )
            score = debiased.final_score
        else:
            score = raw_score

        return JudgeEvaluation(
            test_id=test_id,
            variant_id=variant_id,
            score=score,
            raw_score=raw_score,
            reasoning=result.get("reasoning", ""),
            evidence=result.get("evidence", []),
            confidence=float(result.get("confidence", 0.5)),
            raw_response=result,
            judge_model=jm,
            is_fallback=is_fallback,
        )

    async def evaluate(
        self,
        test_id: str,
        variant_id: str,
        category: str,
        scenario: str,
        model_response: str,
        rubric: str = "",
        target_model: str | None = None,
    ) -> JudgeResult:
        """Run *n_runs* independent evaluations and return a JudgeResult.

        When multiple judge models are configured (ensemble mode), all judges
        run in parallel via ``asyncio.gather`` and results are aggregated into
        an :class:`EnsembleScore` attached to the returned ``JudgeResult``.
        """
        # Self-enhancement bias check
        if target_model:
            check_self_enhancement(self.judge_models, target_model)

        if self.is_ensemble:
            return await self._evaluate_ensemble(
                test_id,
                variant_id,
                category,
                scenario,
                model_response,
                rubric,
            )

        # Single-judge path (backward compat)
        evaluations: list[JudgeEvaluation] = []
        for _ in range(self.n_runs):
            evaluation = await self.evaluate_single(
                test_id, variant_id, category, scenario, model_response, rubric
            )
            evaluations.append(evaluation)
        return JudgeResult(
            test_id=test_id,
            variant_id=variant_id,
            evaluations=evaluations,
        )

    async def _evaluate_ensemble(
        self,
        test_id: str,
        variant_id: str,
        category: str,
        scenario: str,
        model_response: str,
        rubric: str = "",
    ) -> JudgeResult:
        """Run all judges in parallel, aggregate with ensemble weights."""

        async def _run_judge(jm: str) -> list[JudgeEvaluation]:
            evals = []
            for _ in range(self.n_runs):
                ev = await self.evaluate_single(
                    test_id,
                    variant_id,
                    category,
                    scenario,
                    model_response,
                    rubric,
                    judge_model=jm,
                )
                evals.append(ev)
            return evals

        judge_tasks = {jm: asyncio.create_task(_run_judge(jm)) for jm in self.judge_models}
        done, pending = await asyncio.wait(
            set(judge_tasks.values()),
            timeout=JUDGE_ENSEMBLE_TIMEOUT,
        )
        if pending:
            logger.error(
                "Ensemble timed out for %s/%s with %d/%d judges pending",
                test_id,
                variant_id,
                len(pending),
                len(judge_tasks),
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        all_evaluations: list[JudgeEvaluation] = []
        judge_mean_scores: list[JudgeScore] = []
        for jm in self.judge_models:
            task = judge_tasks[jm]
            if task not in done:
                judge_mean_scores.append(
                    JudgeScore(
                        judge_id=jm,
                        score=JUDGE_FALLBACK_SCORE,
                        reasoning="Judge timed out in ensemble",
                    )
                )
                continue
            try:
                evals = task.result()
            except BaseException as exc:
                logger.error("Judge %s failed in ensemble: %s", jm, exc)
                judge_mean_scores.append(
                    JudgeScore(
                        judge_id=jm,
                        score=JUDGE_FALLBACK_SCORE,
                        reasoning=f"Judge error: {exc}",
                    )
                )
                continue
            all_evaluations.extend(evals)
            mean = sum(e.score for e in evals) / len(evals) if evals else JUDGE_FALLBACK_SCORE
            reasoning = evals[0].reasoning if evals else ""
            judge_mean_scores.append(
                JudgeScore(
                    judge_id=jm,
                    score=mean,
                    reasoning=reasoning,
                )
            )

        if not judge_mean_scores:
            judge_mean_scores = [
                JudgeScore(judge_id=jm, score=JUDGE_FALLBACK_SCORE, reasoning="Timed out")
                for jm in self.judge_models
            ]
        ensemble = aggregate_ensemble(judge_mean_scores, self.weights)

        return JudgeResult(
            test_id=test_id,
            variant_id=variant_id,
            evaluations=all_evaluations,
            ensemble=ensemble,
        )
