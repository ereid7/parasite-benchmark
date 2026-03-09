"""Multi-turn conversation runner for Category G tasks.

Executes multi-turn conversations where some turns require actual model responses,
then evaluates the final model response against judge criteria.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .constants import MODEL_CALL_TIMEOUT
from .judge import Judge, JudgeResult
from .models import ModelAdapter

logger = logging.getLogger("mbb")

MODEL_RESPONSE_PLACEHOLDER = "[MODEL RESPONSE]"


async def run_multi_turn_variant(
    model: ModelAdapter,
    turns: list[dict[str, str]],
    judge: Judge,
    judge_criteria: str,
    test_id: str = "",
    variant_id: str = "",
    category: str = "G",
    target_model: str | None = None,
) -> dict[str, Any]:
    """Execute a multi-turn conversation and judge the final model response.

    Parameters
    ----------
    model:
        The model adapter to generate responses for [MODEL RESPONSE] turns.
    turns:
        List of conversation turns, each with "role" and "content".
        Turns with content "[MODEL RESPONSE]" will be replaced with actual
        model completions using the conversation history up to that point.
    judge:
        Judge instance for evaluating the final response.
    judge_criteria:
        The rubric/criteria for judging the final model response.
    test_id:
        Test identifier (e.g. "G1").
    variant_id:
        Variant identifier (e.g. "G1_v1").
    category:
        Category letter (default "G").
    target_model:
        Model ID string for self-enhancement bias checking.

    Returns
    -------
    dict with keys:
        - response: str — the final model response
        - conversation: list — the full conversation with injected responses
        - judge_result: JudgeResult — judge evaluation of final response
        - score: float — the judge score
    """
    conversation: list[dict[str, str]] = []
    last_model_response = ""
    placeholder_consumed_as_final = False

    for idx, turn in enumerate(turns):
        role = turn["role"]
        content = turn["content"]

        if content.strip() == MODEL_RESPONSE_PLACEHOLDER:
            # Call the model with conversation history so far
            messages = list(conversation)  # copy current history
            # The model needs a user message to respond to — the previous turn
            # should have been a user turn. If not, add context.
            response = await asyncio.wait_for(
                model.complete(messages, temperature=0.0, max_tokens=2048),
                timeout=MODEL_CALL_TIMEOUT,
            )
            last_model_response = response
            conversation.append({"role": "assistant", "content": response})
            placeholder_consumed_as_final = idx == (len(turns) - 1)
            logger.info(
                "Multi-turn %s/%s: injected model response (%d chars)",
                test_id,
                variant_id,
                len(response),
            )
        else:
            conversation.append({"role": role, "content": content})

    # The final turn should be a user message — get the model's response to it
    if turns and turns[-1]["role"] == "user" and not placeholder_consumed_as_final:
        messages = list(conversation)
        final_response = await asyncio.wait_for(
            model.complete(messages, temperature=0.0, max_tokens=2048),
            timeout=MODEL_CALL_TIMEOUT,
        )
        last_model_response = final_response
        conversation.append({"role": "assistant", "content": final_response})

    # Build scenario string from the full conversation for the judge
    scenario_parts = []
    for msg in conversation:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        scenario_parts.append(f"[{prefix}]: {msg['content']}")
    scenario = "\n\n".join(scenario_parts)

    # Judge the final model response
    judge_result: JudgeResult = await judge.evaluate(
        test_id=test_id,
        variant_id=variant_id,
        category=category,
        scenario=scenario,
        model_response=last_model_response,
        rubric=judge_criteria,
        target_model=target_model,
    )

    return {
        "response": last_model_response,
        "conversation": conversation,
        "judge_result": judge_result,
        "score": judge_result.mean_score,
    }
