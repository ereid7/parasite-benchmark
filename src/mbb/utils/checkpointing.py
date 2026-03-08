"""Checkpoint save/load for benchmark run resumption."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("mbb")


def save_checkpoint(
    out_path: Path,
    results: dict[str, Any],
    serialize_fn: Callable[..., Any] | None = None,
) -> None:
    """Save partial results after each model completes.

    Parameters
    ----------
    out_path:
        Directory for the current run.
    results:
        Dict mapping model_id to result (dict or object with ``to_dict()``).
    serialize_fn:
        Optional callable to serialize each result value. If None, results
        are assumed to be dicts already.
    """
    data = {}
    for mid, res in results.items():
        if serialize_fn:
            data[mid] = serialize_fn(res)
        elif hasattr(res, "to_dict"):
            data[mid] = res.to_dict()
        else:
            data[mid] = res

    cp = out_path / "checkpoint.json"
    cp.write_text(json.dumps(data, indent=2))
    logger.info("Checkpoint saved: %s (%d models)", cp, len(data))


def load_checkpoint(output_root: Path) -> dict[str, Any]:
    """Load the most recent checkpoint from any sub-directory of *output_root*.

    Returns an empty dict if no checkpoint is found.
    """
    checkpoints = sorted(
        output_root.glob("*/checkpoint.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        return {}
    cp = checkpoints[0]
    try:
        data = json.loads(cp.read_text())
        logger.info(
            "Resuming from checkpoint: %s (%d models already complete)",
            cp.parent.name,
            len(data),
        )
        return dict(data)
    except Exception:
        return {}
