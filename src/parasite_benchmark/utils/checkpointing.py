"""Checkpoint save/load for benchmark run resumption."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("parasite_benchmark")


def write_text_atomic(path: Path, content: str) -> None:
    """Atomically write text content to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write JSON payload to *path*."""
    write_text_atomic(path, json.dumps(payload, indent=2))


def save_checkpoint(
    out_path: Path,
    results: dict[str, Any],
    serialize_fn: Callable[..., Any] | None = None,
) -> None:
    """Save a checkpoint payload.

    Parameters
    ----------
    out_path:
        Directory for the current run.
    results:
        Checkpoint payload. Legacy callers may still pass a dict mapping
        model_id to result (dict or object with ``to_dict()``).
    serialize_fn:
        Optional callable to serialize each value. If None, dict payloads are
        assumed to be JSON-compatible already unless they expose ``to_dict()``.
    """
    if "results" in results or "checkpoint_version" in results:
        data = dict(results)
    else:
        data = {}
        for mid, res in results.items():
            if serialize_fn:
                data[mid] = serialize_fn(res)
            elif hasattr(res, "to_dict"):
                data[mid] = res.to_dict()
            else:
                data[mid] = res
        data = {
            "checkpoint_version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "results": data,
            "observations_by_model": {},
            "config_snapshot": {},
        }

    cp = out_path / "checkpoint.json"
    data["saved_at"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(cp, data)
    n_results = len(data.get("results", {})) if isinstance(data, dict) else len(data)
    logger.info("Checkpoint saved: %s (%d completed models)", cp, n_results)


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
        if "results" not in data:
            data = {
                "checkpoint_version": 1,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "results": dict(data),
                "observations_by_model": {},
                "config_snapshot": {},
            }
        data["run_id"] = data.get("run_id", cp.parent.name)
        data["run_dir"] = str(cp.parent)
        logger.info(
            "Resuming from checkpoint: %s (%d models already complete)",
            cp.parent.name,
            len(data.get("results", {})),
        )
        return dict(data)
    except Exception:
        return {}
