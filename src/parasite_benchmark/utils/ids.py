"""Run ID generation."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def generate_run_id() -> str:
    """Generate a unique run identifier: ``YYYYMMDD_HHMMSS_<6-hex>``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid4().hex[:6]}"
