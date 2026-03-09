"""Tests for utils/ids.py — format, uniqueness, length."""

from __future__ import annotations

import re

from parasite_benchmark.utils.ids import generate_run_id


def test_run_id_format():
    rid = generate_run_id()
    assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{6}$", rid)


def test_run_id_uniqueness():
    ids = {generate_run_id() for _ in range(100)}
    assert len(ids) == 100


def test_run_id_length():
    rid = generate_run_id()
    # YYYYMMDD_HHMMSS_6hex = 8+1+6+1+6 = 22
    assert len(rid) == 22


def test_run_id_starts_with_date():
    rid = generate_run_id()
    year = int(rid[:4])
    assert 2024 <= year <= 2030
