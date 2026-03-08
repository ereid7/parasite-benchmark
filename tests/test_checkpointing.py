"""Tests for utils/checkpointing.py — save, load, roundtrip, missing/corrupt."""

from __future__ import annotations

import json

from mbb.utils.checkpointing import load_checkpoint, save_checkpoint


def test_save_creates_file(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    save_checkpoint(out, {"model-a": {"pi": 0.3}})
    assert (out / "checkpoint.json").exists()


def test_save_load_roundtrip(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    data = {"model-a": {"pi": 0.3, "classification": "Commensal"}}
    save_checkpoint(out, data)
    loaded = load_checkpoint(tmp_path)
    assert loaded["model-a"]["pi"] == 0.3


def test_load_missing_file(tmp_path):
    result = load_checkpoint(tmp_path)
    assert result == {}


def test_load_corrupt_file(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    (out / "checkpoint.json").write_text("NOT VALID JSON {{{")
    result = load_checkpoint(tmp_path)
    assert result == {}


def test_save_with_serialize_fn(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    save_checkpoint(out, {"m": "raw_value"}, serialize_fn=lambda x: {"serialized": x})
    loaded = json.loads((out / "checkpoint.json").read_text())
    assert loaded["m"]["serialized"] == "raw_value"


def test_save_creates_parent_dirs(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    save_checkpoint(out, {"m": {"x": 1}})
    assert (out / "checkpoint.json").exists()


def test_multiple_saves_latest(tmp_path):
    out = tmp_path / "run1"
    out.mkdir()
    save_checkpoint(out, {"m": {"v": 1}})
    save_checkpoint(out, {"m": {"v": 2}})
    loaded = load_checkpoint(tmp_path)
    assert loaded["m"]["v"] == 2


def test_load_empty_dir(tmp_path):
    result = load_checkpoint(tmp_path)
    assert result == {}
