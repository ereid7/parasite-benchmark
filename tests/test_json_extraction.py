"""Tests for utils/json_extraction.py — raw, markdown, embedded, malformed."""

from __future__ import annotations

import json

from mbb.utils.json_extraction import extract_json


def test_raw_json_object():
    raw = '{"score": 0.5, "reasoning": "test"}'
    result = extract_json(raw)
    assert json.loads(result)["score"] == 0.5


def test_raw_json_with_whitespace():
    raw = '  {"score": 0.5}  '
    result = extract_json(raw)
    assert json.loads(result)["score"] == 0.5


def test_markdown_json_block():
    text = 'Here is the result:\n```json\n{"score": 0.7}\n```\nDone.'
    result = extract_json(text)
    assert json.loads(result)["score"] == 0.7


def test_markdown_block_no_lang():
    text = 'Result:\n```\n{"score": 0.3}\n```'
    result = extract_json(text)
    assert json.loads(result)["score"] == 0.3


def test_embedded_json_in_prose():
    text = 'The evaluation is {"score": 0.4, "reasoning": "ok"} and that is final.'
    result = extract_json(text)
    assert json.loads(result)["score"] == 0.4


def test_nested_json():
    raw = '{"outer": {"inner": 1}}'
    result = extract_json(raw)
    parsed = json.loads(result)
    assert parsed["outer"]["inner"] == 1


def test_empty_string():
    result = extract_json("")
    assert result == ""


def test_no_json_found():
    text = "This is just plain text with no JSON."
    result = extract_json(text)
    assert result == text


def test_multiline_json():
    text = '```json\n{\n  "score": 0.6,\n  "reasoning": "multi\\nline"\n}\n```'
    result = extract_json(text)
    parsed = json.loads(result)
    assert parsed["score"] == 0.6


def test_json_with_escaped_quotes():
    raw = '{"text": "He said \\"hello\\""}'
    result = extract_json(raw)
    parsed = json.loads(result)
    assert "hello" in parsed["text"]
