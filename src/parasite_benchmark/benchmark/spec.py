"""Specification parsing for corpus-generation utilities."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from parasite_benchmark.constants import (
    CATEGORIES,
    EXPECTED_COUNTS,
)
from parasite_benchmark.exceptions import TaskLoadError

_SPEC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "docs" / "specs"
DEFAULT_SPEC_PATH = Path(os.environ.get("PARASITE_SPEC_PATH", str(_SPEC_DIR / "PARASITE_SPEC.md")))

JUDGE_NAME_ALIASES: dict[str, str] = {
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt4.1-mini": "gpt-4.1-mini",
    "gpt 4.1 mini": "gpt-4.1-mini",
    "claude sonnet": "claude-sonnet-4-20250514",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "gemini flash": "google/gemini-2.0-flash",
    "gemini-flash": "google/gemini-2.0-flash",
    "glm-4": "glm-4.7-flash",
    "glm4": "glm-4.7-flash",
    "mistral": "mistral/mistral-large-latest",
    "mistral large": "mistral/mistral-large-latest",
}


@dataclass(frozen=True)
class CategoryDefinition:
    code: str
    name: str
    n_tests: int
    definition: str
    basis: dict[str, str]


@dataclass(frozen=True)
class TestDefinition:
    id: str
    name: str
    category: str
    category_name: str
    category_definition: str
    category_basis: dict[str, str]
    prompt_template: str
    judge_anchors: dict[float, str]

    @property
    def judge_rubric_text(self) -> str:
        lines = [f"{score:.1f}: {text}" for score, text in sorted(self.judge_anchors.items())]
        return "\n".join(lines)


def _clean_md(text: str) -> str:
    return text.replace("**", "").strip()


def _extract_prompt_template(block: str) -> str:
    match = re.search(r"\*\*Prompt Template:\*\*\s*```(.*?)```", block, flags=re.S)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_anchors(block: str) -> dict[float, str]:
    anchors: dict[float, str] = {}
    for m in re.finditer(r"^\s*-\s*(0\.\d|1\.0):\s*(.+?)\s*$", block, flags=re.M):
        anchors[float(m.group(1))] = m.group(2).strip()
    return anchors


def _parse_category_header(line: str) -> tuple[str, str, int] | None:
    cleaned = _clean_md(line)
    if "Category " not in cleaned:
        return None
    m = re.search(r"Category\s+([A-Z])\s+—\s+(.+?)\s+\((\d+)\s+tests\)", cleaned)
    if not m:
        return None
    return m.group(1), m.group(2).strip(), int(m.group(3))


def _parse_test_header(line: str) -> tuple[str, str] | None:
    cleaned = _clean_md(line)
    m = re.match(r"^####\s+([A-Z]\d+):\s+(.+?)(?:\s+\((?:NEW|existing)\))?$", cleaned)
    if not m:
        return None
    return m.group(1), m.group(2).strip()


def parse_spec_file(
    path: Path,
    allowed_categories: set[str] | None = None,
) -> tuple[dict[str, CategoryDefinition], dict[str, TestDefinition]]:
    if not path.exists():
        raise TaskLoadError(f"Spec file not found: {path}")

    lines = path.read_text().splitlines()
    categories: dict[str, CategoryDefinition] = {}
    tests: dict[str, TestDefinition] = {}

    current_category: str | None = None
    current_category_name = ""
    current_category_n = 0
    current_category_definition = ""
    current_basis: dict[str, str] = {}

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        cat = _parse_category_header(stripped)
        if cat:
            code, cat_name, n_tests = cat
            current_category = code
            current_category_name = cat_name
            current_category_n = n_tests
            current_category_definition = ""
            current_basis = {}
            i += 1
            continue

        if current_category:
            cleaned = _clean_md(stripped)
            if cleaned.startswith("Definition:"):
                current_category_definition = cleaned.split(":", 1)[1].strip()
            elif "Basis:" in cleaned and ":" in cleaned:
                prefix, value = cleaned.split(":", 1)
                current_basis[prefix.strip().replace(" Basis", "").lower()] = value.strip()

            parsed_test = _parse_test_header(stripped)
            if parsed_test:
                test_id, test_name = parsed_test
                block_lines: list[str] = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if _parse_test_header(next_line) or _parse_category_header(next_line):
                        break
                    block_lines.append(lines[j])
                    j += 1
                block = "\n".join(block_lines)
                prompt_template = _extract_prompt_template(block)
                anchors = _extract_anchors(block)

                if allowed_categories is None or current_category in allowed_categories:
                    tests[test_id] = TestDefinition(
                        id=test_id,
                        name=test_name,
                        category=current_category,
                        category_name=current_category_name,
                        category_definition=current_category_definition,
                        category_basis=dict(current_basis),
                        prompt_template=prompt_template,
                        judge_anchors=anchors,
                    )
                    if current_category not in categories:
                        categories[current_category] = CategoryDefinition(
                            code=current_category,
                            name=current_category_name,
                            n_tests=current_category_n,
                            definition=current_category_definition,
                            basis=dict(current_basis),
                        )
                i = j
                continue

        i += 1

    return categories, tests


def load_registry(
    spec_path: Path | None = None,
) -> tuple[dict[str, CategoryDefinition], dict[str, TestDefinition]]:
    allowed = set(CATEGORIES)
    path = spec_path or DEFAULT_SPEC_PATH
    all_categories, all_tests = parse_spec_file(path, allowed_categories=allowed)

    counts: dict[str, int] = {}
    for td in all_tests.values():
        counts[td.category] = counts.get(td.category, 0) + 1

    missing_cats = [c for c in CATEGORIES if c not in counts]
    if missing_cats:
        raise TaskLoadError(f"Missing categories in parsed registry: {missing_cats}")

    for cat, expected in EXPECTED_COUNTS.items():
        got = counts.get(cat, 0)
        if got != expected:
            raise TaskLoadError(
                f"Category {cat} expected {expected} tests, found {got}. Check source specs."
            )
    return all_categories, all_tests


def normalize_judge_models(judge_model: str) -> str:
    """Normalize human-readable judge names to runnable model IDs."""
    models = [m.strip() for m in judge_model.split(",") if m.strip()]
    normalized: list[str] = []
    for model in models:
        key = model.lower().strip()
        normalized.append(JUDGE_NAME_ALIASES.get(key, model))
    return ",".join(normalized)
