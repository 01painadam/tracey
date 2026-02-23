"""Outcome-classification tests for utils/trace_parsing.py.

These are pure-data tests and should not require Streamlit.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOT = Path(__file__).resolve().parents[1]
tp = _load_module("trace_parsing", ROOT / "utils" / "trace_parsing.py")


def _row(messages):
    return {"output": {"messages": messages}}


def test_hard_error_overrides_non_empty_answer() -> None:
    row = _row(
        [
            {"type": "human", "content": "hi"},
            {"type": "ai", "content": "ok", "status": "error"},
        ]
    )
    assert tp.classify_outcome(row, "ok") == "ERROR"


def test_empty_no_ai_message_is_EMPTY() -> None:
    row = _row([{"type": "human", "content": "hi"}])
    assert tp.classify_outcome(row, "") == "EMPTY"


def test_empty_answer_with_ai_message_is_ERROR() -> None:
    row = _row(
        [
            {"type": "human", "content": "hi"},
            {"type": "ai", "content": ""},
        ]
    )
    assert tp.classify_outcome(row, "") == "ERROR"


def test_soft_error_answer_detected() -> None:
    row = _row(
        [
            {"type": "human", "content": "hi"},
            {"type": "ai", "content": "Sorry, something went wrong."},
        ]
    )
    assert tp.classify_outcome(row, "Sorry, something went wrong.") == "SOFT_ERROR"


def test_defer_when_no_tools_used_in_active_turn() -> None:
    row = _row(
        [
            {"type": "human", "content": "hi"},
            {"type": "ai", "content": "Here is an answer."},
        ]
    )
    assert tp.classify_outcome(row, "Here is an answer.") == "DEFER"


def test_answer_when_tools_used_in_active_turn() -> None:
    row = _row(
        [
            {"type": "human", "content": "hi"},
            {"type": "tool", "name": "pull_data", "content": "{}"},
            {"type": "ai", "content": "Done"},
        ]
    )
    assert tp.classify_outcome(row, "Done") == "ANSWER"


def test_tool_used_earlier_does_not_count_for_active_turn() -> None:
    # Tool used before the last human prompt should not influence the last-turn outcome.
    row = _row(
        [
            {"type": "human", "content": "turn 1"},
            {"type": "tool", "name": "pull_data", "content": "{}"},
            {"type": "ai", "content": "turn 1 answer"},
            {"type": "human", "content": "turn 2"},
            {"type": "ai", "content": "turn 2 answer"},
        ]
    )
    assert tp.classify_outcome(row, "turn 2 answer") == "DEFER"
