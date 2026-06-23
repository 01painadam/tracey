"""Tests for the Zeno API client (field mapping + pagination).

All HTTP is mocked — no real API calls.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from utils import zeno_api


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resp(status_code: int, json_data: Any = None, text: str = "") -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status_code
    r.text = text
    if json_data is not None:
        r.json.return_value = json_data
    else:
        r.json.side_effect = ValueError("no json")
    return r


def _item(i: int, **kw: Any) -> dict[str, Any]:
    base = {
        "id": f"t{i}",
        "session_id": f"s{i}",
        "user_id": f"u{i}",
        "environment": "production",
        "trace_timestamp": "2026-06-01T12:00:00Z",
        "outcome": "ANSWER",
        "prompt": f"prompt {i}",
        "aoi_name": "Brazil",
        "aoi_type": "country",
        "turn_tokens": 100,
        "turn_tool_calls": 2,
        "tool_error_count": 0,
        "latency_seconds": 1.5,
        "total_cost": 0.01,
        "datasets_analysed": ["tree_cover_loss", "land_cover"],
        "language": "en",
        "language_confidence": 0.97,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_window_iso_is_half_open_and_inclusive_of_end_day():
    start_iso, end_iso = zeno_api._window_iso(date(2026, 6, 1), date(2026, 6, 7))
    assert start_iso == datetime(2026, 6, 1, tzinfo=timezone.utc).isoformat()
    # end is the day AFTER end_date at 00:00 so the whole of 2026-06-07 is in range
    assert end_iso == datetime(2026, 6, 8, tzinfo=timezone.utc).isoformat()


def test_map_trace_row_maps_to_tracey_columns():
    row = zeno_api.map_trace_row(_item(1))
    assert row["trace_id"] == "t1"
    assert row["session_id"] == "s1"
    assert row["user_id"] == "u1"
    assert row["outcome"] == "ANSWER"
    assert row["latency_seconds"] == 1.5
    assert row["total_cost"] == 0.01
    # per-turn metrics
    assert row["tool_call_count"] == 2
    assert row["total_input_tokens"] == 100  # turn_tokens
    # Gemini has no reasoning tokens
    assert row["total_reasoning_tokens"] == 0
    assert row["reasoning_ratio"] == 0.0
    # datasets list -> comma-joined string the charts expect
    assert row["datasets_analysed"] == "tree_cover_loss, land_cover"
    # timestamp parsed; date derived
    assert isinstance(row["timestamp"], datetime)
    assert row["date"] == date(2026, 6, 1)


def test_map_trace_row_tool_error_sets_has_internal_error():
    assert zeno_api.map_trace_row(_item(1, tool_error_count=3))["has_internal_error"] is True
    assert zeno_api.map_trace_row(_item(1, tool_error_count=0))["has_internal_error"] is False


def test_map_trace_row_handles_missing_optional_fields():
    row = zeno_api.map_trace_row({"id": "t9", "trace_timestamp": None})
    assert row["trace_id"] == "t9"
    assert row["timestamp"] is None and row["date"] is None
    assert row["datasets_analysed"] == ""
    assert row["tool_call_count"] == 0
    assert row["prompt"] == ""


# --------------------------------------------------------------------------- #
# fetch_traces_window
# --------------------------------------------------------------------------- #
def test_fetch_traces_window_paginates_and_maps():
    page1 = {"total": 3, "limit": 2, "offset": 0, "items": [_item(0), _item(1)]}
    page2 = {"total": 3, "limit": 2, "offset": 2, "items": [_item(2)]}
    with patch("utils.zeno_api.requests.get") as mget:
        mget.side_effect = [_resp(200, page1), _resp(200, page2)]
        rows = zeno_api.fetch_traces_window(
            base_url="https://zeno.example",
            token="tok",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 7),
            page_size=2,
        )
    assert [r["trace_id"] for r in rows] == ["t0", "t1", "t2"]
    assert mget.call_count == 2
    # bearer auth + window params are passed through
    _, kwargs = mget.call_args_list[0]
    assert kwargs["headers"]["Authorization"] == "Bearer tok"
    assert "start" in kwargs["params"] and "end" in kwargs["params"]


def test_fetch_traces_window_respects_max_traces():
    page = {"total": 100, "limit": 2, "offset": 0, "items": [_item(0), _item(1)]}
    with patch("utils.zeno_api.requests.get") as mget:
        mget.return_value = _resp(200, page)
        rows = zeno_api.fetch_traces_window(
            base_url="https://zeno.example",
            token="tok",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 7),
            page_size=2,
            max_traces=1,
        )
    assert len(rows) == 1


def test_fetch_traces_window_raw_returns_unmapped_items():
    page = {"total": 1, "limit": 200, "offset": 0, "items": [_item(0)]}
    with patch("utils.zeno_api.requests.get") as mget:
        mget.return_value = _resp(200, page)
        rows = zeno_api.fetch_traces_window(
            base_url="https://zeno.example",
            token="tok",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 7),
            raw=True,
        )
    assert rows[0]["id"] == "t0"  # raw key, not trace_id
    assert "trace_id" not in rows[0]


def test_forbidden_raises_zeno_api_error_without_retry():
    with patch("utils.zeno_api.requests.get") as mget:
        mget.return_value = _resp(403, text="Superuser privileges required")
        with pytest.raises(zeno_api.ZenoAPIError) as exc:
            zeno_api.fetch_traces_window(
                base_url="https://zeno.example",
                token="tok",
                start_date=date(2026, 6, 1),
                end_date=date(2026, 6, 7),
            )
        assert exc.value.status_code == 403
        assert mget.call_count == 1  # 4xx is not retried


def test_transient_5xx_is_retried_then_raises():
    with patch("utils.zeno_api.requests.get") as mget:
        mget.return_value = _resp(500, text="boom")
        with pytest.raises(zeno_api.ZenoAPIError):
            zeno_api.fetch_traces_window(
                base_url="https://zeno.example",
                token="tok",
                start_date=date(2026, 6, 1),
                end_date=date(2026, 6, 7),
                retry=2,
                backoff=0.0,
            )
        assert mget.call_count == 3  # initial + 2 retries


# --------------------------------------------------------------------------- #
# fetch_sessions
# --------------------------------------------------------------------------- #
def test_fetch_sessions_paginates():
    s = lambda i: {"session_id": f"s{i}", "user_id": "u", "turn_count": 2}
    page1 = {"total": 3, "limit": 2, "offset": 0, "items": [s(0), s(1)]}
    page2 = {"total": 3, "limit": 2, "offset": 2, "items": [s(2)]}
    with patch("utils.zeno_api.requests.get") as mget:
        mget.side_effect = [_resp(200, page1), _resp(200, page2)]
        rows = zeno_api.fetch_sessions(
            base_url="https://zeno.example",
            token="tok",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 7),
            page_size=2,
        )
    assert [r["session_id"] for r in rows] == ["s0", "s1", "s2"]
