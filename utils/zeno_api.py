"""Zeno API client for tracey's read-path (analytics / sessions / single trace).

Talks to the project-zeno superuser-gated trace endpoints:

- ``GET /api/traces``            list/filter (derived columns; never raw output)
- ``GET /api/traces/analytics``  server-side aggregates over the window
- ``GET /api/traces/sessions``   one row per conversation thread
- ``GET /api/traces/{id}``       full detail (incl. input/output fetched live
                                 from Langfuse, plus ``raw_available``)

Langfuse stays the source of truth for raw traces. This client only consumes the
server-side derived/aggregated data so tracey no longer re-aggregates thousands
of full traces in memory. Auth is a superuser Resource Watch bearer token.

The list/window helpers map Zeno's field names onto tracey's existing dataframe
column names so the downstream pandas/Altair charts work unchanged.
"""

import random
import time as time_mod
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable

import requests

# Zeno's list/sessions endpoints cap ``limit`` at 200.
MAX_PAGE_SIZE = 200
DEFAULT_TIMEOUT_S = 60.0


class ZenoAPIError(Exception):
    """Non-retryable Zeno API failure (e.g. 401/403/404/4xx)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def get_zeno_headers(token: str) -> dict[str, str]:
    """Build the bearer-auth header for Zeno API requests."""
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# Low-level request helper
# --------------------------------------------------------------------------- #
def _request_json(
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retry: int = 4,
    backoff: float = 1.0,
) -> Any:
    """GET ``url`` and return parsed JSON, retrying transient 429/5xx errors.

    Raises ``ZenoAPIError`` for auth/4xx failures (with the status code) and
    after exhausting retries on transient failures.
    """
    attempt = 0
    last_exc: Exception | None = None
    while attempt <= retry:
        try:
            resp = requests.get(
                url, headers=headers, params=params, timeout=timeout_s
            )
        except requests.RequestException as exc:
            last_exc = exc
            sleep_s = backoff * (2**attempt) + random.uniform(0, 0.25)
            time_mod.sleep(sleep_s)
            attempt += 1
            continue

        status = resp.status_code
        if status == 200:
            return resp.json()

        # Auth / client errors are not retryable — surface a clear message.
        if status in (401, 403):
            raise ZenoAPIError(
                "Zeno API needs a superuser token "
                f"(HTTP {status}: {_short(resp.text)})",
                status_code=status,
            )
        if 400 <= status < 500 and status != 429:
            raise ZenoAPIError(
                f"Zeno API request failed (HTTP {status}: {_short(resp.text)})",
                status_code=status,
            )

        # 429 / 5xx — retry with backoff.
        last_exc = ZenoAPIError(
            f"Zeno API transient error (HTTP {status})", status_code=status
        )
        sleep_s = backoff * (2**attempt) + random.uniform(0, 0.25)
        time_mod.sleep(sleep_s)
        attempt += 1

    if isinstance(last_exc, ZenoAPIError):
        raise last_exc
    raise ZenoAPIError(f"Zeno API request failed: {last_exc}")


def _short(text: str, limit: int = 200) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


# --------------------------------------------------------------------------- #
# Time-window helpers
# --------------------------------------------------------------------------- #
def _as_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_dt(s: Any) -> datetime | None:
    """Parse an ISO timestamp (tolerating a trailing ``Z``) into a datetime."""
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    try:
        txt = str(s).replace("Z", "+00:00")
        return datetime.fromisoformat(txt)
    except ValueError:
        return None


def _window_iso(start_date: date, end_date: date) -> tuple[str, str]:
    """Convert an inclusive [start_date, end_date] day range to the API's
    half-open ``[start, end)`` UTC instants (end is the day *after* end_date so
    the whole of end_date is included)."""
    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(
        end_date + timedelta(days=1), time.min, tzinfo=timezone.utc
    )
    return start.isoformat(), end.isoformat()


# --------------------------------------------------------------------------- #
# Field mapping: Zeno list item -> tracey dataframe row
# --------------------------------------------------------------------------- #
def map_trace_row(item: dict[str, Any]) -> dict[str, Any]:
    """Map one Zeno ``TraceListItem`` onto tracey's dataframe columns.

    Numbers are the server-side **per-turn** metrics (correct, and lower than
    old tracey's cumulative ones). Reasoning tokens are N/A on Gemini -> 0.
    ``answer`` is not in the list response; it's fetched via the detail endpoint
    only when a single trace is opened.
    """
    ts = _parse_dt(item.get("trace_timestamp"))
    datasets = item.get("datasets_analysed") or []
    if isinstance(datasets, list):
        datasets_str = ", ".join(str(d) for d in datasets if d)
    else:
        datasets_str = str(datasets or "")
    tool_err = item.get("tool_error_count") or 0

    return {
        "trace_id": item.get("id"),
        "timestamp": ts,
        "date": ts.date() if ts else None,
        "environment": item.get("environment"),
        "session_id": item.get("session_id"),
        "user_id": item.get("user_id"),
        "latency_seconds": _as_float(item.get("latency_seconds")),
        "total_cost": _as_float(item.get("total_cost")),
        "outcome": item.get("outcome"),
        "prompt": item.get("prompt") or "",
        "answer": "",  # not in list payload; see fetch_trace for detail view
        "aoi_name": item.get("aoi_name") or "",
        "aoi_type": item.get("aoi_type") or "",
        "datasets_analysed": datasets_str,
        "tool_call_count": int(item.get("turn_tool_calls") or 0),
        # The list response exposes a single total (turn_tokens), not the
        # input/output split — that's only on the detail endpoint.
        "total_input_tokens": int(item.get("turn_tokens") or 0),
        "total_output_tokens": 0,
        "total_reasoning_tokens": 0,  # Gemini reports no reasoning tokens
        "reasoning_ratio": 0.0,
        "has_internal_error": bool(tool_err and int(tool_err) > 0),
        "primary_dataset_name": item.get("primary_dataset_name"),
        "has_insight": item.get("has_insight"),
        "is_global": item.get("is_global"),
    }


# --------------------------------------------------------------------------- #
# Public fetchers
# --------------------------------------------------------------------------- #
def fetch_traces_window(
    *,
    base_url: str,
    token: str,
    start_date: date,
    end_date: date,
    environment: str | None = None,
    outcome: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    prompt_contains: str | None = None,
    max_traces: int = 25000,
    page_size: int = MAX_PAGE_SIZE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retry: int = 4,
    backoff: float = 1.0,
    on_progress: Callable[[int, int], None] | None = None,
    raw: bool = False,
) -> list[dict[str, Any]]:
    """Paginate ``GET /api/traces`` over [start_date, end_date] and return rows.

    By default rows are mapped to tracey's dataframe columns (see
    :func:`map_trace_row`). Pass ``raw=True`` to get the unmapped Zeno items
    (used by the trace-explorer picker, which also needs ids/prompts as-is).
    """
    start_iso, end_iso = _window_iso(start_date, end_date)
    url = f"{base_url.rstrip('/')}/api/traces"
    headers = get_zeno_headers(token)
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))

    items: list[dict[str, Any]] = []
    offset = 0
    pages = 0
    while True:
        params: dict[str, Any] = {
            "start": start_iso,
            "end": end_iso,
            "limit": page_size,
            "offset": offset,
        }
        if environment:
            params["environment"] = environment
        if outcome:
            params["outcome"] = outcome
        if user_id:
            params["user_id"] = user_id
        if session_id:
            params["session_id"] = session_id
        if prompt_contains:
            params["prompt_contains"] = prompt_contains

        body = _request_json(
            url,
            headers=headers,
            params=params,
            timeout_s=timeout_s,
            retry=retry,
            backoff=backoff,
        )
        batch = body.get("items") or []
        total = int(body.get("total") or 0)
        items.extend(batch)
        pages += 1
        offset += len(batch)

        if on_progress is not None:
            try:
                on_progress(pages, len(items))
            except Exception:
                pass

        if not batch or offset >= total or len(items) >= max_traces:
            break

    if len(items) > max_traces:
        items = items[:max_traces]

    if raw:
        return items
    return [map_trace_row(it) for it in items]


def fetch_analytics(
    *,
    base_url: str,
    token: str,
    start_date: date,
    end_date: date,
    environment: str | None = None,
    outcome: str | None = None,
    user_id: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Server-side aggregates for the window (``GET /api/traces/analytics``)."""
    start_iso, end_iso = _window_iso(start_date, end_date)
    params: dict[str, Any] = {"start": start_iso, "end": end_iso}
    if environment:
        params["environment"] = environment
    if outcome:
        params["outcome"] = outcome
    if user_id:
        params["user_id"] = user_id
    return _request_json(
        f"{base_url.rstrip('/')}/api/traces/analytics",
        headers=get_zeno_headers(token),
        params=params,
        timeout_s=timeout_s,
    )


def fetch_sessions(
    *,
    base_url: str,
    token: str,
    start_date: date,
    end_date: date,
    environment: str | None = None,
    user_id: str | None = None,
    max_sessions: int = 25000,
    page_size: int = MAX_PAGE_SIZE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Paginate ``GET /api/traces/sessions`` (one row per thread, newest first).

    Returns the raw session items: ``session_id, user_id, environment,
    first_prompt, first_timestamp, last_timestamp, turn_count``.
    """
    start_iso, end_iso = _window_iso(start_date, end_date)
    url = f"{base_url.rstrip('/')}/api/traces/sessions"
    headers = get_zeno_headers(token)
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))

    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        params: dict[str, Any] = {
            "start": start_iso,
            "end": end_iso,
            "limit": page_size,
            "offset": offset,
        }
        if environment:
            params["environment"] = environment
        if user_id:
            params["user_id"] = user_id

        body = _request_json(
            url, headers=headers, params=params, timeout_s=timeout_s
        )
        batch = body.get("items") or []
        total = int(body.get("total") or 0)
        items.extend(batch)
        offset += len(batch)
        if not batch or offset >= total or len(items) >= max_sessions:
            break

    return items[:max_sessions]


def fetch_trace(
    *,
    base_url: str,
    token: str,
    trace_id: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Full detail for one trace (``GET /api/traces/{id}``).

    Includes the derived columns plus ``input``/``output`` (the AgentState
    snapshot) fetched live from Langfuse, and ``raw_available`` which is
    ``False`` when Langfuse 404s or is unreachable.
    """
    return _request_json(
        f"{base_url.rstrip('/')}/api/traces/{trace_id}",
        headers=get_zeno_headers(token),
        timeout_s=timeout_s,
    )
