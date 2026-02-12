"""General data processing and formatting utilities."""

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any


def maybe_load_dotenv() -> None:
    """Attempt to load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        return


def iso_utc(dt: datetime) -> str:
    """Convert a datetime to ISO format in UTC."""
    return dt.astimezone(timezone.utc).isoformat()


def as_float(x: Any) -> float | None:
    """Safely convert a value to float."""
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return t


def safe_json_loads(text: str) -> dict[str, Any]:
    """Safely parse JSON, handling code fences and returning a dict."""
    t = strip_code_fences(text)
    try:
        out = json.loads(t)
        return out if isinstance(out, dict) else {"raw": out}
    except Exception:
        return {"raw": t}


def csv_bytes_any(rows: list[dict[str, Any]]) -> bytes:
    """Convert arbitrary dict rows to CSV bytes."""
    if not rows:
        return b""
    fields: list[str] = sorted({k for r in rows for k in r.keys()})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k) for k in fields})
    return buf.getvalue().encode("utf-8")


def format_report_date(d: Any) -> str:
    """Format a date or datetime with ordinal suffix (e.g. 'Mon 3rd Jun')."""
    try:
        if isinstance(d, datetime):
            dt = d
        else:
            dt = datetime(d.year, d.month, d.day)
        day = int(dt.day)
        if 11 <= (day % 100) <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return dt.strftime("%a ") + f"{day}{suffix} " + dt.strftime("%b")
    except Exception:
        return str(d)


def normalize_prompt(s: Any) -> str:
    """Normalize a prompt string for comparison: lowercase, collapse whitespace, strip trailing dots."""
    if not isinstance(s, str):
        return ""
    out = " ".join(s.strip().split()).lower()
    while out.endswith("."):
        out = out[:-1].rstrip()
    return out


def safe_quantile(s: "pd.Series", q: float) -> float:
    """Compute a quantile from a series, coercing to numeric and ignoring NaN."""
    try:
        import pandas as pd
        import math
        numeric = pd.to_numeric(s, errors="coerce").dropna()
        if len(numeric) == 0:
            return 0.0
        result = float(numeric.quantile(q))
        return 0.0 if math.isnan(result) else result
    except Exception:
        return 0.0


def init_session_state(defaults: dict[str, Any]) -> None:
    """Initialize multiple session state keys with defaults if not already set.

    Example:
        init_session_state({
            "my_list": [],
            "my_flag": False,
            "my_count": 0,
        })
    """
    import streamlit as st
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
