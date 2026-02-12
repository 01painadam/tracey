"""Tests for utils.data_helpers — format_report_date, normalize_prompt, safe_quantile."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from utils.data_helpers import (
    format_report_date,
    normalize_prompt,
    safe_quantile,
    as_float,
    csv_bytes_any,
)


# ---------------------------------------------------------------------------
# format_report_date
# ---------------------------------------------------------------------------

class TestFormatReportDate:
    def test_date_object(self):
        d = datetime.date(2024, 6, 3)
        result = format_report_date(d)
        assert "3rd" in result
        assert "Jun" in result

    def test_datetime_object(self):
        dt = datetime.datetime(2024, 6, 11, 10, 30)
        result = format_report_date(dt)
        assert "11th" in result
        assert "Jun" in result

    def test_ordinal_st(self):
        assert "1st" in format_report_date(datetime.date(2024, 1, 1))
        assert "21st" in format_report_date(datetime.date(2024, 1, 21))
        assert "31st" in format_report_date(datetime.date(2024, 1, 31))

    def test_ordinal_nd(self):
        assert "2nd" in format_report_date(datetime.date(2024, 1, 2))
        assert "22nd" in format_report_date(datetime.date(2024, 1, 22))

    def test_ordinal_rd(self):
        assert "3rd" in format_report_date(datetime.date(2024, 1, 3))
        assert "23rd" in format_report_date(datetime.date(2024, 1, 23))

    def test_ordinal_th(self):
        for day in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20):
            assert "th" in format_report_date(datetime.date(2024, 1, day))

    def test_teens_always_th(self):
        assert "11th" in format_report_date(datetime.date(2024, 1, 11))
        assert "12th" in format_report_date(datetime.date(2024, 1, 12))
        assert "13th" in format_report_date(datetime.date(2024, 1, 13))

    def test_fallback_on_bad_input(self):
        result = format_report_date("not a date")
        assert result == "not a date"

    def test_none_fallback(self):
        result = format_report_date(None)
        assert result == "None"


# ---------------------------------------------------------------------------
# normalize_prompt
# ---------------------------------------------------------------------------

class TestNormalizePrompt:
    def test_basic(self):
        assert normalize_prompt("Hello World.") == "hello world"

    def test_strips_trailing_dots(self):
        assert normalize_prompt("Test...") == "test"

    def test_collapses_whitespace(self):
        assert normalize_prompt("  foo   bar  baz  ") == "foo bar baz"

    def test_non_string(self):
        assert normalize_prompt(None) == ""
        assert normalize_prompt(123) == ""

    def test_empty_string(self):
        assert normalize_prompt("") == ""
        assert normalize_prompt("   ") == ""

    def test_preserves_non_dot_punctuation(self):
        assert normalize_prompt("Hello, world!") == "hello, world!"

    def test_unicode(self):
        assert normalize_prompt("Où est la forêt?") == "où est la forêt?"


# ---------------------------------------------------------------------------
# safe_quantile
# ---------------------------------------------------------------------------

class TestSafeQuantile:
    def test_basic(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert safe_quantile(s, 0.5) == 3.0

    def test_p95(self):
        s = pd.Series(range(100))
        result = safe_quantile(s, 0.95)
        assert result == pytest.approx(94.05, abs=0.1)

    def test_with_nan(self):
        s = pd.Series([1, 2, None, 4, 5])
        result = safe_quantile(s, 0.5)
        assert result == pytest.approx(3.0, abs=0.5)

    def test_non_numeric(self):
        s = pd.Series(["a", "b", "c"])
        assert safe_quantile(s, 0.5) == 0.0

    def test_empty_series(self):
        assert safe_quantile(pd.Series(dtype=float), 0.5) == 0.0


# ---------------------------------------------------------------------------
# as_float (existing helper, adding coverage)
# ---------------------------------------------------------------------------

class TestAsFloat:
    def test_int(self):
        assert as_float(5) == 5.0

    def test_float(self):
        assert as_float(3.14) == pytest.approx(3.14)

    def test_string_number(self):
        assert as_float("2.5") == 2.5

    def test_none(self):
        assert as_float(None) is None

    def test_bad_string(self):
        assert as_float("not a number") is None


# ---------------------------------------------------------------------------
# csv_bytes_any (existing helper, adding coverage)
# ---------------------------------------------------------------------------

class TestCsvBytesAny:
    def test_basic(self):
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = csv_bytes_any(rows)
        assert isinstance(result, bytes)
        text = result.decode("utf-8")
        assert "a" in text
        assert "b" in text

    def test_empty(self):
        assert csv_bytes_any([]) == b""

    def test_mixed_keys(self):
        rows = [{"a": 1}, {"b": 2}]
        result = csv_bytes_any(rows)
        text = result.decode("utf-8")
        assert "a" in text
        assert "b" in text
