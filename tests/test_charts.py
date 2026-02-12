"""Tests for utils.charts — every public chart function.

Each test verifies:
- The function returns the expected Altair type (Chart, LayerChart, or None).
- Dynamic bar heights scale correctly.
- Edge cases (empty data, single row) don't crash.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import pytest

from utils.charts import (
    aoi_name_bar,
    aoi_type_pie,
    category_pie_chart,
    cost_histogram,
    daily_cost_chart,
    daily_latency_chart,
    daily_outcome_chart,
    daily_volume_chart,
    dynamic_bar_height,
    language_bar_chart,
    latency_histogram,
    outcome_pie_chart,
    prompt_length_histogram,
    prompt_utilisation_daily_chart,
    prompt_utilisation_histogram,
    reasoning_tokens_histogram,
    simple_pie_chart,
    starter_breakdown_pie,
    starter_vs_other_pie,
    success_rate_bar_chart,
    tool_calls_vs_latency_chart,
    tool_flow_arc_chart,
    tool_flow_sankey_data,
    tool_success_rate_chart,
    user_segment_bar_chart,
    user_segment_pie_chart,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal DataFrames used across tests
# ---------------------------------------------------------------------------

def _daily_metrics(n_days: int = 5) -> pd.DataFrame:
    """Build a minimal daily_metrics DataFrame."""
    import datetime
    dates = [datetime.date(2024, 6, d + 1) for d in range(n_days)]
    return pd.DataFrame({
        "date": dates,
        "traces": [10 + i for i in range(n_days)],
        "unique_users": [3 + i for i in range(n_days)],
        "unique_threads": [5 + i for i in range(n_days)],
        "success_rate": [0.8] * n_days,
        "defer_rate": [0.05] * n_days,
        "soft_error_rate": [0.05] * n_days,
        "error_rate": [0.05] * n_days,
        "empty_rate": [0.05] * n_days,
        "mean_cost": [0.01] * n_days,
        "p95_cost": [0.03] * n_days,
        "mean_latency": [5.0] * n_days,
        "p95_latency": [12.0] * n_days,
    })


def _trace_df(n: int = 20) -> pd.DataFrame:
    """Build a minimal trace DataFrame with outcome and lang_query columns."""
    outcomes = ["ANSWER"] * 14 + ["DEFER"] * 2 + ["ERROR"] * 2 + ["SOFT_ERROR"] * 1 + ["EMPTY"] * 1
    return pd.DataFrame({
        "outcome": outcomes[:n],
        "lang_query": (["en"] * 10 + ["fr"] * 5 + ["de"] * 3 + ["es"] * 2)[:n],
        "lang_query_conf": [0.9] * n,
        "total_cost": [0.01 * (i + 1) for i in range(n)],
        "latency_seconds": [3.0 + i * 0.5 for i in range(n)],
        "tool_call_count": list(range(n)),
        "has_internal_error": [False] * (n - 2) + [True] * 2,
    })


# ---------------------------------------------------------------------------
# dynamic_bar_height
# ---------------------------------------------------------------------------

class TestDynamicBarHeight:
    def test_minimum(self):
        assert dynamic_bar_height(1) >= 180

    def test_scales(self):
        assert dynamic_bar_height(20) > dynamic_bar_height(5)

    def test_maximum_cap(self):
        assert dynamic_bar_height(1000) <= 800

    def test_custom_params(self):
        h = dynamic_bar_height(10, bar_px=50, min_px=100, max_px=400)
        assert h == 400  # 10 * 50 = 500, capped at 400

    def test_zero_entries(self):
        assert dynamic_bar_height(0) == 180


# ---------------------------------------------------------------------------
# Daily trend charts
# ---------------------------------------------------------------------------

class TestDailyVolumeChart:
    def test_returns_chart(self):
        c = daily_volume_chart(_daily_metrics())
        assert isinstance(c, alt.Chart)

    def test_single_day(self):
        c = daily_volume_chart(_daily_metrics(1))
        assert isinstance(c, alt.Chart)


class TestDailyOutcomeChart:
    def test_returns_chart(self):
        c = daily_outcome_chart(_daily_metrics())
        assert isinstance(c, alt.Chart)

    def test_custom_order(self):
        c = daily_outcome_chart(
            _daily_metrics(),
            outcome_order=["Error", "Error (Empty)", "Defer", "Soft error", "Success"],
        )
        assert isinstance(c, alt.Chart)

    def test_custom_colors(self):
        c = daily_outcome_chart(
            _daily_metrics(),
            outcome_colors={"Success": "#00FF00"},
        )
        assert isinstance(c, alt.Chart)


class TestDailyCostChart:
    def test_returns_chart(self):
        c = daily_cost_chart(_daily_metrics())
        assert isinstance(c, alt.Chart)


class TestDailyLatencyChart:
    def test_returns_chart(self):
        c = daily_latency_chart(_daily_metrics())
        assert isinstance(c, alt.Chart)


# ---------------------------------------------------------------------------
# Distribution charts (pie / histogram)
# ---------------------------------------------------------------------------

class TestOutcomePieChart:
    def test_returns_chart(self):
        c = outcome_pie_chart(_trace_df())
        assert isinstance(c, alt.Chart)

    def test_single_outcome(self):
        df = pd.DataFrame({"outcome": ["ANSWER"] * 5})
        c = outcome_pie_chart(df)
        assert isinstance(c, alt.Chart)


class TestLanguageBarChart:
    def test_returns_chart(self):
        c = language_bar_chart(_trace_df())
        assert isinstance(c, alt.Chart)

    def test_no_lang_column(self):
        df = pd.DataFrame({"outcome": ["ANSWER"]})
        assert language_bar_chart(df) is None

    def test_empty_lang(self):
        df = pd.DataFrame({"lang_query": ["", None]})
        assert language_bar_chart(df) is None

    def test_top_n(self):
        df = _trace_df()
        c = language_bar_chart(df, top_n=2)
        assert isinstance(c, alt.Chart)


class TestLatencyHistogram:
    def test_returns_chart(self):
        c = latency_histogram(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert isinstance(c, alt.Chart)

    def test_empty_series(self):
        assert latency_histogram(pd.Series(dtype=float)) is None


class TestCostHistogram:
    def test_returns_chart(self):
        c = cost_histogram(pd.Series([0.01, 0.02, 0.03]))
        assert isinstance(c, alt.Chart)

    def test_empty_series(self):
        assert cost_histogram(pd.Series(dtype=float)) is None


class TestCategoryPieChart:
    def test_returns_chart(self):
        s = pd.Series(["a", "b", "a", "c", "a"])
        c = category_pie_chart(s, "cat", "Categories")
        assert isinstance(c, alt.Chart)

    def test_explode_csv(self):
        s = pd.Series(["a,b", "b,c", "a"])
        c = category_pie_chart(s, "cat", "Categories", explode_csv=True)
        assert isinstance(c, alt.Chart)

    def test_empty(self):
        s = pd.Series(["", None])
        assert category_pie_chart(s, "cat", "Categories") is None


class TestReasoningTokensHistogram:
    def test_returns_chart(self):
        c = reasoning_tokens_histogram(pd.Series([0.1, 0.5, 0.8]))
        assert isinstance(c, alt.Chart)

    def test_empty(self):
        assert reasoning_tokens_histogram(pd.Series(dtype=float)) is None

    def test_negative_filtered(self):
        assert reasoning_tokens_histogram(pd.Series([-0.1, -0.5])) is None


# ---------------------------------------------------------------------------
# Tool charts
# ---------------------------------------------------------------------------

class TestToolSuccessRateChart:
    def test_returns_chart(self):
        df = pd.DataFrame({
            "tool_name": ["tool_a", "tool_b"],
            "total": [10, 5],
            "success": [8, 3],
            "ambiguity": [1, 1],
            "semantic_error": [0, 1],
            "error": [1, 0],
        })
        c = tool_success_rate_chart(df)
        assert isinstance(c, alt.Chart)

    def test_empty(self):
        df = pd.DataFrame(columns=["tool_name", "total", "success", "ambiguity", "semantic_error", "error"])
        assert tool_success_rate_chart(df) is None


class TestToolCallsVsLatencyChart:
    def test_returns_layer(self):
        df = _trace_df()
        c = tool_calls_vs_latency_chart(df)
        assert isinstance(c, alt.LayerChart)

    def test_empty(self):
        df = pd.DataFrame(columns=["tool_call_count", "latency_seconds"])
        assert tool_calls_vs_latency_chart(df) is None

    def test_missing_columns(self):
        df = pd.DataFrame({"x": [1]})
        assert tool_calls_vs_latency_chart(df) is None


class TestSuccessRateBarChart:
    def test_returns_chart(self):
        df = pd.DataFrame({
            "env": ["prod", "staging"],
            "traces": [100, 50],
            "success_rate": [0.8, 0.6],
            "defer_rate": [0.05, 0.1],
            "soft_error_rate": [0.05, 0.1],
            "error_rate": [0.05, 0.1],
            "empty_rate": [0.05, 0.1],
        })
        c = success_rate_bar_chart(df, "env", "Environment")
        assert isinstance(c, alt.Chart)

    def test_empty(self):
        df = pd.DataFrame(columns=["env", "traces", "success_rate", "defer_rate", "soft_error_rate", "error_rate", "empty_rate"])
        assert success_rate_bar_chart(df, "env", "Environment") is None


# ---------------------------------------------------------------------------
# Tool flow
# ---------------------------------------------------------------------------

class TestToolFlowSankeyData:
    def test_basic_flow(self):
        def _extract(t):
            return t.get("flow", [])
        traces = [
            {"flow": [("tool_a", "success"), ("tool_b", "success")]},
            {"flow": [("tool_a", "error")]},
        ]
        df = tool_flow_sankey_data(traces, _extract)
        assert len(df) > 0
        assert set(df.columns) == {"source", "target", "status", "count"}

    def test_empty_traces(self):
        df = tool_flow_sankey_data([], lambda t: [])
        assert df.empty

    def test_no_flows(self):
        df = tool_flow_sankey_data([{"x": 1}], lambda t: [])
        assert df.empty


class TestToolFlowArcChart:
    def test_returns_chart(self):
        df = pd.DataFrame({
            "source": ["START", "tool_a"],
            "target": ["tool_a", "END"],
            "status": ["success", "success"],
            "count": [5, 5],
        })
        c = tool_flow_arc_chart(df)
        assert c is not None

    def test_empty(self):
        assert tool_flow_arc_chart(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# Prompt utilisation charts
# ---------------------------------------------------------------------------

class TestPromptUtilisationHistogram:
    def test_returns_chart(self):
        df = pd.DataFrame({"prompts": [1, 2, 3, 1, 5, 2]})
        c = prompt_utilisation_histogram(df)
        assert isinstance(c, alt.Chart)


class TestPromptUtilisationDailyChart:
    def test_returns_layer(self):
        import datetime
        df = pd.DataFrame({
            "date": [datetime.date(2024, 6, d) for d in range(1, 4)],
            "mean_prompts_per_user": [3.0, 4.0, 5.0],
            "median_prompts_per_user": [2.0, 3.0, 4.0],
            "p95_prompts_per_user": [8.0, 9.0, 10.0],
        })
        c = prompt_utilisation_daily_chart(df)
        assert isinstance(c, alt.LayerChart)

    def test_custom_limit(self):
        import datetime
        df = pd.DataFrame({
            "date": [datetime.date(2024, 6, 1)],
            "mean_prompts_per_user": [3.0],
            "median_prompts_per_user": [2.0],
            "p95_prompts_per_user": [8.0],
        })
        c = prompt_utilisation_daily_chart(df, prompt_limit=5)
        assert isinstance(c, alt.LayerChart)


# ---------------------------------------------------------------------------
# User segment charts
# ---------------------------------------------------------------------------

class TestUserSegmentBarChart:
    def test_returns_chart(self):
        import datetime
        df = pd.DataFrame({
            "date": [datetime.date(2024, 6, d) for d in range(1, 4)],
            "new_users": [5, 2, 0],
            "returning_users": [10, 12, 15],
        })
        c = user_segment_bar_chart(df, "new_users", "returning_users", "New", "Returning", "#aaa", "#bbb")
        assert isinstance(c, alt.Chart)

    def test_single_day(self):
        import datetime
        df = pd.DataFrame({
            "date": [datetime.date(2024, 6, 1)],
            "a": [5],
            "b": [10],
        })
        c = user_segment_bar_chart(df, "a", "b", "A", "B", "#aaa", "#bbb")
        assert isinstance(c, alt.Chart)


class TestUserSegmentPieChart:
    def test_returns_chart(self):
        c = user_segment_pie_chart("New", "Returning", 10, 40, "#aaa", "#bbb")
        assert isinstance(c, alt.Chart)

    def test_one_zero(self):
        c = user_segment_pie_chart("New", "Returning", 0, 40, "#aaa", "#bbb")
        assert isinstance(c, alt.Chart)

    def test_both_zero(self):
        c = user_segment_pie_chart("New", "Returning", 0, 0, "#aaa", "#bbb")
        assert isinstance(c, alt.Chart)


# ---------------------------------------------------------------------------
# Starter prompt charts
# ---------------------------------------------------------------------------

class TestStarterVsOtherPie:
    def test_returns_chart(self):
        c = starter_vs_other_pie(10, 90)
        assert isinstance(c, alt.Chart)

    def test_zero_starter(self):
        c = starter_vs_other_pie(0, 50)
        assert isinstance(c, alt.Chart)


class TestStarterBreakdownPie:
    def test_returns_chart(self):
        df = pd.DataFrame({"label": ["A", "B", "C"], "count": [5, 3, 2]})
        c = starter_breakdown_pie(df)
        assert isinstance(c, alt.Chart)

    def test_single_entry(self):
        df = pd.DataFrame({"label": ["A"], "count": [10]})
        c = starter_breakdown_pie(df)
        assert isinstance(c, alt.Chart)


# ---------------------------------------------------------------------------
# Prompt length histogram
# ---------------------------------------------------------------------------

class TestPromptLengthHistogram:
    def test_chars(self):
        s = pd.Series([10, 50, 100, 200])
        c = prompt_length_histogram(s, "prompt_len_chars", "Prompt length (characters)")
        assert isinstance(c, alt.Chart)

    def test_words(self):
        s = pd.Series([2, 5, 10, 20])
        c = prompt_length_histogram(s, "prompt_len_words", "Prompt length (words)")
        assert isinstance(c, alt.Chart)


# ---------------------------------------------------------------------------
# AOI charts
# ---------------------------------------------------------------------------

class TestAoiTypePie:
    def test_returns_chart(self):
        df = pd.DataFrame({
            "aoi_type": ["country", "admin"],
            "count": [30, 10],
            "percent": [75.0, 25.0],
        })
        c = aoi_type_pie(df, ["country", "admin"])
        assert isinstance(c, alt.Chart)

    def test_no_domain(self):
        df = pd.DataFrame({
            "aoi_type": ["country"],
            "count": [10],
            "percent": [100.0],
        })
        c = aoi_type_pie(df)
        assert isinstance(c, alt.Chart)


class TestAoiNameBar:
    def test_returns_chart(self):
        df = pd.DataFrame({
            "aoi_name": ["Kenya", "France", "Brazil"],
            "count": [10, 8, 5],
        })
        c = aoi_name_bar(df)
        assert isinstance(c, alt.Chart)

    def test_with_aoi_type(self):
        df = pd.DataFrame({
            "aoi_name": ["Kenya", "France"],
            "count": [10, 8],
            "aoi_type": ["country", "country"],
        })
        c = aoi_name_bar(df, aoi_type_domain=["country"])
        assert isinstance(c, alt.Chart)

    def test_dynamic_height(self):
        df = pd.DataFrame({
            "aoi_name": [f"place_{i}" for i in range(30)],
            "count": list(range(30, 0, -1)),
        })
        c = aoi_name_bar(df)
        # 30 entries * 22px = 660, which should be > 200 min
        assert isinstance(c, alt.Chart)


# ---------------------------------------------------------------------------
# Simple pie chart (generic helper)
# ---------------------------------------------------------------------------

class TestSimplePieChart:
    def test_returns_chart(self):
        df = pd.DataFrame({"label": ["A", "B"], "count": [10, 5]})
        c = simple_pie_chart(df)
        assert isinstance(c, alt.Chart)

    def test_filters_zeros(self):
        df = pd.DataFrame({"label": ["A", "B", "C"], "count": [10, 0, 5]})
        c = simple_pie_chart(df)
        assert isinstance(c, alt.Chart)

    def test_custom_columns(self):
        df = pd.DataFrame({"status": ["ok", "fail"], "transitions": [8, 2]})
        c = simple_pie_chart(df, label_col="status", count_col="transitions")
        assert isinstance(c, alt.Chart)

    def test_all_zero(self):
        df = pd.DataFrame({"label": ["A"], "count": [0]})
        c = simple_pie_chart(df)
        assert isinstance(c, alt.Chart)
