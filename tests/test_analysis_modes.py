"""Tests for analysis mode data transformations.

Tests the pure data logic without Streamlit rendering.
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures — synthetic enriched traces
# ---------------------------------------------------------------------------

def _make_enriched(n: int = 20, **overrides) -> list[dict[str, Any]]:
    """Build synthetic enriched trace data."""
    topics = ["deforestation", "land_cover_change", "fire_disturbance", "restoration", "biodiversity"]
    query_types = ["spatial_analysis", "trend_analysis", "comparison", "data_lookup", "explanation"]
    outcomes = ["success", "success", "success", "partial", "failure"]
    complexities = ["simple", "moderate", "complex"]
    failure_modes = [None, None, None, "missing_dataset", "capability_gap"]

    traces = []
    base_date = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    for i in range(n):
        trace = {
            "trace_id": f"t{i}",
            "timestamp": base_date + datetime.timedelta(days=i),
            "session_id": f"s{i % 5}",
            "user_id": f"u{i % 3}",
            "prompt": f"Test prompt {i}",
            "answer": f"Test answer {i}",
            "topic": topics[i % len(topics)],
            "query_type": query_types[i % len(query_types)],
            "outcome": outcomes[i % len(outcomes)],
            "complexity": complexities[i % len(complexities)],
            "failure_mode": failure_modes[i % len(failure_modes)],
            "dataset_required": f"dataset_{i % 3}" if i % 4 != 0 else None,
            "geographic_scope": f"Country_{i % 4}" if i % 3 != 0 else None,
            "intent": f"intent_{i % 7}",
            "outcome_heuristic": "ANSWER" if i % 5 != 4 else "ERROR",
            "tools_used": ["pull_data"] if i % 2 == 0 else [],
            "datasets_analysed": ["tree_cover_loss"] if i % 3 == 0 else [],
            "aoi_name": "",
            "aoi_type": "",
            "is_welcome_prompt": i < 3,  # first 3 are welcome prompts
            **overrides,
        }
        traces.append(trace)
    return traces


# ---------------------------------------------------------------------------
# Topic Patterns
# ---------------------------------------------------------------------------

class TestTopicPatterns:
    def test_topic_distribution(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        counts = df["topic"].value_counts()
        assert len(counts) == 5  # 5 unique topics
        assert counts.sum() == 20

    def test_dataset_demand_counts(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        demand = df[df["dataset_required"].notna()]
        assert len(demand) == 15  # 20 - 5 where i%4==0

    def test_organic_vs_welcome_split(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        organic = df[~df["is_welcome_prompt"]]
        welcome = df[df["is_welcome_prompt"]]
        assert len(welcome) == 3
        assert len(organic) == 17

    def test_geographic_scope(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        geo = df[df["geographic_scope"].notna()]
        geo_counts = geo["geographic_scope"].value_counts()
        assert len(geo_counts) > 0


# ---------------------------------------------------------------------------
# JTBD
# ---------------------------------------------------------------------------

class TestJTBD:
    def test_intent_grouping(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        intent_counts = df["intent"].value_counts()
        assert len(intent_counts) == 7  # 7 unique intents (i%7)

    def test_intent_success_rate(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        stats = df.groupby("intent").agg(
            total=("outcome", "size"),
            successes=("outcome", lambda x: (x == "success").sum()),
        ).reset_index()
        stats["success_rate"] = stats["successes"] / stats["total"]
        assert (stats["success_rate"] >= 0).all()
        assert (stats["success_rate"] <= 1).all()

    def test_complexity_distribution(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        cx = df["complexity"].value_counts()
        assert set(cx.index).issubset({"simple", "moderate", "complex"})


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------

class TestTrends:
    def test_topic_trends_over_time(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["period"] = df["timestamp"].dt.to_period("W").dt.to_timestamp()
        topic_trend = df.groupby(["period", "topic"]).size().reset_index(name="count")
        assert len(topic_trend) > 0
        assert "count" in topic_trend.columns

    def test_success_rate_over_time(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["period"] = df["timestamp"].dt.to_period("W").dt.to_timestamp()
        period_stats = df.groupby("period").agg(
            total=("outcome", "size"),
            successes=("outcome", lambda x: (x == "success").sum()),
        ).reset_index()
        period_stats["success_rate"] = period_stats["successes"] / period_stats["total"]
        assert (period_stats["success_rate"] >= 0).all()
        assert (period_stats["success_rate"] <= 1).all()

    def test_emerging_topics(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["period"] = df["timestamp"].dt.to_period("W").dt.to_timestamp()
        periods = sorted(df["period"].unique())
        assert len(periods) >= 2  # 20 days should span multiple weeks


# ---------------------------------------------------------------------------
# Failure Patterns
# ---------------------------------------------------------------------------

class TestFailurePatterns:
    def test_failure_mode_distribution(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        failures = df[df["outcome"].isin(["failure", "partial"])]
        assert len(failures) > 0
        fm = failures[failures["failure_mode"].notna()]
        assert len(fm) > 0

    def test_heuristic_vs_llm_comparison(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        confusion = df.groupby(["outcome_heuristic", "outcome"]).size().reset_index(name="count")
        assert len(confusion) > 0

    def test_failure_rate_by_complexity(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        cx_stats = df.groupby("complexity").agg(
            total=("outcome", "size"),
            failures=("outcome", lambda x: x.isin(["failure", "partial"]).sum()),
        ).reset_index()
        cx_stats["failure_rate"] = cx_stats["failures"] / cx_stats["total"].clip(lower=1)
        assert (cx_stats["failure_rate"] >= 0).all()
        assert (cx_stats["failure_rate"] <= 1).all()


# ---------------------------------------------------------------------------
# Feature Requests
# ---------------------------------------------------------------------------

class TestFeatureRequests:
    def test_capability_gaps_identified(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        cap_gaps = df[df["failure_mode"] == "capability_gap"]
        assert len(cap_gaps) > 0

    def test_missing_datasets_identified(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        missing = df[df["failure_mode"] == "missing_dataset"]
        assert len(missing) > 0

    def test_query_type_success_rates(self):
        traces = _make_enriched(20)
        df = pd.DataFrame(traces)
        qt_stats = df.groupby("query_type").agg(
            total=("outcome", "size"),
            successes=("outcome", lambda x: (x == "success").sum()),
        ).reset_index()
        qt_stats["success_rate"] = qt_stats["successes"] / qt_stats["total"].clip(lower=1)
        assert len(qt_stats) == 5  # 5 query types


# ---------------------------------------------------------------------------
# Report data building
# ---------------------------------------------------------------------------

class TestReportDataBuilding:
    def test_build_analysis_data(self):
        from tabs.product_intel.report import _build_analysis_data

        traces = _make_enriched(20)
        data = _build_analysis_data(traces)
        assert isinstance(data, str)
        assert "TOTAL TRACES: 20" in data
        assert "TOPIC DISTRIBUTION" in data
        assert "OUTCOME DISTRIBUTION" in data
        assert "REPRESENTATIVE TRACES" in data

    def test_analysis_data_includes_welcome_split(self):
        from tabs.product_intel.report import _build_analysis_data

        traces = _make_enriched(20)
        data = _build_analysis_data(traces)
        assert "organic: 17" in data
        assert "welcome prompt: 3" in data

    def test_analysis_data_includes_session_stats(self):
        from tabs.product_intel.report import _build_analysis_data

        traces = _make_enriched(20)
        data = _build_analysis_data(traces)
        assert "TOTAL SESSIONS" in data
        assert "BOUNCE SESSIONS" in data
