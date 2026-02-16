"""Trend Analysis mode — temporal patterns in enriched data."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from tabs.product_intel.enrichment import get_enriched_traces


def render(base_thread_url: str) -> None:
    """Render Trend Analysis."""
    st.markdown("### 📈 Trend Analysis")
    st.caption("How topics, query types, and success rates shift over time.")

    enriched = get_enriched_traces()
    if not enriched:
        st.info("Run **⚡ Enrichment** first to populate this analysis.")
        return

    df = pd.DataFrame(enriched)

    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        st.warning("No timestamp data available for trend analysis.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if len(df) < 2:
        st.info("Need at least 2 traces with timestamps for trend analysis.")
        return

    # Resolution selector
    date_range_days = (df["timestamp"].max() - df["timestamp"].min()).days
    if date_range_days > 60:
        default_res = "Weekly"
    elif date_range_days > 14:
        default_res = "Weekly"
    else:
        default_res = "Daily"

    resolution = st.radio(
        "Aggregation",
        ["Daily", "Weekly", "Monthly"],
        index=["Daily", "Weekly", "Monthly"].index(default_res),
        horizontal=True,
        key="trend_resolution",
    )

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}
    freq = freq_map[resolution]
    df["period"] = df["timestamp"].dt.to_period(freq[0]).dt.to_timestamp()

    # --- Topic trends ---
    st.markdown("#### Topic trends")
    if "topic" in df.columns:
        topic_trend = df.groupby(["period", "topic"]).size().reset_index(name="count")
        chart = alt.Chart(topic_trend).mark_area(opacity=0.7).encode(
            x=alt.X("period:T", title="Period"),
            y=alt.Y("count:Q", title="Traces", stack="zero"),
            color=alt.Color("topic:N", title="Topic"),
            tooltip=["period:T", "topic:N", "count:Q"],
        ).properties(height=350)
        st.altair_chart(chart, width="stretch")

    # --- Query type trends ---
    st.markdown("#### Query type trends")
    if "query_type" in df.columns:
        qt_trend = df.groupby(["period", "query_type"]).size().reset_index(name="count")
        chart = alt.Chart(qt_trend).mark_area(opacity=0.7).encode(
            x=alt.X("period:T", title="Period"),
            y=alt.Y("count:Q", title="Traces", stack="zero"),
            color=alt.Color("query_type:N", title="Query type"),
            tooltip=["period:T", "query_type:N", "count:Q"],
        ).properties(height=300)
        st.altair_chart(chart, width="stretch")

    # --- Success rate over time ---
    st.markdown("#### Success rate over time")
    if "outcome" in df.columns:
        period_outcomes = df.groupby("period").agg(
            total=("outcome", "size"),
            successes=("outcome", lambda x: (x == "success").sum()),
        ).reset_index()
        period_outcomes["success_rate"] = (
            period_outcomes["successes"] / period_outcomes["total"].clip(lower=1)
        ).round(3)

        chart = alt.Chart(period_outcomes).mark_line(
            point=True, strokeWidth=2
        ).encode(
            x=alt.X("period:T", title="Period"),
            y=alt.Y("success_rate:Q", title="Success rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["period:T", "success_rate:Q", "total:Q"],
        ).properties(height=250)
        st.altair_chart(chart, width="stretch")

    # --- Emerging / declining topics ---
    st.markdown("#### Emerging & declining topics")
    if "topic" in df.columns and len(df["period"].unique()) >= 2:
        periods = sorted(df["period"].unique())
        midpoint = periods[len(periods) // 2]
        first_half = df[df["period"] < midpoint]
        second_half = df[df["period"] >= midpoint]

        if len(first_half) > 0 and len(second_half) > 0:
            first_dist = first_half["topic"].value_counts(normalize=True)
            second_dist = second_half["topic"].value_counts(normalize=True)

            all_topics = set(first_dist.index) | set(second_dist.index)
            changes = []
            for t in all_topics:
                share_1 = first_dist.get(t, 0)
                share_2 = second_dist.get(t, 0)
                if share_1 > 0:
                    pct_change = (share_2 - share_1) / share_1
                elif share_2 > 0:
                    pct_change = 1.0
                else:
                    pct_change = 0.0
                changes.append({
                    "topic": t,
                    "first_half_share": round(share_1, 3),
                    "second_half_share": round(share_2, 3),
                    "change": round(pct_change, 3),
                })

            changes_df = pd.DataFrame(changes).sort_values("change", ascending=False)
            st.dataframe(changes_df, hide_index=True, width="stretch")
