"""JTBD / Use Case Mapping analysis mode."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from tabs.product_intel.enrichment import get_enriched_traces
from tabs.product_intel.shared import render_trace_evidence


def _normalize_intent(s: str) -> str:
    """Normalize intent string for grouping: lowercase, strip, collapse whitespace."""
    return " ".join(str(s).strip().lower().split())


def render(base_thread_url: str) -> None:
    """Render JTBD / Use Case Mapping analysis."""
    st.markdown("### 🎯 Jobs to Be Done")
    st.caption("Cluster user intents into jobs-to-be-done with frequency, success rate, and examples.")

    enriched = get_enriched_traces()
    if not enriched:
        st.info("Run **⚡ Enrichment** first to populate this analysis.")
        return

    df = pd.DataFrame(enriched)

    if "intent" not in df.columns or df["intent"].isna().all():
        st.warning("No intent data found in enriched traces.")
        return

    # Normalize intents for grouping
    df["intent_norm"] = df["intent"].apply(_normalize_intent)
    df = df[df["intent_norm"] != "unknown"]

    if len(df) == 0:
        st.info("All intents are 'unknown'. Enrichment may need better prompts.")
        return

    # --- Top intents by frequency ---
    st.markdown("#### Top intents")
    intent_stats = (
        df.groupby("intent_norm")
        .agg(
            count=("intent_norm", "size"),
            success_count=("outcome", lambda x: (x == "success").sum()),
            failure_count=("outcome", lambda x: x.isin(["failure", "partial"]).sum()),
            example_prompt=("prompt", "first"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    intent_stats["success_rate"] = (
        intent_stats["success_count"] / intent_stats["count"].clip(lower=1)
    ).round(2)

    top_n = min(20, len(intent_stats))
    top_intents = intent_stats.head(top_n)

    # Bar chart colored by outcome
    chart_data = []
    for _, row in top_intents.iterrows():
        chart_data.append({"intent": row["intent_norm"], "outcome": "success", "count": int(row["success_count"])})
        chart_data.append({"intent": row["intent_norm"], "outcome": "failure/partial", "count": int(row["failure_count"])})
        other = int(row["count"]) - int(row["success_count"]) - int(row["failure_count"])
        if other > 0:
            chart_data.append({"intent": row["intent_norm"], "outcome": "unclear", "count": other})

    chart_df = pd.DataFrame(chart_data)
    intent_order = top_intents["intent_norm"].tolist()

    chart = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("count:Q", title="Traces", stack="zero"),
        y=alt.Y("intent:N", sort=intent_order, title=None),
        color=alt.Color(
            "outcome:N",
            scale=alt.Scale(
                domain=["success", "failure/partial", "unclear"],
                range=["#12b76a", "#f04438", "#98a2b3"],
            ),
            title="Outcome",
        ),
        tooltip=["intent:N", "outcome:N", "count:Q"],
    ).properties(height=max(200, top_n * 28))
    st.altair_chart(chart, width="stretch")

    # Intent table
    display_table = top_intents[["intent_norm", "count", "success_rate", "example_prompt"]].copy()
    display_table.columns = ["Intent", "Count", "Success Rate", "Example Prompt"]
    display_table["Example Prompt"] = display_table["Example Prompt"].astype(str).str[:100] + "..."
    st.dataframe(display_table, hide_index=True, width="stretch")

    # Drill-down
    selected = st.selectbox(
        "Drill into intent", ["(none)"] + intent_order,
        key="jtbd_drill",
    )
    if selected != "(none)":
        matching = [e for e in enriched if _normalize_intent(e.get("intent", "")) == selected]
        render_trace_evidence(matching, base_thread_url, f"Traces for: {selected}")

    # --- Complexity distribution ---
    st.markdown("#### Complexity distribution")
    if "complexity" in df.columns:
        cx_counts = df["complexity"].value_counts().reset_index()
        cx_counts.columns = ["complexity", "count"]
        chart = alt.Chart(cx_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("complexity:N", title="Complexity"),
            tooltip=["complexity:N", "count:Q"],
        ).properties(height=250)
        st.altair_chart(chart, width="stretch")

    # --- Complexity × Outcome ---
    st.markdown("#### Complexity × Outcome")
    if "complexity" in df.columns and "outcome" in df.columns:
        cross = df.groupby(["complexity", "outcome"]).size().reset_index(name="count")
        chart = alt.Chart(cross).mark_bar().encode(
            x=alt.X("complexity:N", title="Complexity"),
            y=alt.Y("count:Q", title="Traces", stack="zero"),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(
                    domain=["success", "partial", "failure", "unclear"],
                    range=["#12b76a", "#f79009", "#f04438", "#98a2b3"],
                ),
            ),
            tooltip=["complexity:N", "outcome:N", "count:Q"],
        ).properties(height=300)
        st.altair_chart(chart, width="stretch")
