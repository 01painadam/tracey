"""Topic & Dataset Patterns analysis mode."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from tabs.product_intel.enrichment import get_enriched_traces
from tabs.product_intel.shared import render_trace_evidence
from utils.enrichment_schema import dataset_catalog_keywords


def render(base_thread_url: str) -> None:
    """Render Topic & Dataset Patterns analysis."""
    st.markdown("### 📊 Topic & Dataset Patterns")
    st.caption("What users ask about, what datasets they need, and how queries are distributed.")

    enriched = get_enriched_traces()
    if not enriched:
        st.info("Run **⚡ Enrichment** first to populate this analysis.")
        return

    df = pd.DataFrame(enriched)

    # --- Topic distribution ---
    st.markdown("#### Topic distribution")
    if "topic" in df.columns:
        topic_counts = df["topic"].value_counts().reset_index()
        topic_counts.columns = ["topic", "count"]

        chart = alt.Chart(topic_counts).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("count:Q", title="Traces"),
            y=alt.Y("topic:N", sort="-x", title=None),
            color=alt.Color("topic:N", legend=None),
            tooltip=["topic:N", "count:Q"],
        ).properties(height=max(180, len(topic_counts) * 28))
        st.altair_chart(chart, use_container_width=True)

        # Drill-down
        selected_topic = st.selectbox(
            "Drill into topic", ["(none)"] + topic_counts["topic"].tolist(),
            key="topic_drill",
        )
        if selected_topic != "(none)":
            matching = [e for e in enriched if e.get("topic") == selected_topic]
            render_trace_evidence(matching, base_thread_url, f"Traces for topic: {selected_topic}")

    # --- Dataset demand vs supply ---
    st.markdown("#### Dataset demand vs supply")
    if "dataset_required" in df.columns:
        demand = df[df["dataset_required"].notna() & (df["dataset_required"] != "")]
        if len(demand):
            demand_counts = demand["dataset_required"].value_counts().reset_index()
            demand_counts.columns = ["dataset", "demand_count"]

            # Separate organic vs welcome prompt
            if "is_welcome_prompt" in df.columns:
                organic = demand[~demand["is_welcome_prompt"]]
                welcome = demand[demand["is_welcome_prompt"]]
                organic_counts = organic["dataset_required"].value_counts().reset_index()
                organic_counts.columns = ["dataset", "organic"]
                welcome_counts = welcome["dataset_required"].value_counts().reset_index()
                welcome_counts.columns = ["dataset", "welcome"]
                demand_counts = demand_counts.merge(organic_counts, on="dataset", how="left")
                demand_counts = demand_counts.merge(welcome_counts, on="dataset", how="left")
                demand_counts = demand_counts.fillna(0)

            # Compare against known datasets
            known_kw = dataset_catalog_keywords()
            known_names_lower = {n.lower() for n in known_kw}
            demand_counts["in_catalog"] = demand_counts["dataset"].apply(
                lambda d: any(
                    d.lower() in n or n in d.lower()
                    for n in known_names_lower
                )
            )

            st.dataframe(demand_counts, hide_index=True, use_container_width=True)

            # Unmet demand
            unmet = demand_counts[~demand_counts["in_catalog"]]
            if len(unmet):
                st.markdown("##### ⚠️ Unmet dataset demand")
                st.caption("Datasets users ask for that don't match the known catalog.")
                st.dataframe(unmet[["dataset", "demand_count"]], hide_index=True, use_container_width=True)
        else:
            st.info("No dataset requirements identified in enriched traces.")

    # --- Query type distribution ---
    st.markdown("#### Query type distribution")
    if "query_type" in df.columns:
        qt_counts = df["query_type"].value_counts().reset_index()
        qt_counts.columns = ["query_type", "count"]
        chart = alt.Chart(qt_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("query_type:N", title="Query type"),
            tooltip=["query_type:N", "count:Q"],
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    # --- Geographic scope ---
    st.markdown("#### Geographic scope")
    if "geographic_scope" in df.columns:
        geo = df[df["geographic_scope"].notna() & (df["geographic_scope"] != "")]
        if len(geo):
            geo_counts = geo["geographic_scope"].value_counts().head(20).reset_index()
            geo_counts.columns = ["geography", "count"]
            chart = alt.Chart(geo_counts).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("count:Q", title="Traces"),
                y=alt.Y("geography:N", sort="-x", title=None),
                tooltip=["geography:N", "count:Q"],
            ).properties(height=max(180, len(geo_counts) * 28))
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No geographic scope data in enriched traces.")

    # --- Topic × Query type heatmap ---
    st.markdown("#### Topic × Query type")
    if "topic" in df.columns and "query_type" in df.columns:
        cross = df.groupby(["topic", "query_type"]).size().reset_index(name="count")
        if len(cross):
            chart = alt.Chart(cross).mark_rect().encode(
                x=alt.X("query_type:N", title="Query type"),
                y=alt.Y("topic:N", title="Topic"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Count"),
                tooltip=["topic:N", "query_type:N", "count:Q"],
            ).properties(height=max(200, df["topic"].nunique() * 28))
            st.altair_chart(chart, use_container_width=True)
