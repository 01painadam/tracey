"""Feature Request Extraction mode — identify capability gaps from traces."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from tabs.product_intel.enrichment import get_enriched_traces
from tabs.product_intel.shared import render_trace_evidence
from utils.enrichment_schema import dataset_catalog_keywords


def render(base_thread_url: str) -> None:
    """Render Feature Request Extraction."""
    st.markdown("### 💡 Feature Gaps & Requests")
    st.caption("Identify missing capabilities and unmet dataset demand from trace evidence.")

    enriched = get_enriched_traces()
    if not enriched:
        st.info("Run **⚡ Enrichment** first to populate this analysis.")
        return

    df = pd.DataFrame(enriched)

    # --- Capability gaps (failure_mode = capability_gap) ---
    st.markdown("#### Capability gaps")
    cap_gaps = df[df["failure_mode"] == "capability_gap"] if "failure_mode" in df.columns else pd.DataFrame()

    if len(cap_gaps):
        st.metric("Traces with capability gaps", len(cap_gaps))

        if "query_type" in cap_gaps.columns:
            qt_counts = cap_gaps["query_type"].value_counts().reset_index()
            qt_counts.columns = ["query_type", "count"]
            chart = alt.Chart(qt_counts).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("count:Q", title="Traces"),
                y=alt.Y("query_type:N", sort="-x", title=None),
                tooltip=["query_type:N", "count:Q"],
            ).properties(height=max(150, len(qt_counts) * 28))
            st.altair_chart(chart, use_container_width=True)

        render_trace_evidence(
            cap_gaps.to_dict("records"), base_thread_url,
            "Traces showing capability gaps",
        )
    else:
        st.info("No capability gap failures detected.")

    # --- Missing datasets ---
    st.markdown("#### Missing datasets")
    known_kw = dataset_catalog_keywords()
    known_names_lower = {n.lower() for n in known_kw}

    missing_ds = df[df["failure_mode"] == "missing_dataset"] if "failure_mode" in df.columns else pd.DataFrame()
    if len(missing_ds):
        st.metric("Traces with missing dataset", len(missing_ds))

        if "dataset_required" in missing_ds.columns:
            ds_counts = (
                missing_ds[missing_ds["dataset_required"].notna()]
                ["dataset_required"].value_counts().reset_index()
            )
            ds_counts.columns = ["dataset", "count"]
            ds_counts["in_catalog"] = ds_counts["dataset"].apply(
                lambda d: any(d.lower() in n or n in d.lower() for n in known_names_lower)
            )
            st.dataframe(ds_counts, hide_index=True, use_container_width=True)

        render_trace_evidence(
            missing_ds.to_dict("records"), base_thread_url,
            "Traces requesting missing datasets",
        )
    else:
        st.info("No missing dataset failures detected.")

    # --- Unsupported analysis types ---
    st.markdown("#### Unsupported analysis types")
    if "query_type" in df.columns and "outcome" in df.columns:
        qt_success = df.groupby("query_type").agg(
            total=("outcome", "size"),
            successes=("outcome", lambda x: (x == "success").sum()),
        ).reset_index()
        qt_success["success_rate"] = (
            qt_success["successes"] / qt_success["total"].clip(lower=1)
        ).round(3)
        qt_success = qt_success.sort_values("success_rate")

        # Flag query types with <50% success as potentially unsupported
        weak = qt_success[qt_success["success_rate"] < 0.5]
        if len(weak):
            st.caption("Query types with <50% success rate — may indicate unsupported capabilities.")
            st.dataframe(weak, hide_index=True, use_container_width=True)
        else:
            st.success("All query types have ≥50% success rate.")

    # --- Feature request summary ---
    st.markdown("#### Summary")
    summary_rows = []

    if len(cap_gaps):
        for _, row in cap_gaps.iterrows():
            summary_rows.append({
                "type": "Capability gap",
                "description": str(row.get("intent", ""))[:100],
                "topic": row.get("topic", ""),
                "query_type": row.get("query_type", ""),
            })

    if len(missing_ds):
        for _, row in missing_ds.iterrows():
            summary_rows.append({
                "type": "Missing dataset",
                "description": str(row.get("intent", ""))[:100],
                "topic": row.get("topic", ""),
                "dataset": row.get("dataset_required", ""),
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        # Group and count
        if "description" in summary_df.columns:
            grouped = summary_df.groupby(["type", "topic"]).agg(
                count=("description", "size"),
                example=("description", "first"),
            ).reset_index().sort_values("count", ascending=False)
            st.dataframe(grouped, hide_index=True, use_container_width=True)
    else:
        st.info("No feature gaps identified. This is good news — or enrichment data may be limited.")
