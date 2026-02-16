"""Failure Pattern Analysis mode."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from tabs.product_intel.enrichment import get_enriched_traces
from tabs.product_intel.shared import render_trace_evidence


def render(base_thread_url: str) -> None:
    """Render Failure Pattern Analysis."""
    st.markdown("### ❌ Failure Patterns")
    st.caption("Categorise failure modes, identify systematic weaknesses, with trace evidence.")

    enriched = get_enriched_traces()
    if not enriched:
        st.info("Run **⚡ Enrichment** first to populate this analysis.")
        return

    df = pd.DataFrame(enriched)

    if "outcome" not in df.columns:
        st.warning("No outcome data in enriched traces.")
        return

    failures = df[df["outcome"].isin(["failure", "partial"])]
    total = len(df)
    n_fail = len(failures)

    st.metric(
        "Failure/partial rate",
        f"{n_fail}/{total} ({n_fail / max(1, total):.1%})",
    )

    if n_fail == 0:
        st.success("No failures or partial outcomes detected.")
        return

    # --- Failure mode distribution ---
    st.markdown("#### Failure mode distribution")
    if "failure_mode" in failures.columns:
        fm = failures[failures["failure_mode"].notna() & (failures["failure_mode"] != "")]
        if len(fm):
            fm_counts = fm["failure_mode"].value_counts().reset_index()
            fm_counts.columns = ["failure_mode", "count"]

            chart = alt.Chart(fm_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color(
                    "failure_mode:N",
                    scale=alt.Scale(scheme="reds"),
                    title="Failure mode",
                ),
                tooltip=["failure_mode:N", "count:Q"],
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

            # Drill-down
            selected_fm = st.selectbox(
                "Drill into failure mode",
                ["(none)"] + fm_counts["failure_mode"].tolist(),
                key="failure_drill",
            )
            if selected_fm != "(none)":
                matching = [e for e in enriched if e.get("failure_mode") == selected_fm]
                render_trace_evidence(matching, base_thread_url, f"Traces with failure: {selected_fm}")
        else:
            st.info("Failures detected but no specific failure modes assigned.")

    # --- Failure by topic ---
    st.markdown("#### Failures by topic")
    if "topic" in failures.columns and "failure_mode" in failures.columns:
        cross = failures.groupby(["topic", "failure_mode"]).size().reset_index(name="count")
        cross = cross[cross["failure_mode"].notna() & (cross["failure_mode"] != "")]
        if len(cross):
            chart = alt.Chart(cross).mark_bar().encode(
                x=alt.X("count:Q", title="Traces", stack="zero"),
                y=alt.Y("topic:N", sort="-x", title=None),
                color=alt.Color("failure_mode:N", title="Failure mode"),
                tooltip=["topic:N", "failure_mode:N", "count:Q"],
            ).properties(height=max(200, cross["topic"].nunique() * 28))
            st.altair_chart(chart, use_container_width=True)

    # --- Heuristic vs LLM outcome comparison ---
    st.markdown("#### Heuristic vs LLM outcome")
    st.caption("Compare the deterministic outcome classification with the LLM's assessment.")
    if "outcome_heuristic" in df.columns and "outcome" in df.columns:
        confusion = df.groupby(["outcome_heuristic", "outcome"]).size().reset_index(name="count")
        chart = alt.Chart(confusion).mark_rect().encode(
            x=alt.X("outcome:N", title="LLM outcome"),
            y=alt.Y("outcome_heuristic:N", title="Heuristic outcome"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Count"),
            tooltip=["outcome_heuristic:N", "outcome:N", "count:Q"],
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)

        # Disagreements
        disagree = df[df["outcome_heuristic"].apply(
            lambda h: {"ANSWER": "success", "DEFER": "partial", "ERROR": "failure",
                       "SOFT_ERROR": "failure", "EMPTY": "failure"}.get(h, "unclear")
        ) != df["outcome"]]
        if len(disagree):
            st.caption(f"**{len(disagree)}** traces where heuristic and LLM disagree")
            render_trace_evidence(
                disagree.to_dict("records")[:20], base_thread_url,
                "Heuristic/LLM disagreements",
            )

    # --- Failure by complexity ---
    st.markdown("#### Failure rate by complexity")
    if "complexity" in df.columns:
        cx_stats = df.groupby("complexity").agg(
            total=("outcome", "size"),
            failures=("outcome", lambda x: x.isin(["failure", "partial"]).sum()),
        ).reset_index()
        cx_stats["failure_rate"] = (cx_stats["failures"] / cx_stats["total"].clip(lower=1)).round(3)

        chart = alt.Chart(cx_stats).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("complexity:N", title="Complexity",
                     sort=["simple", "moderate", "complex"]),
            y=alt.Y("failure_rate:Q", title="Failure rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("complexity:N", legend=None),
            tooltip=["complexity:N", "failure_rate:Q", "total:Q", "failures:Q"],
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)
