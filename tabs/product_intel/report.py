"""Report Generator — evidence-backed markdown from enriched data using Claude CLI."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from tabs.product_intel.enrichment import get_enriched_traces
from utils.claude_cli import call_claude, is_claude_available
from utils.enrichment_schema import dataset_catalog_summary
from utils.prompt_fixtures import DEFAULT_REPORT_PROMPT


def _build_analysis_data(enriched: list[dict[str, Any]]) -> str:
    """Build structured analysis data for the report prompt.

    Aggregates enrichment results into a compact summary with
    representative trace examples, suitable for a single LLM call.
    """
    df = pd.DataFrame(enriched)
    sections: list[str] = []

    # --- Basic stats ---
    total = len(df)
    n_welcome = int(df["is_welcome_prompt"].sum()) if "is_welcome_prompt" in df.columns else 0
    n_organic = total - n_welcome
    sections.append(f"TOTAL TRACES: {total} (organic: {n_organic}, welcome prompt: {n_welcome})")

    # --- Date range ---
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
        if len(ts):
            sections.append(f"DATE RANGE: {ts.min().date()} to {ts.max().date()}")

    # --- Topic distribution ---
    if "topic" in df.columns:
        topic_counts = df["topic"].value_counts().to_dict()
        sections.append(f"TOPIC DISTRIBUTION: {json.dumps(topic_counts)}")

    # --- Query type distribution ---
    if "query_type" in df.columns:
        qt_counts = df["query_type"].value_counts().to_dict()
        sections.append(f"QUERY TYPE DISTRIBUTION: {json.dumps(qt_counts)}")

    # --- Outcome distribution ---
    if "outcome" in df.columns:
        oc_counts = df["outcome"].value_counts().to_dict()
        sections.append(f"OUTCOME DISTRIBUTION: {json.dumps(oc_counts)}")

        # Outcomes by welcome vs organic
        if "is_welcome_prompt" in df.columns:
            welcome_outcomes = df[df["is_welcome_prompt"]]["outcome"].value_counts().to_dict()
            organic_outcomes = df[~df["is_welcome_prompt"]]["outcome"].value_counts().to_dict()
            sections.append(f"WELCOME PROMPT OUTCOMES: {json.dumps(welcome_outcomes)}")
            sections.append(f"ORGANIC PROMPT OUTCOMES: {json.dumps(organic_outcomes)}")

    # --- Failure modes ---
    if "failure_mode" in df.columns:
        fm = df[df["failure_mode"].notna() & (df["failure_mode"] != "")]
        if len(fm):
            fm_counts = fm["failure_mode"].value_counts().to_dict()
            sections.append(f"FAILURE MODES: {json.dumps(fm_counts)}")

    # --- Dataset demand ---
    if "dataset_required" in df.columns:
        ds = df[df["dataset_required"].notna() & (df["dataset_required"] != "")]
        if len(ds):
            ds_counts = ds["dataset_required"].value_counts().to_dict()
            sections.append(f"DATASET DEMAND: {json.dumps(ds_counts)}")

            if "is_welcome_prompt" in df.columns:
                organic_ds = ds[~ds["is_welcome_prompt"]]["dataset_required"].value_counts().to_dict()
                sections.append(f"ORGANIC DATASET DEMAND: {json.dumps(organic_ds)}")

    # --- Datasets actually used (from trace context) ---
    if "datasets_analysed" in df.columns:
        all_ds = []
        for row in df["datasets_analysed"]:
            if isinstance(row, list):
                all_ds.extend(row)
            elif isinstance(row, str) and row:
                all_ds.extend([d.strip() for d in row.split(",") if d.strip()])
        if all_ds:
            from collections import Counter
            ds_used = dict(Counter(all_ds).most_common(20))
            sections.append(f"DATASETS ACTUALLY USED: {json.dumps(ds_used)}")

    # --- Geographic scope ---
    if "geographic_scope" in df.columns:
        geo = df[df["geographic_scope"].notna() & (df["geographic_scope"] != "")]
        if len(geo):
            geo_counts = geo["geographic_scope"].value_counts().head(20).to_dict()
            sections.append(f"GEOGRAPHIC DISTRIBUTION: {json.dumps(geo_counts)}")

    # --- Top intents ---
    if "intent" in df.columns:
        intent_counts = df["intent"].value_counts().head(15).to_dict()
        sections.append(f"TOP INTENTS: {json.dumps(intent_counts)}")

    # --- Complexity ---
    if "complexity" in df.columns:
        cx_counts = df["complexity"].value_counts().to_dict()
        sections.append(f"COMPLEXITY DISTRIBUTION: {json.dumps(cx_counts)}")

    # --- Session behaviour ---
    if "session_id" in df.columns:
        sessions = df.groupby("session_id").agg(
            turns=("trace_id", "count"),
            has_welcome=("is_welcome_prompt", "any") if "is_welcome_prompt" in df.columns else ("trace_id", "count"),
        ).reset_index()
        sections.append(f"TOTAL SESSIONS: {len(sessions)}")
        sections.append(f"MEAN TURNS PER SESSION: {sessions['turns'].mean():.1f}")
        sections.append(f"MEDIAN TURNS PER SESSION: {sessions['turns'].median():.0f}")
        bounce = int((sessions["turns"] == 1).sum())
        sections.append(f"BOUNCE SESSIONS (1 turn): {bounce} ({bounce / max(1, len(sessions)):.1%})")

    # --- Dataset catalog ---
    catalog = dataset_catalog_summary()
    if catalog:
        catalog_names = [d["dataset_name"] for d in catalog]
        sections.append(f"AVAILABLE DATASETS IN CATALOG: {json.dumps(catalog_names)}")

    # --- Representative traces (3 per outcome category) ---
    sections.append("\nREPRESENTATIVE TRACES:")
    for outcome in ["success", "partial", "failure"]:
        subset = df[df["outcome"] == outcome] if "outcome" in df.columns else pd.DataFrame()
        for _, row in subset.head(3).iterrows():
            sid = row.get("session_id", "unknown")
            sections.append(
                f"[{outcome}] session={sid} | "
                f"topic={row.get('topic', '')} | type={row.get('query_type', '')} | "
                f"intent={str(row.get('intent', ''))[:80]} | "
                f"failure={row.get('failure_mode', '')} | "
                f"welcome={row.get('is_welcome_prompt', False)} | "
                f"prompt={str(row.get('prompt', ''))[:150]}"
            )

    return "\n".join(sections)


def render(base_thread_url: str) -> None:
    """Render the Report Generator tab."""
    st.markdown("### 📝 Intelligence Report")
    st.caption(
        "Generate a structured product intelligence report from enriched trace data. "
        "Uses Claude (via CLI) for analytical narrative generation."
    )

    enriched = get_enriched_traces()
    if not enriched:
        st.info("Run **⚡ Enrichment** first to generate a report.")
        return

    # --- Editable system prompt ---
    with st.expander("📝 Edit system prompt", expanded=False):
        st.caption(
            "This prompt is sent to Claude to generate the report. "
            "Use `{trace_count}`, `{date_range}`, and `{analysis_data}` as placeholders."
        )
        report_prompt_template = st.text_area(
            "Report prompt",
            value=st.session_state.get("report_prompt_template", DEFAULT_REPORT_PROMPT),
            height=400,
            key="report_prompt_template",
            label_visibility="collapsed",
        )

    # Claude availability check
    claude_ok = is_claude_available()
    if not claude_ok:
        st.warning(
            "⚠️ Claude CLI not found. Install and authenticate the `claude` CLI "
            "for report generation. The report generator requires a Claude Max subscription."
        )

    # Report state
    if "intelligence_report" not in st.session_state:
        st.session_state.intelligence_report = ""

    col1, col2 = st.columns([1, 3])
    with col1:
        generate_btn = st.button(
            "📝 Generate Report",
            type="primary",
            key="report_generate_btn",
            disabled=not claude_ok,
        )
    with col2:
        st.caption(f"Will analyse {len(enriched)} enriched traces")

    if generate_btn:
        with st.spinner("Building analysis data..."):
            analysis_data = _build_analysis_data(enriched)

        # Build date range string
        df = pd.DataFrame(enriched)
        date_range = "unknown period"
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
            if len(ts):
                date_range = f"{ts.min().date()} to {ts.max().date()}"

        prompt = report_prompt_template.format(
            trace_count=len(enriched),
            date_range=date_range,
            analysis_data=analysis_data,
        )

        with st.spinner("Generating report with Claude..."):
            report = call_claude(prompt, timeout_seconds=300)

        if report:
            st.session_state.intelligence_report = report
            st.rerun()
        else:
            st.error("Report generation failed. Check Claude CLI authentication.")

    # Display report
    report = st.session_state.intelligence_report
    if report:
        st.markdown("---")
        st.markdown(report)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download report (Markdown)",
                report.encode("utf-8"),
                "product_intelligence_report.md",
                "text/markdown",
                key="report_download_md",
            )
        with col2:
            with st.expander("📋 Copy for Slack"):
                st.code(report, language="markdown")
