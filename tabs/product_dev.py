"""Product Intelligence — pipeline orchestrator.

Slim router that connects the enrichment engine, analysis modes,
evidence mining, and report generator.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from utils.llm_helpers import get_gemini_model_options
from utils.trace_parsing import normalize_trace_format, first_human_prompt, final_ai_message


# ---------------------------------------------------------------------------
# Helpers (kept for legacy modes)
# ---------------------------------------------------------------------------


def _normalize_rows(traces: list[dict[str, Any]], limit: int = 500) -> list[dict[str, Any]]:
    """Normalize traces to rows with prompt/answer (legacy format)."""
    out: list[dict[str, Any]] = []
    for t in traces:
        n = normalize_trace_format(t)
        prompt = first_human_prompt(n)
        answer = final_ai_message(n)
        if not prompt and not answer:
            continue
        out.append(
            {
                "trace_id": n.get("id"),
                "timestamp": n.get("timestamp"),
                "session_id": n.get("sessionId"),
                "environment": n.get("environment"),
                "prompt": prompt,
                "answer": answer,
            }
        )
        if len(out) >= limit:
            break
    return out


def _render_llm_settings(
    gemini_api_key: str,
    gemini_model_options: list[str],
) -> tuple[str, bool, int, int]:
    """Render LLM settings expander and return (model, use_batching, batch_size, max_chars)."""
    default_model = "gemini-2.5-flash-lite"
    if default_model not in gemini_model_options and gemini_model_options:
        default_model = gemini_model_options[0]

    with st.expander("⚙️ LLM Settings", expanded=False):
        gemini_model = st.selectbox(
            "Gemini model",
            options=gemini_model_options,
            index=gemini_model_options.index(default_model) if default_model in gemini_model_options else 0,
            key="product_dev_gemini_model",
        )
        if not gemini_api_key:
            st.warning("No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")
        else:
            st.success("Gemini API key configured ✓")

        use_batching = st.checkbox(
            "Batch traces per Gemini request",
            value=bool(st.session_state.get("product_dev_use_batching", True)),
            key="product_dev_use_batching",
        )
        st.number_input(
            "Batch size (auto-capped by model)",
            min_value=1,
            max_value=50,
            value=int(st.session_state.get("product_dev_batch_size", 25)),
            key="product_dev_batch_size",
            disabled=not use_batching,
        )
        st.number_input(
            "Max chars per trace (prompt/output)",
            min_value=200,
            max_value=20000,
            value=int(st.session_state.get("product_dev_max_chars_per_trace", 4000)),
            key="product_dev_max_chars_per_trace",
            disabled=not use_batching,
        )

    return (
        gemini_model,
        bool(st.session_state.get("product_dev_use_batching", True)),
        int(st.session_state.get("product_dev_batch_size", 25)),
        int(st.session_state.get("product_dev_max_chars_per_trace", 4000)),
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render(
    base_thread_url: str,
    gemini_api_key: str,
) -> None:
    """Render the Product Intelligence tab."""
    st.subheader("🧠 Product Intelligence")
    st.caption(
        "Enrich traces with structured metadata, explore patterns across topics, "
        "failures, and user intents, then generate evidence-backed reports."
    )

    traces: list[dict[str, Any]] = st.session_state.get("stats_traces", [])

    if not traces:
        st.info(
            "This tab helps you **mine traces for product insights**:\n\n"
            "- ⚡ **Enrichment** – Extract structured metadata from all traces\n"
            "- 📊 **Topic & Dataset Patterns** – What users ask about and what data they need\n"
            "- 🎯 **JTBD / Use Cases** – Cluster user intents into jobs-to-be-done\n"
            "- 📈 **Trends** – How topics and success rates shift over time\n"
            "- ❌ **Failure Patterns** – Categorise and cluster failure modes\n"
            "- 💡 **Feature Requests** – Identify missing capabilities\n"
            "- 🔍 **Evidence Mining** – Search traces for a hypothesis\n"
            "- 📝 **Report** – Generate stakeholder-ready intelligence report\n\n"
            "Use the sidebar **🚀 Fetch traces** button first."
        )
        return

    gemini_model_options = get_gemini_model_options(gemini_api_key) if gemini_api_key else [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    gemini_model, use_batching, batch_size, max_chars_per_trace = _render_llm_settings(
        gemini_api_key,
        gemini_model_options,
    )

    # --- Pipeline tabs ---
    show_report = os.environ.get("ENABLE_CLAUDE_REPORT", "").lower() in ("1", "true", "yes")

    tab_labels = [
        "⚡ Enrichment",
        "📊 Topics & Datasets",
        "🎯 JTBD",
        "📈 Trends",
        "❌ Failures",
        "💡 Feature Gaps",
        "🔍 Evidence Mining",
    ]
    if show_report:
        tab_labels.append("📝 Report")

    tabs = st.tabs(tab_labels)

    # Tab 0: Enrichment
    with tabs[0]:
        from tabs.product_intel.enrichment import render as render_enrichment
        render_enrichment(
            traces=traces,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            max_chars_per_trace=max_chars_per_trace,
        )

    # Tab 1: Topic & Dataset Patterns
    with tabs[1]:
        from tabs.product_intel.topic_patterns import render as render_topics
        render_topics(base_thread_url=base_thread_url)

    # Tab 2: JTBD
    with tabs[2]:
        from tabs.product_intel.jtbd import render as render_jtbd
        render_jtbd(base_thread_url=base_thread_url)

    # Tab 3: Trends
    with tabs[3]:
        from tabs.product_intel.trends import render as render_trends
        render_trends(base_thread_url=base_thread_url)

    # Tab 4: Failures
    with tabs[4]:
        from tabs.product_intel.failure_patterns import render as render_failures
        render_failures(base_thread_url=base_thread_url)

    # Tab 5: Feature Gaps
    with tabs[5]:
        from tabs.product_intel.feature_requests import render as render_features
        render_features(base_thread_url=base_thread_url)

    # Tab 6: Evidence Mining
    with tabs[6]:
        from tabs.product_intel.evidence_mining import render as render_evidence
        rows = _normalize_rows(traces, limit=500)
        render_evidence(
            rows=rows,
            base_thread_url=base_thread_url,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            use_batching=use_batching,
            batch_size=batch_size,
            max_chars_per_trace=max_chars_per_trace,
        )

    # Tab 7: Report (only when ENABLE_CLAUDE_REPORT is set)
    if show_report:
        with tabs[7]:
            from tabs.product_intel.report import render as render_report
            render_report(base_thread_url=base_thread_url)
