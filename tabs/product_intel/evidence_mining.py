"""Evidence Mining mode — search traces using natural language hypothesis."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from utils.llm_helpers import (
    call_gemini,
    chunked,
    parse_json_any,
    parse_json_dict,
    truncate_text,
)
from utils.data_helpers import csv_bytes_any
from utils.prompt_fixtures import DEFAULT_EVIDENCE_PROMPT


def render(
    rows: list[dict[str, Any]],
    base_thread_url: str,
    gemini_api_key: str,
    gemini_model: str,
    use_batching: bool,
    batch_size: int,
    max_chars_per_trace: int,
) -> None:
    """Render the Evidence Mining tab."""
    st.markdown("### 🔍 Evidence Mining")
    st.caption("Search traces using natural language to find evidence supporting a product hypothesis.")

    hypothesis = st.text_area(
        "Describe what you're looking for",
        placeholder="e.g. Find prompts where users ask about land cover change trends.",
        height=80,
        key="evidence_hypothesis",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        search_in = st.radio(
            "Search in",
            options=["Prompt only", "Output only", "Both"],
            index=2,
            horizontal=True,
            key="evidence_search_in",
        )
    with col2:
        max_results = st.number_input("Max results", min_value=5, max_value=200, value=50, key="evidence_max")

    with st.expander("📝 Edit system prompt", expanded=False):
        st.caption("Use `{hypothesis}` and `{search_text}` as placeholders.")
        evidence_prompt_template = st.text_area(
            "System prompt",
            value=DEFAULT_EVIDENCE_PROMPT,
            height=200,
            key="evidence_prompt_template",
            label_visibility="collapsed",
        )

    if "evidence_results" not in st.session_state:
        st.session_state.evidence_results = []

    if st.button("🔎 Search with LLM", type="primary", key="evidence_search_btn"):
        if not hypothesis.strip():
            st.warning("Please describe what you're looking for.")
            return
        if not gemini_api_key:
            st.error("Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")
            return
        if not rows:
            st.warning("No traces available.")
            return

        results: list[dict[str, Any]] = []
        progress = st.progress(0.0, text="Scoring traces...")

        def _score_single(r: dict[str, Any]) -> dict[str, Any] | None:
            if search_in == "Prompt only":
                search_text = r.get("prompt", "")
            elif search_in == "Output only":
                search_text = r.get("answer", "")
            else:
                search_text = f"PROMPT: {r.get('prompt', '')}\n\nOUTPUT: {r.get('answer', '')}"
            search_text = truncate_text(str(search_text or ""), int(max_chars_per_trace))

            scoring_prompt = evidence_prompt_template.format(
                hypothesis=hypothesis,
                search_text=search_text,
            )
            resp_txt = call_gemini(gemini_api_key, gemini_model, scoring_prompt)
            parsed = parse_json_dict(resp_txt)
            if parsed.get("relevant") or (isinstance(parsed.get("score"), (int, float)) and parsed["score"] >= 50):
                return {
                    **r,
                    "relevance_score": parsed.get("score", 50),
                    "relevance_reason": parsed.get("reason", ""),
                }
            return None

        scanned = 0
        if use_batching and int(batch_size) > 1:
            batches = chunked(rows, int(batch_size))
            for b in batches:
                batch_payload = []
                for r in b:
                    if search_in == "Prompt only":
                        stxt = r.get("prompt", "")
                    elif search_in == "Output only":
                        stxt = r.get("answer", "")
                    else:
                        stxt = f"PROMPT: {r.get('prompt', '')}\n\nOUTPUT: {r.get('answer', '')}"
                    batch_payload.append(
                        {
                            "trace_id": r.get("trace_id"),
                            "search_text": truncate_text(str(stxt or ""), int(max_chars_per_trace)),
                        }
                    )

                batch_prompt = (
                    "You are a relevance scorer. For each item in TRACES, score how relevant it is to the HYPOTHESIS.\n\n"
                    "Return a JSON array with one object per input item, each object having keys: trace_id, relevant (true/false), score (0-100), reason.\n\n"
                    f"HYPOTHESIS: {hypothesis}\n\n"
                    f"TRACES: {json.dumps(batch_payload)}"
                )

                parsed_any = None
                try:
                    resp_txt = call_gemini(gemini_api_key, gemini_model, batch_prompt)
                    parsed_any = parse_json_any(resp_txt)
                except Exception:
                    parsed_any = None

                if isinstance(parsed_any, list):
                    by_tid: dict[str, dict[str, Any]] = {}
                    for obj in parsed_any:
                        if isinstance(obj, dict) and obj.get("trace_id") is not None:
                            by_tid[str(obj.get("trace_id"))] = obj
                    for r in b:
                        scanned += 1
                        obj = by_tid.get(str(r.get("trace_id") or ""))
                        if isinstance(obj, dict):
                            score = obj.get("score")
                            relevant = obj.get("relevant")
                            if relevant or (isinstance(score, (int, float)) and float(score) >= 50):
                                results.append(
                                    {
                                        **r,
                                        "relevance_score": score if score is not None else 50,
                                        "relevance_reason": obj.get("reason", ""),
                                    }
                                )
                        progress.progress(min(1.0, scanned / len(rows)), text=f"Scored {scanned}/{len(rows)}")
                        if len(results) >= int(max_results):
                            break
                else:
                    for r in b:
                        scanned += 1
                        try:
                            one = _score_single(r)
                            if one:
                                results.append(one)
                        except Exception as e:
                            st.warning(f"Error scoring trace: {e}")
                        progress.progress(min(1.0, scanned / len(rows)), text=f"Scored {scanned}/{len(rows)}")
                        if len(results) >= int(max_results):
                            break

                if len(results) >= int(max_results):
                    break
        else:
            for r in rows:
                scanned += 1
                try:
                    one = _score_single(r)
                    if one:
                        results.append(one)
                except Exception as e:
                    st.warning(f"Error scoring trace: {e}")
                progress.progress(min(1.0, scanned / len(rows)), text=f"Scored {scanned}/{len(rows)}")
                if len(results) >= int(max_results):
                    break

        progress.empty()
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        st.session_state.evidence_results = results
        st.rerun()

    results = st.session_state.evidence_results
    if results:
        st.success(f"Found **{len(results)}** relevant traces")

        st.markdown("**Results table**")
        table_rows: list[dict[str, Any]] = []
        for r in results:
            table_rows.append(
                {
                    "relevance_score": r.get("relevance_score", ""),
                    "relevance_reason": r.get("relevance_reason", ""),
                    "prompt": r.get("prompt", ""),
                    "answer": r.get("answer", ""),
                    "url": f"{base_thread_url.rstrip('/')}/{r.get('session_id')}" if r.get("session_id") else "",
                }
            )
        st.dataframe(table_rows, width="stretch")

        evidence_csv_bytes = csv_bytes_any(table_rows)
        st.download_button(
            "⬇️ Download results CSV",
            evidence_csv_bytes,
            "evidence_mining_results.csv",
            "text/csv",
            key="evidence_results_csv",
        )

        st.markdown("**Top matches (expand for details)**")
        for r in results[:20]:
            with st.expander(f"🎯 Score: {r.get('relevance_score', '?')} — {r.get('prompt', '')[:80]}..."):
                st.caption(f"**Reason:** {r.get('relevance_reason', 'N/A')}")
                st.markdown("**Prompt:**")
                st.code(r.get("prompt", ""), language=None)
                st.markdown("**Output:**")
                st.code(r.get("answer", "")[:1000] + ("..." if len(r.get("answer", "")) > 1000 else ""), language=None)
                url = f"{base_thread_url.rstrip('/')}/{r.get('session_id')}" if r.get("session_id") else ""
                if url:
                    st.link_button("🔗 View in GNW", url)
