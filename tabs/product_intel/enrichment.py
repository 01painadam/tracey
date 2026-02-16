"""Stage 1: Enrichment engine — batch-process traces through Gemini for structured metadata."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import streamlit as st

from utils.enrichment_schema import (
    BatchEnrichmentResponse,
    TraceEnrichment,
    coerce_enrichment,
    dataset_catalog_summary,
    is_welcome_prompt,
    load_starter_prompts,
)
from utils.llm_helpers import (
    call_gemini_structured,
    chunked,
    model_aware_batch_size,
    truncate_text,
)
from utils.prompt_fixtures import (
    DEFAULT_ENRICHMENT_BATCH_PROMPT,
    DEFAULT_ENRICHMENT_PROMPT,
    DEFAULT_ENRICHMENT_SYSTEM,
)
from utils.trace_parsing import (
    classify_outcome,
    extract_trace_context,
    normalize_trace_format,
    first_human_prompt,
    final_ai_message,
    active_turn_prompt,
    active_turn_answer,
    parse_trace_dt,
)


# ---------------------------------------------------------------------------
# Trace normalization (richer than the old _normalize_rows)
# ---------------------------------------------------------------------------


def normalize_trace_rich(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a trace into a flat dict with prompt, answer, and deterministic context.

    Unlike the old ``_normalize_rows()`` which only extracted first_human_prompt
    and final_ai_message, this uses active_turn extraction and adds full context
    from ``extract_trace_context()`` and ``classify_outcome()``.
    """
    n = normalize_trace_format(trace)
    prompt = active_turn_prompt(n) or first_human_prompt(n)
    answer = active_turn_answer(n) or final_ai_message(n)
    if not prompt and not answer:
        return None

    ctx = extract_trace_context(n)
    outcome_heuristic = classify_outcome(n, answer or "")
    dt = parse_trace_dt(n)

    return {
        "trace_id": n.get("id"),
        "timestamp": dt,
        "session_id": n.get("sessionId"),
        "user_id": n.get("userId") or (n.get("metadata") or {}).get("user_id"),
        "environment": n.get("environment"),
        "prompt": prompt,
        "answer": answer,
        # Deterministic context
        "outcome_heuristic": outcome_heuristic,
        "tools_used": ctx.get("tools_used", []),
        "datasets_analysed": ctx.get("datasets_analysed", []),
        "aoi_name": ctx.get("aoi_name", ""),
        "aoi_type": ctx.get("aoi_type", ""),
    }


# ---------------------------------------------------------------------------
# Enrichment state management
# ---------------------------------------------------------------------------

_STATE_KEY = "enriched_traces"
_HASH_KEY = "enriched_traces_hash"


def _trace_ids_hash(traces: list[dict[str, Any]]) -> str:
    """Compute a hash of trace IDs to detect stale enrichment data."""
    ids = sorted(str(t.get("id") or t.get("trace_id") or "") for t in traces)
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]


def get_enriched_traces() -> list[dict[str, Any]]:
    """Get enriched traces from session state."""
    return st.session_state.get(_STATE_KEY, [])


def is_enrichment_stale(raw_traces: list[dict[str, Any]]) -> bool:
    """Check if enrichment data is stale (raw traces changed since last enrichment)."""
    current_hash = _trace_ids_hash(raw_traces)
    stored_hash = st.session_state.get(_HASH_KEY, "")
    return current_hash != stored_hash


def _needs_enrichment(
    row: dict[str, Any],
    existing: dict[str, dict[str, Any]],
) -> bool:
    """Check if a trace row needs enrichment (not in existing cache)."""
    tid = str(row.get("trace_id") or "")
    return tid not in existing


# ---------------------------------------------------------------------------
# Batch enrichment logic
# ---------------------------------------------------------------------------


def _build_batch_payload(
    rows: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    """Build the JSON payload for a batch enrichment call."""
    payload = []
    for r in rows:
        payload.append({
            "trace_id": r.get("trace_id", ""),
            "user_prompt": truncate_text(str(r.get("prompt") or ""), max_chars),
            "assistant_response": truncate_text(str(r.get("answer") or ""), max_chars),
            "outcome_heuristic": r.get("outcome_heuristic", ""),
            "tools_used": ", ".join(r.get("tools_used") or []),
            "datasets_analysed": ", ".join(r.get("datasets_analysed") or []),
            "aoi_name": r.get("aoi_name", ""),
            "aoi_type": r.get("aoi_type", ""),
        })
    return payload


def _enrich_batch(
    api_key: str,
    model_name: str,
    rows: list[dict[str, Any]],
    max_chars: int,
    system_instruction: str | None = None,
    batch_prompt_template: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Enrich a batch of traces via Gemini structured output.

    Returns a dict mapping trace_id -> coerced enrichment fields.
    """
    payload = _build_batch_payload(rows, max_chars)
    traces_json = json.dumps(payload, ensure_ascii=False)

    template = batch_prompt_template or DEFAULT_ENRICHMENT_BATCH_PROMPT
    prompt = template.format(traces_json=traces_json)

    sys_instr = system_instruction or DEFAULT_ENRICHMENT_SYSTEM
    result = call_gemini_structured(
        api_key,
        model_name,
        prompt,
        response_schema=BatchEnrichmentResponse,
        system_instruction=sys_instr,
        temperature=0.1,
    )

    enrichments: dict[str, dict[str, Any]] = {}

    if isinstance(result, dict) and "items" in result:
        items = result["items"]
    elif isinstance(result, list):
        items = result
    else:
        return enrichments

    for item in items:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("trace_id") or "")
        if not tid:
            continue
        enrichments[tid] = coerce_enrichment(item)

    return enrichments


def _enrich_single(
    api_key: str,
    model_name: str,
    row: dict[str, Any],
    max_chars: int,
    system_instruction: str | None = None,
    single_prompt_template: str | None = None,
) -> dict[str, Any]:
    """Enrich a single trace via Gemini structured output."""
    template = single_prompt_template or DEFAULT_ENRICHMENT_PROMPT
    prompt = template.format(
        user_prompt=truncate_text(str(row.get("prompt") or ""), max_chars),
        assistant_response=truncate_text(str(row.get("answer") or ""), max_chars),
        outcome_heuristic=row.get("outcome_heuristic", ""),
        tools_used=", ".join(row.get("tools_used") or []),
        datasets_analysed=", ".join(row.get("datasets_analysed") or []),
        aoi_name=row.get("aoi_name", ""),
        aoi_type=row.get("aoi_type", ""),
    )

    sys_instr = system_instruction or DEFAULT_ENRICHMENT_SYSTEM
    result = call_gemini_structured(
        api_key,
        model_name,
        prompt,
        response_schema=TraceEnrichment,
        system_instruction=sys_instr,
        temperature=0.1,
    )

    if isinstance(result, dict):
        return coerce_enrichment(result)
    return coerce_enrichment({})


# ---------------------------------------------------------------------------
# Main enrichment runner
# ---------------------------------------------------------------------------


def run_enrichment(
    raw_traces: list[dict[str, Any]],
    api_key: str,
    model_name: str,
    max_chars_per_trace: int = 4000,
    user_batch_size: int | None = None,
    progress_container: Any = None,
    system_instruction: str | None = None,
    batch_prompt_template: str | None = None,
    single_prompt_template: str | None = None,
) -> list[dict[str, Any]]:
    """Run enrichment on raw traces, returning enriched trace dicts.

    Delta-aware: only processes traces not already in session state cache.
    Adds deterministic fields (is_welcome_prompt) after Gemini enrichment.

    Args:
        raw_traces: Raw traces from Langfuse.
        api_key: Gemini API key.
        model_name: Gemini model name.
        max_chars_per_trace: Max chars per prompt/answer field.
        user_batch_size: User-specified batch size (capped by model limit).
        progress_container: Streamlit container for progress bar.
        system_instruction: Custom system instruction (uses default if None).
        batch_prompt_template: Custom batch prompt template (uses default if None).
        single_prompt_template: Custom single prompt template (uses default if None).

    Returns:
        List of enriched trace dicts (all traces, not just newly enriched).
    """
    # Normalize all traces
    normalized: list[dict[str, Any]] = []
    for t in raw_traces:
        row = normalize_trace_rich(t)
        if row is not None:
            normalized.append(row)

    if not normalized:
        return []

    # Load existing enrichments from cache
    existing_enriched: dict[str, dict[str, Any]] = {}
    for et in st.session_state.get(_STATE_KEY, []):
        tid = str(et.get("trace_id") or "")
        if tid:
            existing_enriched[tid] = et

    # Find traces needing enrichment
    to_enrich = [r for r in normalized if _needs_enrichment(r, existing_enriched)]

    # Compute batch size
    batch_sz = model_aware_batch_size(model_name, user_batch_size)

    # Load starter prompts for welcome-prompt detection
    starter_set = load_starter_prompts()

    # Process new traces
    if to_enrich:
        batches = chunked(to_enrich, batch_sz) if batch_sz > 1 else [[r] for r in to_enrich]
        done = 0
        total = len(to_enrich)

        if progress_container is not None:
            progress_bar = progress_container.progress(0.0, text=f"Enriching 0/{total} traces...")
        else:
            progress_bar = None

        for batch in batches:
            if len(batch) > 1:
                # Batch enrichment
                batch_results = _enrich_batch(
                    api_key, model_name, batch, max_chars_per_trace,
                    system_instruction=system_instruction,
                    batch_prompt_template=batch_prompt_template,
                )

                # Fallback to single for any missed traces
                for r in batch:
                    tid = str(r.get("trace_id") or "")
                    if tid in batch_results:
                        enriched_row = {**r, **batch_results[tid]}
                    else:
                        # Batch missed this trace — try single
                        try:
                            single_result = _enrich_single(
                                api_key, model_name, r, max_chars_per_trace,
                                system_instruction=system_instruction,
                                single_prompt_template=single_prompt_template,
                            )
                            enriched_row = {**r, **single_result}
                        except Exception:
                            enriched_row = {**r, **coerce_enrichment({})}

                    # Add deterministic fields
                    enriched_row["is_welcome_prompt"] = is_welcome_prompt(
                        str(enriched_row.get("prompt") or ""), starter_set
                    )
                    existing_enriched[tid] = enriched_row
                    done += 1

            else:
                # Single trace
                r = batch[0]
                tid = str(r.get("trace_id") or "")
                try:
                    single_result = _enrich_single(
                        api_key, model_name, r, max_chars_per_trace,
                        system_instruction=system_instruction,
                        single_prompt_template=single_prompt_template,
                    )
                    enriched_row = {**r, **single_result}
                except Exception:
                    enriched_row = {**r, **coerce_enrichment({})}

                enriched_row["is_welcome_prompt"] = is_welcome_prompt(
                    str(enriched_row.get("prompt") or ""), starter_set
                )
                existing_enriched[tid] = enriched_row
                done += 1

            if progress_bar is not None:
                progress_bar.progress(
                    min(1.0, done / total),
                    text=f"Enriching {done}/{total} traces...",
                )

        if progress_bar is not None:
            progress_bar.empty()

    # Also ensure existing normalized traces that were already enriched get welcome_prompt flag
    for r in normalized:
        tid = str(r.get("trace_id") or "")
        if tid in existing_enriched and "is_welcome_prompt" not in existing_enriched[tid]:
            existing_enriched[tid]["is_welcome_prompt"] = is_welcome_prompt(
                str(existing_enriched[tid].get("prompt") or ""), starter_set
            )

    # Build final list preserving order of normalized traces
    all_enriched = []
    for r in normalized:
        tid = str(r.get("trace_id") or "")
        if tid in existing_enriched:
            all_enriched.append(existing_enriched[tid])
        else:
            # Shouldn't happen, but include raw row as fallback
            r["is_welcome_prompt"] = is_welcome_prompt(str(r.get("prompt") or ""), starter_set)
            all_enriched.append({**r, **coerce_enrichment({})})

    # Store in session state
    st.session_state[_STATE_KEY] = all_enriched
    st.session_state[_HASH_KEY] = _trace_ids_hash(raw_traces)

    return all_enriched


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------


def render(
    traces: list[dict[str, Any]],
    gemini_api_key: str,
    gemini_model: str,
    max_chars_per_trace: int,
) -> None:
    """Render the Enrichment tab UI."""
    st.markdown("### ⚡ Trace Enrichment")
    st.caption(
        "Batch-process traces through Gemini to extract structured metadata: "
        "topic, dataset required, query type, intent, complexity, outcome, and failure mode. "
        "Enriched data feeds all analysis modes and the report generator."
    )

    # --- Editable system prompts ---
    with st.expander("📝 Edit system prompts", expanded=False):
        st.caption(
            "These prompts are sent to Gemini during enrichment. "
            "The **system instruction** sets the analyst persona. "
            "The **batch prompt** is used when processing multiple traces at once. "
            "The **single prompt** is used as a fallback for individual traces."
        )
        enrichment_system = st.text_area(
            "System instruction",
            value=st.session_state.get("enrichment_system_prompt", DEFAULT_ENRICHMENT_SYSTEM),
            height=120,
            key="enrichment_system_prompt",
        )
        enrichment_batch_prompt = st.text_area(
            "Batch enrichment prompt",
            value=st.session_state.get("enrichment_batch_prompt", DEFAULT_ENRICHMENT_BATCH_PROMPT),
            height=300,
            key="enrichment_batch_prompt",
            help="Use `{traces_json}` as placeholder for the batch payload.",
        )
        enrichment_single_prompt = st.text_area(
            "Single trace enrichment prompt",
            value=st.session_state.get("enrichment_single_prompt", DEFAULT_ENRICHMENT_PROMPT),
            height=300,
            key="enrichment_single_prompt",
            help="Placeholders: `{user_prompt}`, `{assistant_response}`, `{outcome_heuristic}`, "
                 "`{tools_used}`, `{datasets_analysed}`, `{aoi_name}`, `{aoi_type}`.",
        )

    enriched = get_enriched_traces()
    stale = is_enrichment_stale(traces) if enriched else True

    # Status display
    if enriched and not stale:
        n_enriched = len(enriched)
        n_welcome = sum(1 for e in enriched if e.get("is_welcome_prompt"))
        st.success(
            f"✅ **{n_enriched}** traces enriched "
            f"({n_welcome} welcome prompts detected). "
            f"Data is current."
        )
    elif enriched and stale:
        st.warning(
            f"⚠️ Enrichment data exists ({len(enriched)} traces) but may be stale — "
            f"traces have changed since last enrichment. Re-run to update."
        )
    else:
        st.info("No enrichment data yet. Click below to start.")

    # Run button
    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button(
            "⚡ Run Enrichment" if not enriched else "🔄 Re-enrich",
            type="primary",
            key="enrichment_run_btn",
        )
    with col2:
        if enriched and not stale:
            # Show how many would need processing on re-run
            existing_ids = {str(e.get("trace_id") or "") for e in enriched}
            new_count = sum(
                1 for t in traces
                if str((normalize_trace_format(t) or {}).get("id") or "") not in existing_ids
            )
            if new_count > 0:
                st.caption(f"Delta: {new_count} new traces to process")
            else:
                st.caption("All traces already enriched")

    if run_btn:
        if not gemini_api_key:
            st.error("Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")
            return

        progress_container = st.container()
        with st.spinner("Running enrichment..."):
            result = run_enrichment(
                raw_traces=traces,
                api_key=gemini_api_key,
                model_name=gemini_model,
                max_chars_per_trace=max_chars_per_trace,
                progress_container=progress_container,
                system_instruction=enrichment_system,
                batch_prompt_template=enrichment_batch_prompt,
                single_prompt_template=enrichment_single_prompt,
            )

        if result:
            st.success(f"Enriched **{len(result)}** traces")
            st.rerun()
        else:
            st.warning("No traces could be enriched.")
            return

    # Display enrichment results
    if enriched:
        _render_enrichment_summary(enriched)


def _render_enrichment_summary(enriched: list[dict[str, Any]]) -> None:
    """Render a summary of enrichment results."""
    import pandas as pd

    df = pd.DataFrame(enriched)

    # Key metrics
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total traces", len(enriched))
    with cols[1]:
        n_welcome = int(df["is_welcome_prompt"].sum()) if "is_welcome_prompt" in df.columns else 0
        st.metric("Welcome prompts", n_welcome)
    with cols[2]:
        n_success = int((df["outcome"] == "success").sum()) if "outcome" in df.columns else 0
        rate = f"{n_success / max(1, len(enriched)):.0%}"
        st.metric("Success rate (LLM)", rate)
    with cols[3]:
        n_failure = int((df["outcome"].isin(["failure", "partial"])).sum()) if "outcome" in df.columns else 0
        st.metric("Failures/partial", n_failure)

    # Distribution tabs
    dist_tabs = st.tabs(["Topics", "Query Types", "Complexity", "Outcomes", "Failure Modes"])

    with dist_tabs[0]:
        if "topic" in df.columns:
            counts = df["topic"].value_counts().reset_index()
            counts.columns = ["topic", "count"]
            st.dataframe(counts, hide_index=True, width="stretch")

    with dist_tabs[1]:
        if "query_type" in df.columns:
            counts = df["query_type"].value_counts().reset_index()
            counts.columns = ["query_type", "count"]
            st.dataframe(counts, hide_index=True, width="stretch")

    with dist_tabs[2]:
        if "complexity" in df.columns:
            counts = df["complexity"].value_counts().reset_index()
            counts.columns = ["complexity", "count"]
            st.dataframe(counts, hide_index=True, width="stretch")

    with dist_tabs[3]:
        if "outcome" in df.columns:
            counts = df["outcome"].value_counts().reset_index()
            counts.columns = ["outcome", "count"]
            st.dataframe(counts, hide_index=True, width="stretch")

    with dist_tabs[4]:
        if "failure_mode" in df.columns:
            fm = df[df["failure_mode"].notna() & (df["failure_mode"] != "")]
            if len(fm):
                counts = fm["failure_mode"].value_counts().reset_index()
                counts.columns = ["failure_mode", "count"]
                st.dataframe(counts, hide_index=True, width="stretch")
            else:
                st.info("No failure modes detected.")

    # Raw data expander
    with st.expander("📋 Raw enriched data", expanded=False):
        display_cols = [
            "trace_id", "prompt", "topic", "query_type", "dataset_required",
            "geographic_scope", "intent", "complexity", "outcome", "failure_mode",
            "is_welcome_prompt",
        ]
        existing_cols = [c for c in display_cols if c in df.columns]
        display_df = df[existing_cols].copy()
        if "prompt" in display_df.columns:
            display_df["prompt"] = display_df["prompt"].astype(str).str[:100] + "..."
        st.dataframe(display_df, hide_index=True, width="stretch")
