"""Prompt Intelligence tab – keyword trends, intent/topic analysis, and Gemini classification."""

import re
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from utils import (
    normalize_trace_format,
    parse_trace_dt,
    first_human_prompt,
    final_ai_message,
    classify_outcome,
    as_float,
    csv_bytes_any,
    get_gemini_model_options,
    chunked,
    truncate_text,
    parse_json_any,
    call_gemini,
    analyse_prompts_bulk,
    top_bigrams,
    extract_trace_context,
    extract_usage_metadata,
    trace_has_internal_error,
)


# ---------------------------------------------------------------------------
# Outcome colour scale (reused across charts)
# ---------------------------------------------------------------------------
_OUTCOME_DOMAIN = ["ANSWER", "DEFER", "SOFT_ERROR", "ERROR"]
_OUTCOME_RANGE = ["#30a46c", "#f5a623", "#e5484d", "#cd2b31"]


def _outcome_color() -> alt.Color:
    return alt.Color(
        "outcome:N",
        title="Outcome",
        scale=alt.Scale(domain=_OUTCOME_DOMAIN, range=_OUTCOME_RANGE),
    )


def _top_n_with_other(
    counts: Counter[str],
    top_n: int,
    total_den: int,
    other_label: str = "Other",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a top-N dataframe and an 'other details' dataframe.

    Returns:
        display_df: includes top N rows plus an aggregated 'Other' row if needed.
        other_df: rows that were rolled up into Other (label, count, percent).
    """
    total_den = max(1, int(total_den))
    items = list(counts.most_common())
    top_items = items[: int(top_n)]
    other_items = items[int(top_n) :]

    display_rows: list[dict[str, Any]] = []
    for label, c in top_items:
        display_rows.append(
            {
                "label": label,
                "count": int(c),
                "percent": round(float(c) / float(total_den) * 100.0, 1),
            }
        )

    other_rows: list[dict[str, Any]] = []
    if other_items:
        other_total = int(sum(c for _, c in other_items))
        display_rows.append(
            {
                "label": other_label,
                "count": other_total,
                "percent": round(float(other_total) / float(total_den) * 100.0, 1),
            }
        )

        for label, c in other_items:
            other_rows.append(
                {
                    "label": label,
                    "count": int(c),
                    "percent": round(float(c) / float(total_den) * 100.0, 1),
                }
            )

    display_df = pd.DataFrame(display_rows)
    other_df = pd.DataFrame(other_rows)
    return display_df, other_df


# ---------------------------------------------------------------------------
# Shared tokeniser (same as analytics.py keyword trends)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "please", "show", "tell", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "help", "thanks", "give", "generate", "analyse", "user",
}


def _tokenize(txt: Any) -> list[str]:
    s = str(txt or "").lower()
    parts = re.findall(r"[a-z0-9_\-]{2,}", s)
    out: list[str] = []
    for p in parts:
        if len(p) < 3:
            continue
        if p.isdigit():
            continue
        if p in _STOPWORDS:
            continue
        if p.startswith("http") or p.startswith("www"):
            continue
        if p.startswith("0x"):
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Gemini classification prompt
# ---------------------------------------------------------------------------
_GEMINI_CLASSIFY_PROMPT = """\
You are a prompt analyst. For each user prompt below, classify it on the following dimensions.
Return a JSON array with one object per prompt. Each object MUST have these keys:
- index: the 0-based index of the prompt in the list
- intents: list of strings – what the user is trying to do (e.g. "Show / Visualise", "Compare", "Quantify", "Trend / Change", "Causes / Drivers", "Locate / Where", or any other intent you detect)
- primary_intent: string – the single most dominant intent
- topics: list of strings – environmental/domain topics mentioned (e.g. "Forest loss", "Grassland", "Cropland", "Restoration", "Wildfire", "Land cover change", "Disturbance", "Urbanisation", "Water / Wetland", "Biodiversity", "Carbon / Climate", "Mining", "Natural land", or any other topic you detect)
- geo_entities: list of strings – geographic entities (countries, regions, states, cities, basins, etc.)
- temporal_refs: list of strings – temporal references (e.g. "Last year", "Since 2015", "Between 2015 and 2024", "Last decade", etc.)
- anti_patterns: list of strings – any issues with the prompt (e.g. "Too short", "No geographic context", "No temporal context", "Vague / ambiguous", "No specific topic", "Excessively long", "Contains URL", or any other issue you detect)

Be creative with classification – you are NOT limited to the example labels above.
Discover novel categories that emerge from the data, however keep categories broad enough to be useful, and avoid being too specific.

PROMPTS:
{prompts_json}
"""


_GEMINI_CONSOLIDATE_PROMPT = """\
You are a label consolidation engine. I have classified user prompts across multiple batches, \
and each batch may have produced semantically similar but differently worded labels.

For each dimension below, I will give you the list of unique labels and their counts. \
Merge semantically equivalent labels into a single canonical label. \
Pick the clearest, most general label as the canonical one.

Return a JSON object with one key per dimension. Each value is an object mapping \
every original label to its canonical label. Labels that are already unique should map to themselves.

Example output format:
{{
  "intents": {{"Show data": "Show / Visualise", "Visualise": "Show / Visualise", "Compare regions": "Compare"}},
  "topics": {{"Deforestation": "Forest loss", "Forest loss": "Forest loss", "Tree cover loss": "Forest loss"}},
  "geo_entities": {{"DRC": "Democratic Republic of Congo", "Congo": "Democratic Republic of Congo"}},
  "temporal_refs": {{"Past year": "Last year", "Last 12 months": "Last year"}},
  "anti_patterns": {{"Too vague": "Vague / ambiguous", "Ambiguous prompt": "Vague / ambiguous"}}
}}

Rules:
- Only merge labels that are semantically similar or synonyms (see examples above)
- Do NOT merge labels that are merely related (e.g. "Forest loss" and "Wildfire" are separate topics).
- For geo_entities, merge different spellings/abbreviations of the same place (e.g. "DRC" → "Democratic Republic of Congo").
- For temporal_refs, merge different wordings of the same timeframe (e.g. "Past year", "Last year", "Last 12 months").
- Every original label MUST appear as a key in the mapping.

DIMENSIONS:
{dimensions_json}
"""


def _consolidate_labels(
    all_results: list[dict[str, Any]],
    gemini_api_key: str,
    gemini_model: str,
) -> list[dict[str, Any]]:
    """Consolidate semantically similar labels across all classification results."""
    dimensions = ["intents", "topics", "geo_entities", "temporal_refs", "anti_patterns"]

    # Collect unique labels + counts per dimension
    dim_counts: dict[str, dict[str, int]] = {d: {} for d in dimensions}
    for result in all_results:
        for dim in dimensions:
            for label in result.get(dim, []):
                if label:
                    dim_counts[dim][label] = dim_counts[dim].get(label, 0) + 1

    # Skip consolidation if there aren't enough unique labels to warrant it
    total_unique = sum(len(v) for v in dim_counts.values())
    if total_unique < 5:
        return all_results

    # Build the payload
    dimensions_payload: dict[str, list[dict[str, Any]]] = {}
    for dim in dimensions:
        if dim_counts[dim]:
            sorted_labels = sorted(dim_counts[dim].items(), key=lambda x: -x[1])
            dimensions_payload[dim] = [{"label": l, "count": c} for l, c in sorted_labels]

    if not dimensions_payload:
        return all_results

    prompt = _GEMINI_CONSOLIDATE_PROMPT.format(
        dimensions_json=json.dumps(dimensions_payload, ensure_ascii=False, indent=2),
    )

    try:
        resp_txt = call_gemini(gemini_api_key, gemini_model, prompt)
        mapping = parse_json_any(resp_txt)
        if not isinstance(mapping, dict):
            return all_results
    except Exception:
        return all_results

    # Apply the mapping
    for result in all_results:
        for dim in dimensions:
            dim_map = mapping.get(dim, {})
            if not isinstance(dim_map, dict) or not dim_map:
                continue
            original = result.get(dim, [])
            result[dim] = [dim_map.get(label, label) for label in original]

        # Rebuild primary_intent from remapped intents
        remapped_intents = result.get("intents", [])
        if remapped_intents:
            intent_map = mapping.get("intents", {})
            old_primary = result.get("primary_intent", "Unclassified")
            result["primary_intent"] = intent_map.get(old_primary, old_primary) if isinstance(intent_map, dict) else old_primary

        # Rebuild has_anti_pattern
        result["has_anti_pattern"] = bool(result.get("anti_patterns"))

    return all_results


def render(
    gemini_api_key: str,
    start_date,
    end_date,
) -> None:
    """Render the Prompt Intelligence tab."""

    st.subheader("🔬 Prompt Intelligence")
    st.caption(
        "Analyse what users prompt for over time — identify trends, anti-patterns, gaps, and opportunities. "
        "Use regex-based classification for instant results or Gemini for deeper, open-ended discovery."
    )

    traces: list[dict[str, Any]] = st.session_state.get("stats_traces", [])
    if not traces:
        st.info(
            "This tab analyses **user prompts** to surface product insights.\n\n"
            "1. Use the sidebar **🚀 Fetch traces** button to load a dataset.\n"
            "2. Come back here to explore keyword trends, intent classification, topic extraction, and more."
        )
        return

    # Build base dataframe
    normed = [normalize_trace_format(t) for t in traces]
    rows: list[dict[str, Any]] = []
    for n in normed:
        prompt = first_human_prompt(n)
        answer = final_ai_message(n)
        dt = parse_trace_dt(n)
        outcome = classify_outcome(n, answer or "")
        rows.append({
            "trace_id": n.get("id"),
            "timestamp": dt,
            "date": dt.date() if dt else None,
            "session_id": n.get("sessionId"),
            "user_id": n.get("userId") or (n.get("metadata") or {}).get("user_id") or (n.get("metadata") or {}).get("userId"),
            "prompt": prompt,
            "answer": answer,
            "outcome": outcome,
        })

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", na_position="last")

    if "prompt" not in df.columns or not len(df):
        st.warning("No prompts found in loaded traces.")
        return

    df["prompt_len_words"] = (
        df["prompt"].fillna("").astype("string")
        .map(lambda x: len([w for w in str(x).strip().split() if w]))
    )

    # =====================================================================
    # Classification mode selector
    # =====================================================================
    st.markdown("---")
    classify_mode = st.radio(
        "Classification method",
        options=["Regex (instant)", "Gemini (LLM)"],
        horizontal=True,
        help="Regex uses predefined keyword patterns — fast and free. Gemini uses an LLM for open-ended discovery — finds novel categories.",
        key="pi_classify_mode",
    )

    # =====================================================================
    # Gemini settings & classification
    # =====================================================================
    prompt_analysis: list[dict[str, Any]] | None = None

    if classify_mode == "Gemini (LLM)":
        prompt_analysis = _render_gemini_classification(df, gemini_api_key)
    else:
        # Regex mode — instant
        prompt_list = df["prompt"].fillna("").astype(str).tolist()
        prompt_analysis = analyse_prompts_bulk(prompt_list)

    if prompt_analysis is None:
        st.info("Run Gemini classification above to see results.")
        # Still show keyword trends (they don't depend on classification)
        _render_keyword_trends(df)
        _render_bigram_trends(df)
        return

    # Attach analysis to df
    df["primary_intent"] = [pa.get("primary_intent", "Unclassified") for pa in prompt_analysis]
    df["all_intents"] = [", ".join(pa.get("intents", [])) for pa in prompt_analysis]
    df["topics"] = [", ".join(pa.get("topics", [])) if pa.get("topics") else "None detected" for pa in prompt_analysis]
    df["geo_entities"] = [", ".join(pa.get("geo_entities", [])) if pa.get("geo_entities") else "" for pa in prompt_analysis]
    df["temporal_refs"] = [", ".join(pa.get("temporal_refs", [])) if pa.get("temporal_refs") else "" for pa in prompt_analysis]
    df["anti_patterns"] = [", ".join(pa.get("anti_patterns", [])) if pa.get("anti_patterns") else "" for pa in prompt_analysis]
    df["has_anti_pattern"] = [bool(pa.get("anti_patterns")) for pa in prompt_analysis]

    # =====================================================================
    # Render all sections
    # =====================================================================
    _render_keyword_trends(df)
    _render_bigram_trends(df)

    st.markdown("---")
    st.markdown(
        "### Prompt Classification Insights",
        help="Automated classification of user prompts by intent, topic, geography, and temporal framing.",
    )
    source_label = "Gemini" if classify_mode == "Gemini (LLM)" else "Regex"
    st.caption(f"Classification source: **{source_label}**")

    _render_intent_section(df, prompt_analysis, classify_mode)
    _render_topic_section(df, prompt_analysis)
    _render_geo_temporal_section(df, prompt_analysis)
    _render_rising_declining(df, prompt_analysis, classify_mode)
    _render_prompt_outcome_correlation(df, prompt_analysis)
    _render_anti_patterns(df, prompt_analysis)
    _render_export(df)


# =========================================================================
# Gemini classification
# =========================================================================
def _render_gemini_classification(
    df: pd.DataFrame,
    gemini_api_key: str,
) -> list[dict[str, Any]] | None:
    """Render Gemini classification UI and return results or None."""

    gemini_model_options = get_gemini_model_options(gemini_api_key) if gemini_api_key else [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    default_model = "gemini-2.5-flash-lite"
    if default_model not in gemini_model_options and gemini_model_options:
        default_model = gemini_model_options[0]

    with st.expander("⚙️ Gemini Classification Settings", expanded=True):
        gc1, gc2, gc3 = st.columns([2, 1, 1])
        with gc1:
            gemini_model = st.selectbox(
                "Gemini model",
                options=gemini_model_options,
                index=gemini_model_options.index(default_model) if default_model in gemini_model_options else 0,
                key="pi_gemini_model",
            )
        with gc2:
            batch_size = st.number_input(
                "Batch size",
                min_value=1,
                max_value=100,
                value=int(st.session_state.get("pi_batch_size", 10)),
                key="pi_batch_size",
                help="Number of prompts to send per Gemini request.",
            )
        with gc3:
            max_prompts = st.number_input(
                "Max prompts",
                min_value=10,
                max_value=5000,
                value=min(len(df), int(st.session_state.get("pi_max_prompts", 500))),
                key="pi_max_prompts",
                help="Maximum number of prompts to classify.",
            )

        if not gemini_api_key:
            st.warning("No Gemini API key found. Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.env`")
        else:
            st.success("Gemini API key configured ✓")

        with st.expander("📝 Edit classification prompt", expanded=False):
            custom_prompt = st.text_area(
                "System prompt",
                value=_GEMINI_CLASSIFY_PROMPT,
                height=300,
                key="pi_gemini_prompt",
                label_visibility="collapsed",
            )

    # Session state for cached results
    if "pi_gemini_results" not in st.session_state:
        st.session_state.pi_gemini_results = None

    btn_c1, btn_c2 = st.columns([1, 1])
    with btn_c1:
        run_btn = st.button("🚀 Classify with Gemini", type="primary", key="pi_gemini_run")
    with btn_c2:
        if st.button("🧹 Clear results", key="pi_gemini_clear", disabled=st.session_state.pi_gemini_results is None):
            st.session_state.pi_gemini_results = None
            st.rerun()

    if run_btn:
        if not gemini_api_key:
            st.error("Gemini API key required.")
            return None

        prompt_list = df["prompt"].fillna("").astype(str).tolist()[:int(max_prompts)]
        if not prompt_list:
            st.warning("No prompts to classify.")
            return None

        all_results: list[dict[str, Any]] = [{}] * len(prompt_list)
        batches = chunked(
            [{"index": i, "prompt": p} for i, p in enumerate(prompt_list)],
            int(batch_size),
        )

        progress = st.progress(0.0, text="Classifying prompts with Gemini...")
        done = 0
        errors = 0

        for batch in batches:
            batch_prompts = [item["prompt"] for item in batch]
            batch_indices = [item["index"] for item in batch]

            prompt_template = custom_prompt if custom_prompt.strip() else _GEMINI_CLASSIFY_PROMPT
            classification_prompt = prompt_template.format(
                prompts_json=json.dumps(batch_prompts, ensure_ascii=False),
            )

            try:
                resp_txt = call_gemini(gemini_api_key, gemini_model, classification_prompt)
                parsed = parse_json_any(resp_txt)

                if isinstance(parsed, list):
                    # Match results back to indices
                    for obj in parsed:
                        if not isinstance(obj, dict):
                            continue
                        idx = obj.get("index")
                        if idx is not None and 0 <= int(idx) < len(prompt_list):
                            all_results[int(idx)] = {
                                "intents": obj.get("intents", []),
                                "primary_intent": obj.get("primary_intent", "Unclassified"),
                                "topics": obj.get("topics", []),
                                "geo_entities": obj.get("geo_entities", []),
                                "temporal_refs": obj.get("temporal_refs", []),
                                "anti_patterns": obj.get("anti_patterns", []),
                                "has_anti_pattern": bool(obj.get("anti_patterns")),
                            }
                    # Also try positional matching for batches where index wasn't returned
                    for i, obj in enumerate(parsed):
                        if not isinstance(obj, dict):
                            continue
                        if i < len(batch_indices):
                            real_idx = batch_indices[i]
                            if not all_results[real_idx]:
                                all_results[real_idx] = {
                                    "intents": obj.get("intents", []),
                                    "primary_intent": obj.get("primary_intent", "Unclassified"),
                                    "topics": obj.get("topics", []),
                                    "geo_entities": obj.get("geo_entities", []),
                                    "temporal_refs": obj.get("temporal_refs", []),
                                    "anti_patterns": obj.get("anti_patterns", []),
                                    "has_anti_pattern": bool(obj.get("anti_patterns")),
                                }
            except Exception as e:
                errors += 1
                st.warning(f"Batch error: {e}")

            done += len(batch)
            progress.progress(min(1.0, done / len(prompt_list)), text=f"Classified {done}/{len(prompt_list)} prompts...")

        progress.empty()

        # Fill any missing entries with empty defaults
        for i in range(len(all_results)):
            if not all_results[i]:
                all_results[i] = {
                    "intents": ["Unclassified"],
                    "primary_intent": "Unclassified",
                    "topics": [],
                    "geo_entities": [],
                    "temporal_refs": [],
                    "anti_patterns": [],
                    "has_anti_pattern": False,
                }

        # Pad if df has more rows than classified
        while len(all_results) < len(df):
            all_results.append({
                "intents": ["Unclassified"],
                "primary_intent": "Unclassified",
                "topics": [],
                "geo_entities": [],
                "temporal_refs": [],
                "anti_patterns": [],
                "has_anti_pattern": False,
            })

        # ----- Consolidation pass: merge semantically similar labels -----
        st.info("Consolidating similar labels across batches...")
        unique_before = sum(
            len({label for r in all_results for label in r.get(dim, [])})
            for dim in ["intents", "topics", "geo_entities", "temporal_refs", "anti_patterns"]
        )
        all_results = _consolidate_labels(all_results, gemini_api_key, gemini_model)
        unique_after = sum(
            len({label for r in all_results for label in r.get(dim, [])})
            for dim in ["intents", "topics", "geo_entities", "temporal_refs", "anti_patterns"]
        )
        merged = unique_before - unique_after

        st.session_state.pi_gemini_results = all_results
        if errors:
            st.warning(f"Completed with {errors} batch error(s).")
        else:
            msg = f"Classified {done} prompts with Gemini."
            if merged > 0:
                msg += f" Consolidated {merged} duplicate labels."
            st.success(msg)
        st.rerun()

    return st.session_state.pi_gemini_results


# =========================================================================
# Keyword Trends (moved from analytics.py)
# =========================================================================
def _render_keyword_trends(df: pd.DataFrame) -> None:
    """Render keyword trends chart (Google Trends style)."""
    if "prompt" not in df.columns or not len(df):
        return

    st.markdown(
        "### Keyword trends",
        help=(
            "Google Trends-style view of how keywords appear in user prompts over time. "
            "This ignores AI responses."
        ),
    )

    base_kw = df.copy()
    base_kw["prompt"] = base_kw["prompt"].fillna("").astype("string")
    base_kw = base_kw.loc[base_kw["prompt"].astype(str).str.strip().ne("")]
    base_kw["prompt_lc"] = base_kw["prompt"].astype(str).str.lower()

    counts: Counter[str] = Counter()
    for p in base_kw["prompt"].tolist():
        counts.update(_tokenize(p))

    if counts:
        max_count = max(counts.values())
    else:
        max_count = 0

    freq_rows: list[dict[str, Any]] = []
    total_tokens = int(sum(counts.values())) if counts else 0
    for w, c in counts.most_common(30):
        norm = (float(c) * 100.0 / float(max_count)) if max_count > 0 else 0.0
        share = (float(c) * 100.0 / float(total_tokens)) if total_tokens > 0 else 0.0
        freq_rows.append({
            "word": w,
            "count": int(c),
            "normalized_0_100": float(norm),
            "share_%": float(share),
        })

    if freq_rows:
        st.dataframe(pd.DataFrame(freq_rows), hide_index=True, width="stretch")

    top_options = [w for (w, _) in counts.most_common(60)]

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        selected = st.multiselect(
            "Track keywords",
            options=top_options,
            default=top_options[:3],
            help="Select one or more keywords to plot.",
            key="pi_keyword_trends_selected",
        )
    with c2:
        extra_raw = st.text_input(
            "Add keywords (comma separated)",
            value="",
            help="Add additional keywords or phrases to track. Comma-separated.",
            key="pi_keyword_trends_extra",
        )
    with c3:
        normalize = st.toggle(
            "Normalize 0–100",
            value=True,
            help="If enabled, each keyword series is scaled so its peak equals 100.",
            key="pi_keyword_trends_normalize",
        )
        cumulative = st.toggle(
            "Cumulative",
            value=False,
            help="If enabled, plots running total mentions per keyword over time.",
            key="pi_keyword_trends_cumulative",
        )

    extra_terms = [x.strip().lower() for x in str(extra_raw or "").split(",") if x.strip()]
    terms = [t for t in (selected + extra_terms) if isinstance(t, str) and t.strip()]
    terms = list(dict.fromkeys(terms))

    if not terms:
        st.info("Select one or more keywords to see trends.")
    has_ts = "timestamp" in base_kw.columns and base_kw["timestamp"].notna().any()

    if has_ts:
        ts = pd.to_datetime(base_kw["timestamp"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(base_kw["date"], utc=True, errors="coerce") if "date" in base_kw.columns else pd.Series([pd.NaT] * len(base_kw))

    base_kw = base_kw.assign(ts=ts).dropna(subset=["ts"])
    base_kw["bucket"] = base_kw["ts"].dt.floor("D")

    if not len(base_kw) or base_kw["bucket"].notna().sum() == 0:
        st.info("Keyword trends requires timestamps or dates.")
    else:
        _render_trends_chart(base_kw, terms, normalize, cumulative, "keyword")


def _render_trends_chart(
    base_kw: pd.DataFrame,
    terms: list[str],
    normalize: bool,
    cumulative: bool,
    chart_type: str,
) -> None:
    """Shared chart renderer for keyword and bigram trends."""
    start = base_kw["bucket"].min()
    end = base_kw["bucket"].max()
    bucket_vals = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    trend_df = pd.DataFrame({"bucket_dt": bucket_vals})

    for t in terms:
        t_lc = str(t).strip().lower()
        if not t_lc:
            continue
        if " " in t_lc:
            hit = base_kw["prompt_lc"].str.contains(t_lc, regex=False, na=False)
        else:
            pat = fr"\b{re.escape(t_lc)}\b"
            hit = base_kw["prompt_lc"].str.contains(pat, regex=True, na=False)

        series = base_kw.assign(_hit=hit).groupby("bucket")["_hit"].sum()
        trend_df[t_lc] = trend_df["bucket_dt"].map(series).fillna(0).astype(int)

    long = trend_df.melt(id_vars=["bucket_dt"], var_name="term", value_name="value")
    long["raw_value"] = long["value"]
    long["daily_mentions"] = long["raw_value"]

    long = long.sort_values(["term", "bucket_dt"])
    long["cumulative_mentions"] = long.groupby("term")["daily_mentions"].cumsum()

    long["plot_raw"] = long["cumulative_mentions"] if cumulative else long["daily_mentions"]
    long["value"] = long["plot_raw"]

    if normalize:
        max_by_term = long.groupby("term")["value"].max().to_dict()

        def _norm(r: pd.Series) -> float:
            m = float(max_by_term.get(r["term"], 0.0) or 0.0)
            return float(r["value"]) * 100.0 / m if m > 0 else 0.0

        long["value"] = long.apply(_norm, axis=1)

    y_axis = alt.Axis(values=[0, 20, 40, 60, 80, 100], ticks=True, tickSize=6, grid=True) if normalize else alt.Axis(tickCount=6, ticks=True, tickSize=6, grid=True)

    y_title = (
        ("Cumulative interest" if cumulative else "Interest")
        if normalize
        else ("Cumulative mentions" if cumulative else "Mentions")
    )

    tooltip = [
        alt.Tooltip("term:N", title="Keyword" if chart_type == "keyword" else "Phrase"),
        alt.Tooltip("bucket_dt:T", title="Date", format="%d %b %Y"),
        alt.Tooltip("daily_mentions:Q", title="Daily mentions"),
        alt.Tooltip("cumulative_mentions:Q", title="Cumulative mentions"),
        alt.Tooltip("value:Q", title=y_title, format=".1f" if normalize else ","),
    ]

    base = (
        alt.Chart(long)
        .encode(
            x=alt.X("bucket_dt:T", title="Time", axis=alt.Axis(format="%d %b", labelAngle=0)),
            y=alt.Y("value:Q", title=y_title, axis=y_axis),
            color=alt.Color("term:N", title="Keyword" if chart_type == "keyword" else "Phrase"),
        )
    )

    nearest = alt.selection_point(
        nearest=True,
        on="mouseover",
        fields=["bucket_dt"],
        empty=False,
        clear="mouseout",
    )

    lines = base.mark_line(point=False)
    points = (
        base.mark_point(size=80)
        .encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=tooltip,
        )
        .add_params(nearest)
    )
    rule = (
        alt.Chart(long)
        .mark_rule()
        .encode(x="bucket_dt:T")
        .transform_filter(nearest)
    )

    chart = (lines + rule + points).properties(height=520)
    st.altair_chart(chart, use_container_width=True)


# =========================================================================
# Bigram Trends (new — mirrors keyword trends)
# =========================================================================
def _render_bigram_trends(df: pd.DataFrame) -> None:
    """Render bigram trends chart (same interaction as keyword trends)."""
    if "prompt" not in df.columns or not len(df):
        return

    st.markdown(
        "### Bigram trends",
        help=(
            "Two-word phrase trends over time — captures richer context than single keywords. "
            "E.g. 'forest loss', 'land cover', 'natural land'."
        ),
    )

    prompt_list = df["prompt"].fillna("").astype(str).tolist()
    bigrams = top_bigrams(prompt_list, top_n=40)

    if not bigrams:
        st.info("No bigrams extracted from prompts.")
        return

    bg_options = [bg for bg, _ in bigrams]

    bg_c1, bg_c2, bg_c3 = st.columns([2, 2, 1])
    with bg_c1:
        bg_selected = st.multiselect(
            "Track phrases",
            options=bg_options,
            default=bg_options[:3],
            help="Select two-word phrases to plot over time.",
            key="pi_bigram_trends_selected",
        )
    with bg_c2:
        bg_extra_raw = st.text_input(
            "Add phrases (comma separated)",
            value="",
            help="Add additional phrases to track. Comma-separated.",
            key="pi_bigram_trends_extra",
        )
    with bg_c3:
        bg_normalize = st.toggle(
            "Normalize 0–100",
            value=True,
            key="pi_bigram_trends_normalize",
        )
        bg_cumulative = st.toggle(
            "Cumulative",
            value=False,
            key="pi_bigram_trends_cumulative",
        )

    bg_extra = [x.strip().lower() for x in str(bg_extra_raw or "").split(",") if x.strip()]
    bg_terms = [t for t in (bg_selected + bg_extra) if isinstance(t, str) and t.strip()]
    bg_terms = list(dict.fromkeys(bg_terms))

    if not bg_terms:
        st.info("Select one or more phrases to see trends.")
        return

    base_bg = df.copy()
    base_bg["prompt"] = base_bg["prompt"].fillna("").astype("string")
    base_bg = base_bg.loc[base_bg["prompt"].astype(str).str.strip().ne("")]
    base_bg["prompt_lc"] = base_bg["prompt"].astype(str).str.lower()

    has_ts = "timestamp" in base_bg.columns and base_bg["timestamp"].notna().any()
    if has_ts:
        ts = pd.to_datetime(base_bg["timestamp"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(base_bg["date"], utc=True, errors="coerce") if "date" in base_bg.columns else pd.Series([pd.NaT] * len(base_bg))

    base_bg = base_bg.assign(ts=ts).dropna(subset=["ts"])
    base_bg["bucket"] = base_bg["ts"].dt.floor("D")

    if not len(base_bg) or base_bg["bucket"].notna().sum() == 0:
        st.info("Bigram trends requires timestamps or dates.")
        return

    _render_trends_chart(base_bg, bg_terms, bg_normalize, bg_cumulative, "bigram")


# =========================================================================
# Intent section
# =========================================================================
def _render_intent_section(
    df: pd.DataFrame,
    prompt_analysis: list[dict[str, Any]],
    classify_mode: str,
) -> None:
    st.markdown(
        "#### What are users trying to do?",
        help="Classifies each prompt into action types. A prompt can match multiple intents.",
    )

    # In regex mode, treat Unclassified as a diagnostic bucket rather than a real category.
    # We'll exclude it from charts and show the underlying prompts in a table instead.
    hide_unclassified = classify_mode != "Gemini (LLM)"
    unclassified_idx: list[int] = []
    if hide_unclassified:
        for i, pa in enumerate(prompt_analysis):
            pi = str(pa.get("primary_intent") or "")
            intents = pa.get("intents", [])
            if pi == "Unclassified" or intents == ["Unclassified"]:
                unclassified_idx.append(i)

    all_intents_flat: list[str] = []
    for pa in prompt_analysis:
        intents = list(pa.get("intents", []) or [])
        if hide_unclassified:
            intents = [x for x in intents if str(x) != "Unclassified"]
        all_intents_flat.extend(intents)
    intent_counts = Counter(all_intents_flat)
    intent_df_raw, intent_other_df = _top_n_with_other(intent_counts, top_n=20, total_den=len(df))
    if not len(intent_df_raw):
        if hide_unclassified and len(unclassified_idx):
            st.info("All prompts were Unclassified by regex intent rules.")
        else:
            st.info("No intents detected.")

        if hide_unclassified and len(unclassified_idx):
            show_unclassified = st.selectbox(
                "Unclassified prompts",
                options=["Hide", f"Show ({len(unclassified_idx)})"],
                index=0,
                key="pi_regex_unclassified_show",
            )
            if str(show_unclassified).startswith("Show"):
                cols = [c for c in ["trace_id", "prompt", "outcome"] if c in df.columns]
                tdf = df.iloc[unclassified_idx][cols].copy() if cols else df.iloc[unclassified_idx].copy()
                st.dataframe(tdf.head(200), hide_index=True, width="stretch")
        return

    intent_df = intent_df_raw.rename(columns={"label": "intent"})
    intent_order = (
        intent_df.loc[intent_df["intent"] != "Other", ["intent", "count"]]
        .sort_values("count", ascending=False)["intent"]
        .astype(str)
        .tolist()
    )
    if "Other" in intent_df["intent"].astype(str).tolist():
        intent_order.append("Other")

    intent_c1, intent_c2 = st.columns(2)

    with intent_c1:
        st.markdown("##### Intent breakdown")
        intent_bar = (
            alt.Chart(intent_df)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Prompts"),
                y=alt.Y("intent:N", sort=intent_order, title="Intent"),
                color=alt.Color("intent:N", title="Intent", legend=None),
                tooltip=[
                    alt.Tooltip("intent:N", title="Intent"),
                    alt.Tooltip("count:Q", title="Prompts", format=","),
                    alt.Tooltip("percent:Q", title="% of traces", format=".1f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(intent_bar, width="stretch")

        if len(intent_other_df):
            with st.expander("Other intents (rolled up)", expanded=False):
                st.dataframe(
                    intent_other_df.rename(columns={"label": "intent"}),
                    hide_index=True,
                    width="stretch",
                )

        if hide_unclassified and len(unclassified_idx):
            show_unclassified = st.selectbox(
                "Unclassified prompts",
                options=["Hide", f"Show ({len(unclassified_idx)})"],
                index=0,
                key="pi_regex_unclassified_show_post",
            )
            if str(show_unclassified).startswith("Show"):
                cols = [c for c in ["trace_id", "prompt", "outcome"] if c in df.columns]
                tdf = df.iloc[unclassified_idx][cols].copy() if cols else df.iloc[unclassified_idx].copy()
                st.dataframe(tdf.head(200), hide_index=True, width="stretch")

    with intent_c2:
        st.markdown("##### Intent vs outcome")
        if "outcome" in df.columns:
            intent_outcome_rows: list[dict[str, Any]] = []
            for pa, outcome in zip(prompt_analysis, df["outcome"].tolist()):
                intents = list(pa.get("intents", []) or [])
                if hide_unclassified:
                    intents = [x for x in intents if str(x) != "Unclassified"]
                for intent in intents:
                    intent_outcome_rows.append({"intent": intent, "outcome": outcome})
            io_df = pd.DataFrame(intent_outcome_rows)
            if len(io_df):
                # Limit intents in this chart too (top 20 + Other) to avoid huge stacked bars
                intent_outcome_counts = Counter(io_df["intent"].fillna("").astype(str).tolist())
                io_top_df, io_other_df = _top_n_with_other(intent_outcome_counts, top_n=20, total_den=len(io_df))
                top_intents = set(io_top_df[io_top_df["label"] != "Other"]["label"].tolist())
                io_df["intent_limited"] = io_df["intent"].map(lambda x: x if x in top_intents else "Other")

                io_summary = io_df.groupby(["intent_limited", "outcome"]).size().reset_index(name="count")
                intent_totals = (
                    io_summary.groupby("intent_limited")["count"].sum().sort_values(ascending=False)
                )
                io_intent_order = [i for i in intent_totals.index.astype(str).tolist() if i != "Other"]
                if "Other" in intent_totals.index.astype(str).tolist():
                    io_intent_order.append("Other")
                io_chart = (
                    alt.Chart(io_summary)
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Prompts", stack="normalize"),
                        y=alt.Y("intent_limited:N", sort=io_intent_order, title="Intent"),
                        color=_outcome_color(),
                        tooltip=[
                            alt.Tooltip("intent_limited:N", title="Intent"),
                            alt.Tooltip("outcome:N", title="Outcome"),
                            alt.Tooltip("count:Q", title="Prompts", format=","),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(io_chart, width="stretch")

                if len(io_other_df):
                    with st.expander("Other intents in Intent vs outcome (rolled up)", expanded=False):
                        st.dataframe(
                            io_other_df.rename(columns={"label": "intent"}),
                            hide_index=True,
                            width="stretch",
                        )


# =========================================================================
# Topic section
# =========================================================================
def _render_topic_section(df: pd.DataFrame, prompt_analysis: list[dict[str, Any]]) -> None:
    st.markdown(
        "#### What topics are users asking about?",
        help="Extracts environmental topics from prompts. Identifies product coverage gaps.",
    )

    all_topics_flat: list[str] = []
    for pa in prompt_analysis:
        all_topics_flat.extend(pa.get("topics", []))
    topic_counts = Counter(all_topics_flat)
    no_topic_count = sum(1 for pa in prompt_analysis if not pa.get("topics"))

    topic_c1, topic_c2 = st.columns(2)

    with topic_c1:
        if topic_counts:
            topic_df_raw, topic_other_df = _top_n_with_other(topic_counts, top_n=20, total_den=len(df))
            topic_df = topic_df_raw.rename(columns={"label": "topic"})
            topic_order = (
                topic_df.loc[topic_df["topic"] != "Other", ["topic", "count"]]
                .sort_values("count", ascending=False)["topic"]
                .astype(str)
                .tolist()
            )
            if "Other" in topic_df["topic"].astype(str).tolist():
                topic_order.append("Other")
            st.markdown("##### Topic frequency")
            topic_bar = (
                alt.Chart(topic_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Prompts"),
                    y=alt.Y("topic:N", sort=topic_order, title="Topic"),
                    color=alt.Color("topic:N", title="Topic", legend=None),
                    tooltip=[
                        alt.Tooltip("topic:N", title="Topic"),
                        alt.Tooltip("count:Q", title="Prompts", format=","),
                        alt.Tooltip("percent:Q", title="% of traces", format=".1f"),
                    ],
                )
                .properties(height=max(200, len(topic_df) * 28))
            )
            st.altair_chart(topic_bar, width="stretch")

            if len(topic_other_df):
                with st.expander("Other topics (rolled up)", expanded=False):
                    st.dataframe(
                        topic_other_df.rename(columns={"label": "topic"}),
                        hide_index=True,
                        width="stretch",
                    )
        else:
            st.info("No environmental topics detected in prompts.")

    with topic_c2:
        st.markdown("##### Topic coverage gap")
        coverage_df = pd.DataFrame([
            {"label": "Topic detected", "count": len(df) - no_topic_count},
            {"label": "No topic detected", "count": no_topic_count},
        ])
        coverage_df["percent"] = (coverage_df["count"] / max(1, int(coverage_df["count"].sum())) * 100).round(1)
        coverage_pie = (
            alt.Chart(coverage_df)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color(
                    "label:N", title="",
                    scale=alt.Scale(domain=["Topic detected", "No topic detected"], range=["#30a46c", "#e5484d"]),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title=""),
                    alt.Tooltip("count:Q", title="Prompts", format=","),
                    alt.Tooltip("percent:Q", title="%", format=".1f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(coverage_pie, width="stretch")

        if no_topic_count > 0:
            with st.expander(f"View prompts with no detected topic ({no_topic_count})", expanded=False):
                no_topic_df = df[df["topics"] == "None detected"][["prompt", "outcome"]].head(50)
                st.dataframe(no_topic_df, hide_index=True, width="stretch")


# =========================================================================
# Geo & Temporal section
# =========================================================================
def _render_geo_temporal_section(df: pd.DataFrame, prompt_analysis: list[dict[str, Any]]) -> None:
    st.markdown("#### Where and when are users focused?")

    geo_c1, geo_c2 = st.columns(2)

    with geo_c1:
        all_geo_flat: list[str] = []
        for pa in prompt_analysis:
            all_geo_flat.extend(pa.get("geo_entities", []))
        geo_counts = Counter(all_geo_flat)
        if geo_counts:
            geo_df_raw, geo_other_df = _top_n_with_other(geo_counts, top_n=20, total_den=len(df))
            geo_df = geo_df_raw.rename(columns={"label": "location"})
            geo_order = (
                geo_df.loc[geo_df["location"] != "Other", ["location", "count"]]
                .sort_values("count", ascending=False)["location"]
                .astype(str)
                .tolist()
            )
            if "Other" in geo_df["location"].astype(str).tolist():
                geo_order.append("Other")
            st.markdown("##### Top locations mentioned")
            geo_bar = (
                alt.Chart(geo_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Mentions"),
                    y=alt.Y("location:N", sort=geo_order, title="Location"),
                    tooltip=[
                        alt.Tooltip("location:N", title="Location"),
                        alt.Tooltip("count:Q", title="Mentions", format=","),
                        alt.Tooltip("percent:Q", title="% of traces", format=".1f"),
                    ],
                )
                .properties(height=max(200, min(len(geo_df), 21) * 22))
            )
            st.altair_chart(geo_bar, width="stretch")

            if len(geo_other_df):
                with st.expander("Other locations (rolled up)", expanded=False):
                    st.dataframe(
                        geo_other_df.rename(columns={"label": "location"}),
                        hide_index=True,
                        width="stretch",
                    )
        else:
            st.info("No geographic entities detected.")

    with geo_c2:
        all_temporal_flat: list[str] = []
        for pa in prompt_analysis:
            all_temporal_flat.extend(pa.get("temporal_refs", []))
        temporal_counts = Counter(all_temporal_flat)
        if temporal_counts:
            temporal_df_raw, temporal_other_df = _top_n_with_other(temporal_counts, top_n=20, total_den=len(df))
            temporal_df = temporal_df_raw.rename(columns={"label": "timeframe"})
            temporal_order = (
                temporal_df.loc[temporal_df["timeframe"] != "Other", ["timeframe", "count"]]
                .sort_values("count", ascending=False)["timeframe"]
                .astype(str)
                .tolist()
            )
            if "Other" in temporal_df["timeframe"].astype(str).tolist():
                temporal_order.append("Other")
            st.markdown("##### Temporal framing")
            temporal_bar = (
                alt.Chart(temporal_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Prompts"),
                    y=alt.Y("timeframe:N", sort=temporal_order, title="Timeframe"),
                    tooltip=[
                        alt.Tooltip("timeframe:N", title="Timeframe"),
                        alt.Tooltip("count:Q", title="Prompts", format=","),
                        alt.Tooltip("percent:Q", title="% of traces", format=".1f"),
                    ],
                )
                .properties(height=max(200, len(temporal_df) * 28))
            )
            st.altair_chart(temporal_bar, width="stretch")

            if len(temporal_other_df):
                with st.expander("Other temporal classes (rolled up)", expanded=False):
                    st.dataframe(
                        temporal_other_df.rename(columns={"label": "timeframe"}),
                        hide_index=True,
                        width="stretch",
                    )
        else:
            st.info("No temporal references detected.")


# =========================================================================
# Rising vs Declining
# =========================================================================
def _render_rising_declining(
    df: pd.DataFrame,
    prompt_analysis: list[dict[str, Any]],
    classify_mode: str,
) -> None:
    if "timestamp" not in df.columns or not df["timestamp"].notna().any():
        return
    ts_valid = df["timestamp"].dropna()
    if len(ts_valid) <= 10:
        return

    midpoint = ts_valid.median()
    first_half_mask = df["timestamp"] <= midpoint
    second_half_mask = df["timestamp"] > midpoint

    # Split prompt_analysis by mask
    first_pa = [pa for pa, m in zip(prompt_analysis, first_half_mask) if m]
    second_pa = [pa for pa, m in zip(prompt_analysis, second_half_mask) if m]

    # Topic momentum
    first_topics: Counter[str] = Counter()
    second_topics: Counter[str] = Counter()
    for pa in first_pa:
        first_topics.update(pa.get("topics", []))
    for pa in second_pa:
        second_topics.update(pa.get("topics", []))

    all_topic_names = set(first_topics.keys()) | set(second_topics.keys())
    if not all_topic_names:
        return

    st.markdown(
        "#### Rising vs declining topics",
        help="Compares topic frequency in the first half vs second half of the date range.",
    )

    n_first = max(1, len(first_pa))
    n_second = max(1, len(second_pa))
    trend_rows: list[dict[str, Any]] = []
    for topic in all_topic_names:
        rate_first = first_topics.get(topic, 0) / n_first * 100
        rate_second = second_topics.get(topic, 0) / n_second * 100
        change = rate_second - rate_first
        trend_rows.append({
            "topic": topic,
            "first_half_%": round(rate_first, 1),
            "second_half_%": round(rate_second, 1),
            "change_pp": round(change, 1),
            "direction": "Rising" if change > 0.5 else ("Declining" if change < -0.5 else "Stable"),
        })

    trend_df = pd.DataFrame(trend_rows).sort_values("change_pp", ascending=False)

    trend_c1, trend_c2 = st.columns(2)
    with trend_c1:
        st.markdown("##### Topic momentum")
        momentum_chart = (
            alt.Chart(trend_df)
            .mark_bar()
            .encode(
                x=alt.X("change_pp:Q", title="Change (pp)"),
                y=alt.Y("topic:N", sort="-x", title="Topic"),
                color=alt.condition(
                    alt.datum.change_pp > 0,
                    alt.value("#30a46c"),
                    alt.value("#e5484d"),
                ),
                tooltip=[
                    alt.Tooltip("topic:N", title="Topic"),
                    alt.Tooltip("first_half_%:Q", title="1st half %", format=".1f"),
                    alt.Tooltip("second_half_%:Q", title="2nd half %", format=".1f"),
                    alt.Tooltip("change_pp:Q", title="Change (pp)", format="+.1f"),
                    alt.Tooltip("direction:N", title="Direction"),
                ],
            )
            .properties(height=max(200, len(trend_df) * 28))
        )
        st.altair_chart(momentum_chart, width="stretch")

    with trend_c2:
        st.markdown("##### Momentum table")
        st.dataframe(trend_df, hide_index=True, width="stretch")

    # Intent momentum
    first_intents: Counter[str] = Counter()
    second_intents: Counter[str] = Counter()
    for pa in first_pa:
        first_intents.update(pa.get("intents", []))
    for pa in second_pa:
        second_intents.update(pa.get("intents", []))

    all_intent_names = set(first_intents.keys()) | set(second_intents.keys())
    if all_intent_names:
        intent_trend_rows: list[dict[str, Any]] = []
        for intent in all_intent_names:
            rate_first = first_intents.get(intent, 0) / n_first * 100
            rate_second = second_intents.get(intent, 0) / n_second * 100
            change = rate_second - rate_first
            intent_trend_rows.append({
                "intent": intent,
                "first_half_%": round(rate_first, 1),
                "second_half_%": round(rate_second, 1),
                "change_pp": round(change, 1),
            })

        intent_trend_df = pd.DataFrame(intent_trend_rows).sort_values("change_pp", ascending=False)
        st.markdown("##### Intent momentum")
        intent_momentum = (
            alt.Chart(intent_trend_df)
            .mark_bar()
            .encode(
                x=alt.X("change_pp:Q", title="Change (pp)"),
                y=alt.Y("intent:N", sort="-x", title="Intent"),
                color=alt.condition(
                    alt.datum.change_pp > 0,
                    alt.value("#30a46c"),
                    alt.value("#e5484d"),
                ),
                tooltip=[
                    alt.Tooltip("intent:N", title="Intent"),
                    alt.Tooltip("first_half_%:Q", title="1st half %", format=".1f"),
                    alt.Tooltip("second_half_%:Q", title="2nd half %", format=".1f"),
                    alt.Tooltip("change_pp:Q", title="Change (pp)", format="+.1f"),
                ],
            )
            .properties(height=max(180, len(intent_trend_df) * 30))
        )
        st.altair_chart(intent_momentum, width="stretch")


# =========================================================================
# Prompt-to-Outcome Correlation
# =========================================================================
def _render_prompt_outcome_correlation(df: pd.DataFrame, prompt_analysis: list[dict[str, Any]]) -> None:
    st.markdown(
        "#### Prompt characteristics vs outcomes",
        help="Which prompt patterns correlate with errors, deferrals, or success.",
    )

    if "outcome" not in df.columns:
        return

    corr_c1, corr_c2 = st.columns(2)

    with corr_c1:
        st.markdown("##### Prompt length vs outcome")
        if "prompt_len_words" in df.columns:
            box_df = df[["prompt_len_words", "outcome"]].dropna()
            if len(box_df):
                box_chart = (
                    alt.Chart(box_df)
                    .mark_boxplot(extent="min-max")
                    .encode(
                        x=alt.X("outcome:N", title="Outcome", sort=_OUTCOME_DOMAIN),
                        y=alt.Y("prompt_len_words:Q", title="Prompt length (words)"),
                        color=_outcome_color(),
                    )
                    .properties(height=280)
                )
                st.altair_chart(box_chart, width="stretch")

    with corr_c2:
        st.markdown("##### Has geo context vs outcome")
        has_geo = df["geo_entities"].astype(str).str.strip().ne("")
        geo_outcome_rows: list[dict[str, Any]] = []
        for has, label in [(True, "Has location"), (False, "No location")]:
            subset = df[has_geo == has]
            if len(subset):
                for ov in _OUTCOME_DOMAIN:
                    rate = float((subset["outcome"] == ov).mean()) * 100
                    geo_outcome_rows.append({"geo_context": label, "outcome": ov, "rate_%": round(rate, 1)})
        if geo_outcome_rows:
            geo_outcome_df = pd.DataFrame(geo_outcome_rows)
            geo_outcome_chart = (
                alt.Chart(geo_outcome_df)
                .mark_bar()
                .encode(
                    x=alt.X("rate_%:Q", title="% of prompts", stack="zero"),
                    y=alt.Y("geo_context:N", title=""),
                    color=_outcome_color(),
                    tooltip=[
                        alt.Tooltip("geo_context:N", title="Geo context"),
                        alt.Tooltip("outcome:N", title="Outcome"),
                        alt.Tooltip("rate_%:Q", title="%", format=".1f"),
                    ],
                )
                .properties(height=120)
            )
            st.altair_chart(geo_outcome_chart, width="stretch")

    # Error rate by topic
    st.markdown("##### Error rate by topic")
    topic_error_rows: list[dict[str, Any]] = []
    for pa, outcome in zip(prompt_analysis, df["outcome"].tolist()):
        for topic in pa.get("topics", []):
            topic_error_rows.append({"topic": topic, "outcome": outcome})
    if topic_error_rows:
        te_df = pd.DataFrame(topic_error_rows)
        te_summary = te_df.groupby("topic").agg(
            total=("outcome", "count"),
            errors=("outcome", lambda x: int((x.isin(["ERROR", "SOFT_ERROR"])).sum())),
            success=("outcome", lambda x: int((x == "ANSWER").sum())),
        ).reset_index()
        te_summary["error_rate_%"] = (te_summary["errors"] / te_summary["total"].clip(lower=1) * 100).round(1)
        te_summary["success_rate_%"] = (te_summary["success"] / te_summary["total"].clip(lower=1) * 100).round(1)
        te_summary = te_summary.sort_values("error_rate_%", ascending=False)
        st.dataframe(te_summary, hide_index=True, width="stretch")


# =========================================================================
# Anti-pattern section
# =========================================================================
def _render_anti_patterns(df: pd.DataFrame, prompt_analysis: list[dict[str, Any]]) -> None:
    st.markdown(
        "#### Prompt anti-patterns",
        help="Detects problematic prompt patterns to inform UX improvements.",
    )

    anti_pattern_count = int(df["has_anti_pattern"].sum())
    clean_count = int(len(df) - anti_pattern_count)

    ap_c1, ap_c2 = st.columns(2)

    with ap_c1:
        st.markdown("##### Anti-pattern prevalence")
        ap_pie_df = pd.DataFrame([
            {"label": "Clean", "count": clean_count},
            {"label": "Has anti-pattern", "count": anti_pattern_count},
        ])
        ap_pie_df["percent"] = (ap_pie_df["count"] / max(1, int(ap_pie_df["count"].sum())) * 100).round(1)
        ap_pie = (
            alt.Chart(ap_pie_df)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color(
                    "label:N", title="",
                    scale=alt.Scale(domain=["Clean", "Has anti-pattern"], range=["#30a46c", "#e5484d"]),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title=""),
                    alt.Tooltip("count:Q", title="Prompts", format=","),
                    alt.Tooltip("percent:Q", title="%", format=".1f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(ap_pie, width="stretch")

    with ap_c2:
        st.markdown("##### Anti-pattern breakdown")
        all_ap_flat: list[str] = []
        for pa in prompt_analysis:
            all_ap_flat.extend(pa.get("anti_patterns", []))
        ap_counts = Counter(all_ap_flat)
        if ap_counts:
            ap_bar_df_raw, ap_other_df = _top_n_with_other(ap_counts, top_n=20, total_den=len(df))
            ap_bar_df = ap_bar_df_raw.rename(columns={"label": "pattern"})
            ap_order = (
                ap_bar_df.loc[ap_bar_df["pattern"] != "Other", ["pattern", "count"]]
                .sort_values("count", ascending=False)["pattern"]
                .astype(str)
                .tolist()
            )
            if "Other" in ap_bar_df["pattern"].astype(str).tolist():
                ap_order.append("Other")
            ap_bar = (
                alt.Chart(ap_bar_df)
                .mark_bar(color="#e5484d")
                .encode(
                    x=alt.X("count:Q", title="Prompts"),
                    y=alt.Y("pattern:N", sort=ap_order, title="Anti-pattern"),
                    tooltip=[
                        alt.Tooltip("pattern:N", title="Anti-pattern"),
                        alt.Tooltip("count:Q", title="Prompts", format=","),
                        alt.Tooltip("percent:Q", title="% of traces", format=".1f"),
                    ],
                )
                .properties(height=max(180, len(ap_bar_df) * 30))
            )
            st.altair_chart(ap_bar, width="stretch")

            if len(ap_other_df):
                with st.expander("Other anti-patterns (rolled up)", expanded=False):
                    st.dataframe(
                        ap_other_df.rename(columns={"label": "pattern"}),
                        hide_index=True,
                        width="stretch",
                    )
        else:
            st.success("No anti-patterns detected in any prompts!")

    # Anti-pattern vs outcome
    if "outcome" in df.columns:
        st.markdown("##### Anti-pattern impact on outcomes")
        ap_outcome_rows: list[dict[str, Any]] = []
        for has, label in [(True, "Has anti-pattern"), (False, "Clean prompt")]:
            subset = df[df["has_anti_pattern"] == has]
            if len(subset):
                for ov in _OUTCOME_DOMAIN:
                    rate = float((subset["outcome"] == ov).mean()) * 100
                    ap_outcome_rows.append({"prompt_quality": label, "outcome": ov, "rate_%": round(rate, 1)})
        if ap_outcome_rows:
            ap_outcome_df = pd.DataFrame(ap_outcome_rows)
            ap_outcome_chart = (
                alt.Chart(ap_outcome_df)
                .mark_bar()
                .encode(
                    x=alt.X("rate_%:Q", title="% of prompts", stack="zero"),
                    y=alt.Y("prompt_quality:N", title=""),
                    color=_outcome_color(),
                    tooltip=[
                        alt.Tooltip("prompt_quality:N", title="Prompt quality"),
                        alt.Tooltip("outcome:N", title="Outcome"),
                        alt.Tooltip("rate_%:Q", title="%", format=".1f"),
                    ],
                )
                .properties(height=120)
            )
            st.altair_chart(ap_outcome_chart, width="stretch")

    if anti_pattern_count > 0:
        with st.expander(f"View anti-pattern prompts ({anti_pattern_count})", expanded=False):
            ap_sample = df[df["has_anti_pattern"]][["prompt", "anti_patterns", "outcome"]].head(50)
            st.dataframe(ap_sample, hide_index=True, width="stretch")


# =========================================================================
# Export
# =========================================================================
def _render_export(df: pd.DataFrame) -> None:
    with st.expander("Prompt Intelligence data export", expanded=False):
        pi_export_cols = [
            "trace_id", "prompt", "primary_intent", "all_intents", "topics",
            "geo_entities", "temporal_refs", "anti_patterns", "has_anti_pattern", "outcome",
        ]
        pi_export_cols = [c for c in pi_export_cols if c in df.columns]
        pi_export = df[pi_export_cols].copy()
        st.dataframe(pi_export, hide_index=True, width="stretch")
        pi_csv = csv_bytes_any(pi_export.to_dict("records"))
        st.download_button(
            "Download Prompt Intelligence CSV",
            pi_csv,
            "prompt_intelligence.csv",
            "text/csv",
            key="pi_export_csv",
        )
