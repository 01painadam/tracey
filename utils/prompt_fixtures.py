"""LLM prompt templates for the Product Intelligence pipeline.

All prompts are GNW (Global Nature Watch) domain-specific.
"""

# ---------------------------------------------------------------------------
# Stage 1: Enrichment prompts
# ---------------------------------------------------------------------------

DEFAULT_ENRICHMENT_SYSTEM = (
    "You are a product analyst for Global Nature Watch (GNW), an AI-powered "
    "geospatial platform that helps users analyse land cover change, deforestation, "
    "biodiversity, restoration, and related environmental topics. "
    "Users interact with a chat assistant that has access to GIS datasets and tools."
)

DEFAULT_ENRICHMENT_PROMPT = """\
Analyse this user–assistant interaction and extract structured metadata.

KNOWN CONTEXT (from system logs — treat as ground truth):
- Heuristic outcome: {outcome_heuristic}
- Tools used: {tools_used}
- Datasets analysed: {datasets_analysed}
- AOI name: {aoi_name}
- AOI type: {aoi_type}

USER PROMPT:
{user_prompt}

ASSISTANT RESPONSE (may be truncated):
{assistant_response}

Extract the following fields:
- topic: Primary topic. One of: land_cover_change, deforestation, biodiversity, restoration, agriculture_cropland, urbanisation, fire_disturbance, water_resources, climate, protected_areas, general_knowledge, other.
- dataset_required: The GNW dataset needed to answer this query (e.g. land_cover, tree_cover_loss, fire_alerts, crop_extent, protected_areas, elevation). Null if unclear.
- query_type: One of: spatial_analysis, trend_analysis, comparison, data_lookup, explanation, metadata, other.
- geographic_scope: Country, region, or area mentioned. Use 'custom_aoi' for drawn polygons. Null if none.
- intent: The user's job-to-be-done as a short phrase (e.g. 'monitor deforestation trends for annual report').
- complexity: One of: simple, moderate, complex.
- outcome: How well the assistant handled it. One of: success, partial, failure, unclear. Use the KNOWN CONTEXT heuristic outcome as a strong signal.
- failure_mode: If outcome is partial/failure, why. One of: missing_dataset, wrong_interpretation, capability_gap, api_error, ambiguous_query, geographic_gap. Null if success/unclear.

Return JSON matching the schema."""

DEFAULT_ENRICHMENT_BATCH_PROMPT = """\
Analyse each user–assistant interaction and extract structured metadata.

For each trace, extract:
- topic: Primary topic. One of: land_cover_change, deforestation, biodiversity, restoration, agriculture_cropland, urbanisation, fire_disturbance, water_resources, climate, protected_areas, general_knowledge, other.
- dataset_required: The GNW dataset needed (e.g. land_cover, tree_cover_loss, fire_alerts, crop_extent, protected_areas, elevation). Null if unclear.
- query_type: One of: spatial_analysis, trend_analysis, comparison, data_lookup, explanation, metadata, other.
- geographic_scope: Country/region/area mentioned. 'custom_aoi' for drawn polygons. Null if none.
- intent: User's job-to-be-done as a short phrase.
- complexity: One of: simple, moderate, complex.
- outcome: How well the assistant handled it. One of: success, partial, failure, unclear. Use each trace's KNOWN CONTEXT as a strong signal.
- failure_mode: If partial/failure, why. One of: missing_dataset, wrong_interpretation, capability_gap, api_error, ambiguous_query, geographic_gap. Null otherwise.

Return a JSON object with an "items" key containing an array. Each item must include "trace_id" plus all fields above.

TRACES:
{traces_json}"""


# ---------------------------------------------------------------------------
# Stage 2: Evidence Mining (kept from original, adapted for GNW)
# ---------------------------------------------------------------------------

DEFAULT_EVIDENCE_PROMPT = """\
You are a relevance scorer for Global Nature Watch trace analysis. \
Given a hypothesis and a trace, score how relevant the trace is.

HYPOTHESIS/CRITERIA: {hypothesis}

TRACE:
{search_text}

Return JSON: {{"relevant": true/false, "score": 0-100, "reason": "brief explanation"}}"""


# ---------------------------------------------------------------------------
# Stage 3: Report generation (used with Claude CLI)
# ---------------------------------------------------------------------------

DEFAULT_REPORT_PROMPT = """\
You are a senior product analyst writing a product intelligence report for \
Global Nature Watch (GNW), an AI geospatial platform.

Below is structured analysis data from {trace_count} user traces \
({date_range}). Session URLs follow the pattern: \
https://www.globalnaturewatch.org/app/threads/{{session_id}}

{analysis_data}

Write a concise product intelligence report (~200 lines of markdown) with \
these sections in order:

## 1. Analysis Type Classification
Classify what kinds of analysis users request. Categories: simple quantification, \
trend/time-series, spatial visualisation, ranking/leaderboard, comparative, \
causal/driver, metadata/info, complex multi-step. Show distribution table + examples.

## 2. Welcome Prompt Impact
Compare welcome-prompt sessions vs organic sessions: engagement (bounce rate, \
session depth), success rates. Flag any metric inflation from welcome prompts.

## 3. Dataset Demand Mapping
Which datasets are used, at what volume? Separate organic vs welcome-prompt-inflated \
counts. Note temporal/capability mismatches.

## 4. Unmet Demand
Topics users ask about that no current dataset serves. Where possible, suggest \
specific datasets from https://data.globalforestwatch.org that could fill gaps.

## 5. Topic & Geographic Distribution
What are people asking about and where? Use the topic and geographic_scope data.

## 6. Session Behaviour
Turn depth, bounce rates, returning users, power user patterns.

## 7. Jobs to Be Done
Identify 5-7 JTBDs with the datasets that serve each one.

## 8. Recommendations
Actionable product implications.

**Formatting rules:**
- Hyperlink session IDs as [session_id](https://www.globalnaturewatch.org/app/threads/{{session_id}})
- Use tables over prose where density helps
- Keep under ~200 lines
- Use markdown"""


# ---------------------------------------------------------------------------
# Legacy prompts (kept for backward compatibility with evidence mining)
# ---------------------------------------------------------------------------

DEFAULT_TAGGING_PROMPT = """\
You are a trace tagger for Global Nature Watch. Given a user prompt and \
assistant output, tag the trace according to these criteria:

{criteria_desc}

USER PROMPT:
{user_prompt}

ASSISTANT OUTPUT:
{assistant_output}

Return JSON with keys: {criteria_keys}. Values should be short labels or categories."""

DEFAULT_GAP_ANALYSIS_PROMPT = """\
You are a product analyst. Analyse these conversation traces between users \
and a GIS AI assistant for Global Nature Watch.

{traces_summary}

Generate a product gap analysis report with these sections:

## User Jobs (JTBD)
List the main jobs users are trying to accomplish, with rough frequency.

## Coverage Assessment
For each job, assess how well the assistant handles it.

## Gap Identification
What are the biggest gaps? Missing capabilities, failure patterns, quality issues.

## Recommendations
Top 3-5 product recommendations.

Format as clean markdown."""
