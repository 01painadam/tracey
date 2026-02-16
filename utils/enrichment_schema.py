"""Pydantic models for structured Gemini enrichment output.

These models define the schema for trace metadata extracted by Gemini
during the enrichment stage of the Product Intelligence pipeline.
Designed for the GNW (Global Nature Watch) GIS domain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Allowed values (used in field descriptions for Gemini + validation)
# ---------------------------------------------------------------------------

TOPIC_VALUES = [
    "land_cover_change",
    "deforestation",
    "biodiversity",
    "restoration",
    "agriculture_cropland",
    "urbanisation",
    "fire_disturbance",
    "water_resources",
    "climate",
    "protected_areas",
    "general_knowledge",
    "other",
]

QUERY_TYPE_VALUES = [
    "spatial_analysis",
    "trend_analysis",
    "comparison",
    "data_lookup",
    "explanation",
    "metadata",
    "other",
]

COMPLEXITY_VALUES = [
    "simple",
    "moderate",
    "complex",
]

OUTCOME_VALUES = [
    "success",
    "partial",
    "failure",
    "unclear",
]

FAILURE_MODE_VALUES = [
    "missing_dataset",
    "wrong_interpretation",
    "capability_gap",
    "api_error",
    "ambiguous_query",
    "geographic_gap",
]


# ---------------------------------------------------------------------------
# Core enrichment model
# ---------------------------------------------------------------------------


class TraceEnrichment(BaseModel):
    """Structured metadata extracted from a single GNW trace by Gemini."""

    topic: str = Field(
        description=(
            "Primary topic of the user query. One of: "
            + ", ".join(TOPIC_VALUES)
        ),
    )

    dataset_required: Optional[str] = Field(
        default=None,
        description=(
            "The GNW dataset needed to answer this query, if identifiable. "
            "e.g. 'land_cover', 'tree_cover_loss', 'fire_alerts', "
            "'crop_extent', 'protected_areas', 'elevation', or null if unclear."
        ),
    )

    query_type: str = Field(
        description=(
            "What kind of request this is. One of: "
            + ", ".join(QUERY_TYPE_VALUES)
        ),
    )

    geographic_scope: Optional[str] = Field(
        default=None,
        description=(
            "Geographic entity mentioned: country name, region, "
            "or 'custom_aoi' for drawn polygons. Null if none mentioned."
        ),
    )

    intent: str = Field(
        description=(
            "The user's job-to-be-done as a short phrase. "
            "e.g. 'monitor deforestation in my region', "
            "'understand land use trends for a report', "
            "'compare restoration progress across countries'."
        ),
    )

    complexity: str = Field(
        description=(
            "How complex the request is. One of: "
            + ", ".join(COMPLEXITY_VALUES)
        ),
    )

    outcome: str = Field(
        description=(
            "How well the assistant handled the query. One of: "
            + ", ".join(OUTCOME_VALUES)
        ),
    )

    failure_mode: Optional[str] = Field(
        default=None,
        description=(
            "If outcome is 'partial' or 'failure', why. One of: "
            + ", ".join(FAILURE_MODE_VALUES)
            + ". Null if outcome is 'success' or 'unclear'."
        ),
    )


class TraceEnrichmentWithId(TraceEnrichment):
    """Enrichment result with trace_id for batch responses."""

    trace_id: str = Field(description="The trace_id from the input.")


class BatchEnrichmentResponse(BaseModel):
    """Response schema for batch enrichment — array of enrichments keyed by trace_id."""

    items: list[TraceEnrichmentWithId]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def enrichment_json_schema() -> dict:
    """Return the JSON schema dict for TraceEnrichment.

    Suitable for passing to ``google.genai.types.GenerateContentConfig(response_schema=...)``.
    """
    return TraceEnrichment.model_json_schema()


def batch_enrichment_json_schema() -> dict:
    """Return the JSON schema dict for BatchEnrichmentResponse."""
    return BatchEnrichmentResponse.model_json_schema()


def validate_enrichment(data: dict) -> TraceEnrichment | None:
    """Validate a dict against the TraceEnrichment schema.

    Returns a TraceEnrichment instance on success, None on validation failure.
    """
    try:
        return TraceEnrichment.model_validate(data)
    except Exception:
        return None


def validate_batch_enrichment(data: dict) -> BatchEnrichmentResponse | None:
    """Validate a dict against the BatchEnrichmentResponse schema.

    Returns a BatchEnrichmentResponse instance on success, None on validation failure.
    """
    try:
        return BatchEnrichmentResponse.model_validate(data)
    except Exception:
        return None


def coerce_enrichment(data: dict) -> dict:
    """Best-effort coercion of a raw dict into valid enrichment fields.

    Normalises known field values to their canonical forms (lowercase, strip).
    Unknown values are replaced with fallbacks rather than rejected.
    Returns a plain dict (not a Pydantic model) — always succeeds.
    """
    out: dict = {}

    # topic
    raw_topic = str(data.get("topic") or "other").strip().lower().replace(" ", "_")
    out["topic"] = raw_topic if raw_topic in TOPIC_VALUES else "other"

    # dataset_required
    raw_ds = data.get("dataset_required")
    if raw_ds and str(raw_ds).strip().lower() not in ("null", "none", "n/a", ""):
        out["dataset_required"] = str(raw_ds).strip().lower().replace(" ", "_")
    else:
        out["dataset_required"] = None

    # query_type
    raw_qt = str(data.get("query_type") or "other").strip().lower().replace(" ", "_")
    out["query_type"] = raw_qt if raw_qt in QUERY_TYPE_VALUES else "other"

    # geographic_scope
    raw_gs = data.get("geographic_scope")
    if raw_gs and str(raw_gs).strip().lower() not in ("null", "none", "n/a", ""):
        out["geographic_scope"] = str(raw_gs).strip()
    else:
        out["geographic_scope"] = None

    # intent
    raw_intent = str(data.get("intent") or "").strip()
    out["intent"] = raw_intent if raw_intent else "unknown"

    # complexity
    raw_cx = str(data.get("complexity") or "moderate").strip().lower()
    out["complexity"] = raw_cx if raw_cx in COMPLEXITY_VALUES else "moderate"

    # outcome
    raw_oc = str(data.get("outcome") or "unclear").strip().lower()
    out["outcome"] = raw_oc if raw_oc in OUTCOME_VALUES else "unclear"

    # failure_mode
    raw_fm = data.get("failure_mode")
    if raw_fm and str(raw_fm).strip().lower() not in ("null", "none", "n/a", ""):
        fm = str(raw_fm).strip().lower().replace(" ", "_")
        out["failure_mode"] = fm if fm in FAILURE_MODE_VALUES else None
    else:
        out["failure_mode"] = None

    return out


# ---------------------------------------------------------------------------
# Welcome / starter prompt detection (deterministic, not LLM)
# ---------------------------------------------------------------------------

_STARTER_PROMPTS_NORMALIZED: set[str] | None = None


def _normalize_for_match(s: Any) -> str:
    """Normalize a string for fuzzy matching: lowercase, collapse whitespace, strip trailing dots."""
    if not isinstance(s, str):
        return ""
    out = " ".join(s.strip().split()).lower()
    while out.endswith("."):
        out = out[:-1].rstrip()
    return out


def load_starter_prompts(path: str | Path | None = None) -> set[str]:
    """Load and normalize starter prompts from starter-prompts.json.

    Returns a set of normalized prompt strings for matching.
    Caches the result after first successful load.
    """
    global _STARTER_PROMPTS_NORMALIZED
    if _STARTER_PROMPTS_NORMALIZED is not None and path is None:
        return _STARTER_PROMPTS_NORMALIZED

    if path is None:
        # Default: repo root / starter-prompts.json
        path = Path(__file__).resolve().parent.parent / "starter-prompts.json"
    else:
        path = Path(path)

    prompts: set[str] = set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("prompts"), list):
            for p in raw["prompts"]:
                if isinstance(p, str) and p.strip():
                    prompts.add(_normalize_for_match(p))
    except Exception:
        pass

    if path is None or str(path).endswith("starter-prompts.json"):
        _STARTER_PROMPTS_NORMALIZED = prompts
    return prompts


def is_welcome_prompt(prompt: str, starter_set: set[str] | None = None) -> bool:
    """Check if a prompt is a verbatim (or near-verbatim) match to a starter prompt.

    Uses normalized string comparison: lowercase, collapsed whitespace,
    stripped trailing dots. This catches copy-paste with minor formatting
    differences.

    Args:
        prompt: The user prompt text to check.
        starter_set: Pre-loaded set of normalized starter prompts.
            If None, loads from starter-prompts.json on first call.

    Returns:
        True if the prompt matches a starter prompt.
    """
    if starter_set is None:
        starter_set = load_starter_prompts()
    if not starter_set:
        return False
    return _normalize_for_match(prompt) in starter_set


# ---------------------------------------------------------------------------
# Dataset catalog (from analytics_datasets.yml)
# ---------------------------------------------------------------------------

_DATASET_CATALOG: list[dict[str, Any]] | None = None


def load_dataset_catalog(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the GNW dataset catalog from analytics_datasets.yml.

    Each entry is a dict with at least: ``dataset_id``, ``dataset_name``,
    ``keywords``, ``analytics_api_endpoint``, ``content_date``,
    ``start_date``, ``end_date``, ``description``.

    Returns an empty list if the file is missing or unparseable.
    Caches the result after first successful load from the default path.
    """
    global _DATASET_CATALOG
    if _DATASET_CATALOG is not None and path is None:
        return _DATASET_CATALOG

    if path is None:
        path = Path(__file__).resolve().parent / "fixtures" / "analytics_datasets.yml"
    else:
        path = Path(path)

    datasets: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
        if _yaml is not None:
            raw = _yaml.safe_load(text)
        else:
            # Fallback: no pyyaml installed — can't parse
            return []
        if isinstance(raw, dict) and isinstance(raw.get("datasets"), list):
            for d in raw["datasets"]:
                if isinstance(d, dict) and d.get("dataset_name"):
                    datasets.append(d)
    except Exception:
        pass

    if path is None or str(path).endswith("analytics_datasets.yml"):
        _DATASET_CATALOG = datasets
    return datasets


def dataset_catalog_keywords() -> dict[str, list[str]]:
    """Return a mapping of dataset_name → keywords for matching.

    Useful for comparing enriched ``dataset_required`` fields against
    the known catalog.
    """
    catalog = load_dataset_catalog()
    out: dict[str, list[str]] = {}
    for d in catalog:
        name = str(d.get("dataset_name") or "").strip()
        kw = d.get("keywords") or []
        if name and isinstance(kw, list):
            out[name] = [str(k).strip().lower() for k in kw if isinstance(k, str) and k.strip()]
    return out


def dataset_catalog_summary() -> list[dict[str, Any]]:
    """Return a compact summary of the dataset catalog for inclusion in prompts.

    Each entry has: dataset_name, keywords, content_date, analytics_api_endpoint.
    """
    catalog = load_dataset_catalog()
    summaries: list[dict[str, Any]] = []
    for d in catalog:
        summaries.append({
            "dataset_id": d.get("dataset_id"),
            "dataset_name": d.get("dataset_name", ""),
            "keywords": d.get("keywords", []),
            "content_date": d.get("content_date", ""),
            "start_date": d.get("start_date", ""),
            "end_date": d.get("end_date", ""),
            "analytics_api_endpoint": d.get("analytics_api_endpoint", ""),
        })
    return summaries
