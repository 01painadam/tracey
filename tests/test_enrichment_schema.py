"""Tests for utils.enrichment_schema — Pydantic models, validation, coercion."""

from __future__ import annotations

import pytest

from utils.enrichment_schema import (
    COMPLEXITY_VALUES,
    FAILURE_MODE_VALUES,
    OUTCOME_VALUES,
    QUERY_TYPE_VALUES,
    TOPIC_VALUES,
    BatchEnrichmentResponse,
    TraceEnrichment,
    TraceEnrichmentWithId,
    batch_enrichment_json_schema,
    coerce_enrichment,
    dataset_catalog_keywords,
    dataset_catalog_summary,
    enrichment_json_schema,
    is_welcome_prompt,
    load_dataset_catalog,
    load_starter_prompts,
    validate_batch_enrichment,
    validate_enrichment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _valid_enrichment() -> dict:
    return {
        "topic": "deforestation",
        "dataset_required": "tree_cover_loss",
        "query_type": "trend_analysis",
        "geographic_scope": "Kenya",
        "intent": "monitor deforestation trends for annual report",
        "complexity": "moderate",
        "outcome": "success",
        "failure_mode": None,
    }


def _valid_enrichment_failure() -> dict:
    return {
        "topic": "fire_disturbance",
        "dataset_required": "fire_alerts",
        "query_type": "spatial_analysis",
        "geographic_scope": "California",
        "intent": "map wildfire damage in my region",
        "complexity": "complex",
        "outcome": "failure",
        "failure_mode": "missing_dataset",
    }


# ---------------------------------------------------------------------------
# TraceEnrichment — valid inputs
# ---------------------------------------------------------------------------

class TestTraceEnrichmentValid:
    def test_accepts_full_valid_dict(self):
        data = _valid_enrichment()
        obj = TraceEnrichment.model_validate(data)
        assert obj.topic == "deforestation"
        assert obj.dataset_required == "tree_cover_loss"
        assert obj.query_type == "trend_analysis"
        assert obj.geographic_scope == "Kenya"
        assert obj.intent == "monitor deforestation trends for annual report"
        assert obj.complexity == "moderate"
        assert obj.outcome == "success"
        assert obj.failure_mode is None

    def test_accepts_failure_with_failure_mode(self):
        data = _valid_enrichment_failure()
        obj = TraceEnrichment.model_validate(data)
        assert obj.outcome == "failure"
        assert obj.failure_mode == "missing_dataset"

    def test_accepts_none_optional_fields(self):
        data = _valid_enrichment()
        data["dataset_required"] = None
        data["geographic_scope"] = None
        data["failure_mode"] = None
        obj = TraceEnrichment.model_validate(data)
        assert obj.dataset_required is None
        assert obj.geographic_scope is None
        assert obj.failure_mode is None

    def test_accepts_all_topic_values(self):
        base = _valid_enrichment()
        for topic in TOPIC_VALUES:
            base["topic"] = topic
            obj = TraceEnrichment.model_validate(base)
            assert obj.topic == topic

    def test_accepts_all_query_type_values(self):
        base = _valid_enrichment()
        for qt in QUERY_TYPE_VALUES:
            base["query_type"] = qt
            obj = TraceEnrichment.model_validate(base)
            assert obj.query_type == qt

    def test_accepts_all_complexity_values(self):
        base = _valid_enrichment()
        for cx in COMPLEXITY_VALUES:
            base["complexity"] = cx
            obj = TraceEnrichment.model_validate(base)
            assert obj.complexity == cx

    def test_accepts_all_outcome_values(self):
        base = _valid_enrichment()
        for oc in OUTCOME_VALUES:
            base["outcome"] = oc
            obj = TraceEnrichment.model_validate(base)
            assert obj.outcome == oc

    def test_accepts_all_failure_mode_values(self):
        base = _valid_enrichment_failure()
        for fm in FAILURE_MODE_VALUES:
            base["failure_mode"] = fm
            obj = TraceEnrichment.model_validate(base)
            assert obj.failure_mode == fm


# ---------------------------------------------------------------------------
# TraceEnrichment — Pydantic allows arbitrary strings (no enum restriction)
# but coerce_enrichment handles normalization
# ---------------------------------------------------------------------------

class TestTraceEnrichmentEdgeCases:
    def test_pydantic_accepts_unknown_topic(self):
        """Pydantic model uses str, not Literal — unknown values pass validation.
        Normalization is handled by coerce_enrichment."""
        data = _valid_enrichment()
        data["topic"] = "unknown_topic_xyz"
        obj = TraceEnrichment.model_validate(data)
        assert obj.topic == "unknown_topic_xyz"

    def test_pydantic_accepts_empty_intent(self):
        data = _valid_enrichment()
        data["intent"] = ""
        obj = TraceEnrichment.model_validate(data)
        assert obj.intent == ""

    def test_rejects_missing_required_field(self):
        data = _valid_enrichment()
        del data["topic"]
        with pytest.raises(Exception):
            TraceEnrichment.model_validate(data)

    def test_rejects_wrong_type_for_required(self):
        data = _valid_enrichment()
        data["topic"] = 12345
        # Pydantic v2 strict: int is not coerced to str
        with pytest.raises(Exception):
            TraceEnrichment.model_validate(data)


# ---------------------------------------------------------------------------
# TraceEnrichmentWithId
# ---------------------------------------------------------------------------

class TestTraceEnrichmentWithId:
    def test_requires_trace_id(self):
        data = _valid_enrichment()
        # Missing trace_id
        with pytest.raises(Exception):
            TraceEnrichmentWithId.model_validate(data)

    def test_accepts_valid_with_trace_id(self):
        data = {**_valid_enrichment(), "trace_id": "abc123"}
        obj = TraceEnrichmentWithId.model_validate(data)
        assert obj.trace_id == "abc123"
        assert obj.topic == "deforestation"


# ---------------------------------------------------------------------------
# BatchEnrichmentResponse
# ---------------------------------------------------------------------------

class TestBatchEnrichmentResponse:
    def test_valid_batch(self):
        items = [
            {**_valid_enrichment(), "trace_id": "t1"},
            {**_valid_enrichment_failure(), "trace_id": "t2"},
        ]
        obj = BatchEnrichmentResponse.model_validate({"items": items})
        assert len(obj.items) == 2
        assert obj.items[0].trace_id == "t1"
        assert obj.items[1].trace_id == "t2"

    def test_empty_items(self):
        obj = BatchEnrichmentResponse.model_validate({"items": []})
        assert len(obj.items) == 0

    def test_rejects_missing_items(self):
        with pytest.raises(Exception):
            BatchEnrichmentResponse.model_validate({})

    def test_rejects_item_without_trace_id(self):
        items = [_valid_enrichment()]  # no trace_id
        with pytest.raises(Exception):
            BatchEnrichmentResponse.model_validate({"items": items})


# ---------------------------------------------------------------------------
# validate_enrichment
# ---------------------------------------------------------------------------

class TestValidateEnrichment:
    def test_returns_model_on_valid(self):
        result = validate_enrichment(_valid_enrichment())
        assert isinstance(result, TraceEnrichment)
        assert result.topic == "deforestation"

    def test_returns_none_on_missing_field(self):
        data = _valid_enrichment()
        del data["outcome"]
        assert validate_enrichment(data) is None

    def test_returns_none_on_garbage(self):
        assert validate_enrichment({"garbage": True}) is None

    def test_returns_none_on_empty(self):
        assert validate_enrichment({}) is None


# ---------------------------------------------------------------------------
# validate_batch_enrichment
# ---------------------------------------------------------------------------

class TestValidateBatchEnrichment:
    def test_returns_model_on_valid(self):
        data = {"items": [{**_valid_enrichment(), "trace_id": "t1"}]}
        result = validate_batch_enrichment(data)
        assert isinstance(result, BatchEnrichmentResponse)
        assert len(result.items) == 1

    def test_returns_none_on_invalid(self):
        assert validate_batch_enrichment({"items": [{"bad": True}]}) is None

    def test_returns_none_on_empty_dict(self):
        assert validate_batch_enrichment({}) is None


# ---------------------------------------------------------------------------
# JSON schema helpers
# ---------------------------------------------------------------------------

class TestJsonSchemas:
    def test_enrichment_json_schema_is_dict(self):
        schema = enrichment_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "topic" in schema["properties"]
        assert "outcome" in schema["properties"]
        assert "failure_mode" in schema["properties"]

    def test_batch_enrichment_json_schema_is_dict(self):
        schema = batch_enrichment_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "items" in schema["properties"]

    def test_enrichment_schema_has_all_fields(self):
        schema = enrichment_json_schema()
        expected_fields = {
            "topic", "dataset_required", "query_type", "geographic_scope",
            "intent", "complexity", "outcome", "failure_mode",
        }
        assert expected_fields == set(schema["properties"].keys())

    def test_schema_roundtrips_through_model(self):
        """Verify schema can validate a dict that was produced from a model."""
        original = _valid_enrichment()
        model = TraceEnrichment.model_validate(original)
        dumped = model.model_dump()
        restored = TraceEnrichment.model_validate(dumped)
        assert restored == model


# ---------------------------------------------------------------------------
# coerce_enrichment
# ---------------------------------------------------------------------------

class TestCoerceEnrichment:
    def test_valid_data_passes_through(self):
        data = _valid_enrichment()
        result = coerce_enrichment(data)
        assert result["topic"] == "deforestation"
        assert result["query_type"] == "trend_analysis"
        assert result["outcome"] == "success"
        assert result["failure_mode"] is None

    def test_unknown_topic_falls_back_to_other(self):
        data = _valid_enrichment()
        data["topic"] = "weird_unknown_topic"
        result = coerce_enrichment(data)
        assert result["topic"] == "other"

    def test_unknown_query_type_falls_back_to_other(self):
        data = _valid_enrichment()
        data["query_type"] = "banana"
        result = coerce_enrichment(data)
        assert result["query_type"] == "other"

    def test_unknown_complexity_falls_back_to_moderate(self):
        data = _valid_enrichment()
        data["complexity"] = "extreme"
        result = coerce_enrichment(data)
        assert result["complexity"] == "moderate"

    def test_unknown_outcome_falls_back_to_unclear(self):
        data = _valid_enrichment()
        data["outcome"] = "kinda_worked"
        result = coerce_enrichment(data)
        assert result["outcome"] == "unclear"

    def test_unknown_failure_mode_falls_back_to_none(self):
        data = _valid_enrichment_failure()
        data["failure_mode"] = "gremlins"
        result = coerce_enrichment(data)
        assert result["failure_mode"] is None

    def test_null_string_dataset_becomes_none(self):
        data = _valid_enrichment()
        data["dataset_required"] = "null"
        result = coerce_enrichment(data)
        assert result["dataset_required"] is None

    def test_none_string_dataset_becomes_none(self):
        data = _valid_enrichment()
        data["dataset_required"] = "None"
        result = coerce_enrichment(data)
        assert result["dataset_required"] is None

    def test_na_string_geographic_scope_becomes_none(self):
        data = _valid_enrichment()
        data["geographic_scope"] = "N/A"
        result = coerce_enrichment(data)
        assert result["geographic_scope"] is None

    def test_empty_intent_becomes_unknown(self):
        data = _valid_enrichment()
        data["intent"] = ""
        result = coerce_enrichment(data)
        assert result["intent"] == "unknown"

    def test_none_intent_becomes_unknown(self):
        data = _valid_enrichment()
        data["intent"] = None
        result = coerce_enrichment(data)
        assert result["intent"] == "unknown"

    def test_completely_empty_dict(self):
        result = coerce_enrichment({})
        assert result["topic"] == "other"
        assert result["dataset_required"] is None
        assert result["query_type"] == "other"
        assert result["geographic_scope"] is None
        assert result["intent"] == "unknown"
        assert result["complexity"] == "moderate"
        assert result["outcome"] == "unclear"
        assert result["failure_mode"] is None

    def test_whitespace_handling(self):
        data = _valid_enrichment()
        data["topic"] = "  Deforestation  "
        data["query_type"] = " Trend Analysis "
        result = coerce_enrichment(data)
        assert result["topic"] == "deforestation"
        assert result["query_type"] == "trend_analysis"

    def test_dataset_required_normalizes_spaces(self):
        data = _valid_enrichment()
        data["dataset_required"] = "Tree Cover Loss"
        result = coerce_enrichment(data)
        assert result["dataset_required"] == "tree_cover_loss"

    def test_geographic_scope_preserves_casing(self):
        data = _valid_enrichment()
        data["geographic_scope"] = "  Democratic Republic of Congo  "
        result = coerce_enrichment(data)
        assert result["geographic_scope"] == "Democratic Republic of Congo"

    def test_always_returns_dict(self):
        """coerce_enrichment never raises — always returns a dict."""
        for garbage in [
            {},
            {"topic": None, "outcome": None},
            {"topic": 123, "query_type": [], "intent": False},
        ]:
            result = coerce_enrichment(garbage)
            assert isinstance(result, dict)
            assert "topic" in result
            assert "outcome" in result


# ---------------------------------------------------------------------------
# Welcome / starter prompt detection
# ---------------------------------------------------------------------------

class TestLoadStarterPrompts:
    def test_loads_from_repo_root(self):
        """Loads starter-prompts.json from the default path."""
        prompts = load_starter_prompts()
        assert isinstance(prompts, set)
        assert len(prompts) == 11  # 11 starter prompts in the fixture

    def test_loads_from_explicit_path(self, tmp_path):
        fixture = tmp_path / "starters.json"
        fixture.write_text('{"prompts": ["Hello world.", "Test prompt"]}')
        prompts = load_starter_prompts(fixture)
        assert len(prompts) == 2
        assert "hello world" in prompts  # normalized: lowercase, dot stripped

    def test_returns_empty_set_on_missing_file(self, tmp_path):
        prompts = load_starter_prompts(tmp_path / "nonexistent.json")
        assert prompts == set()

    def test_returns_empty_set_on_bad_json(self, tmp_path):
        fixture = tmp_path / "bad.json"
        fixture.write_text("not json")
        prompts = load_starter_prompts(fixture)
        assert prompts == set()

    def test_returns_empty_set_on_wrong_structure(self, tmp_path):
        fixture = tmp_path / "wrong.json"
        fixture.write_text('{"other_key": [1, 2]}')
        prompts = load_starter_prompts(fixture)
        assert prompts == set()


class TestIsWelcomePrompt:
    def test_exact_match(self):
        starter_set = load_starter_prompts()
        assert is_welcome_prompt(
            "Show changes in grassland extent in Montana.",
            starter_set,
        )

    def test_match_without_trailing_dot(self):
        starter_set = load_starter_prompts()
        assert is_welcome_prompt(
            "Show changes in grassland extent in Montana",
            starter_set,
        )

    def test_match_with_different_casing(self):
        starter_set = load_starter_prompts()
        assert is_welcome_prompt(
            "SHOW CHANGES IN GRASSLAND EXTENT IN MONTANA.",
            starter_set,
        )

    def test_match_with_extra_whitespace(self):
        starter_set = load_starter_prompts()
        assert is_welcome_prompt(
            "  Show  changes  in  grassland  extent  in  Montana.  ",
            starter_set,
        )

    def test_no_match_for_organic_prompt(self):
        starter_set = load_starter_prompts()
        assert not is_welcome_prompt(
            "What is the deforestation rate in the Amazon?",
            starter_set,
        )

    def test_no_match_for_modified_starter(self):
        """A starter prompt with added words should NOT match (strict matching)."""
        starter_set = load_starter_prompts()
        assert not is_welcome_prompt(
            "Please show changes in grassland extent in Montana for 2020.",
            starter_set,
        )

    def test_no_match_for_empty_string(self):
        starter_set = load_starter_prompts()
        assert not is_welcome_prompt("", starter_set)

    def test_no_match_for_none(self):
        starter_set = load_starter_prompts()
        assert not is_welcome_prompt(None, starter_set)  # type: ignore[arg-type]

    def test_works_with_custom_starter_set(self):
        custom = {"hello world", "test prompt"}
        assert is_welcome_prompt("Hello World.", custom)
        assert is_welcome_prompt("test prompt", custom)
        assert not is_welcome_prompt("something else", custom)

    def test_auto_loads_when_no_set_provided(self):
        """When starter_set is None, auto-loads from starter-prompts.json."""
        # This tests the auto-load path
        result = is_welcome_prompt("Show changes in grassland extent in Montana.")
        assert result is True

    def test_all_11_starter_prompts_match(self):
        """Every prompt in starter-prompts.json should be detected."""
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "starter-prompts.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        prompts = raw["prompts"]
        assert len(prompts) == 11

        starter_set = load_starter_prompts()
        for p in prompts:
            assert is_welcome_prompt(p, starter_set), f"Failed to match: {p}"


# ---------------------------------------------------------------------------
# Dataset catalog
# ---------------------------------------------------------------------------

class TestLoadDatasetCatalog:
    def test_loads_from_default_path(self):
        catalog = load_dataset_catalog()
        assert isinstance(catalog, list)
        assert len(catalog) == 10  # 10 datasets in the fixture

    def test_each_entry_has_required_fields(self):
        catalog = load_dataset_catalog()
        for d in catalog:
            assert "dataset_name" in d
            assert "keywords" in d
            assert "analytics_api_endpoint" in d

    def test_known_datasets_present(self):
        catalog = load_dataset_catalog()
        names = [d["dataset_name"] for d in catalog]
        assert any("DIST-ALERT" in n for n in names)
        assert any("Tree cover loss" in n for n in names)
        assert any("Global land cover" in n for n in names)

    def test_returns_empty_on_missing_file(self, tmp_path):
        result = load_dataset_catalog(tmp_path / "nonexistent.yml")
        assert result == []

    def test_returns_empty_on_bad_yaml(self, tmp_path):
        bad = tmp_path / "bad.yml"
        bad.write_text("not: [valid: yaml: {{")
        result = load_dataset_catalog(bad)
        assert result == []


class TestDatasetCatalogKeywords:
    def test_returns_dict_of_keyword_lists(self):
        kw = dataset_catalog_keywords()
        assert isinstance(kw, dict)
        assert len(kw) == 10
        for name, keywords in kw.items():
            assert isinstance(name, str)
            assert isinstance(keywords, list)
            assert all(isinstance(k, str) for k in keywords)

    def test_keywords_are_lowercase(self):
        kw = dataset_catalog_keywords()
        for keywords in kw.values():
            for k in keywords:
                assert k == k.lower()

    def test_tree_cover_loss_has_deforestation_keyword(self):
        kw = dataset_catalog_keywords()
        tcl_keywords = kw.get("Tree cover loss", [])
        assert "deforestation" in tcl_keywords


class TestDatasetCatalogSummary:
    def test_returns_list_of_compact_dicts(self):
        summaries = dataset_catalog_summary()
        assert isinstance(summaries, list)
        assert len(summaries) == 10
        for s in summaries:
            assert "dataset_name" in s
            assert "keywords" in s
            assert "content_date" in s
            assert "analytics_api_endpoint" in s

    def test_does_not_include_heavy_fields(self):
        """Summary should be compact — no methodology, description, cautions."""
        summaries = dataset_catalog_summary()
        for s in summaries:
            assert "methodology" not in s
            assert "description" not in s
            assert "cautions" not in s
