"""Tests for the enrichment engine — normalization, batching, delta logic."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# normalize_trace_rich
# ---------------------------------------------------------------------------

class TestNormalizeTraceRich:
    def test_extracts_prompt_and_answer(self):
        from tabs.product_intel.enrichment import normalize_trace_rich

        trace = {
            "id": "t1",
            "sessionId": "s1",
            "userId": "u1",
            "environment": "production",
            "timestamp": "2025-01-01T00:00:00Z",
            "input": {"messages": [{"type": "human", "content": "Hello world"}]},
            "output": {"messages": [{"type": "ai", "content": "Hi there"}]},
        }
        result = normalize_trace_rich(trace)
        assert result is not None
        assert result["trace_id"] == "t1"
        assert result["session_id"] == "s1"
        assert result["prompt"] == "Hello world"
        assert result["answer"] == "Hi there"
        assert "outcome_heuristic" in result
        assert "tools_used" in result
        assert "datasets_analysed" in result

    def test_returns_none_for_empty_trace(self):
        from tabs.product_intel.enrichment import normalize_trace_rich

        trace = {"id": "t2", "input": {}, "output": {}}
        result = normalize_trace_rich(trace)
        assert result is None

    def test_includes_deterministic_context(self):
        from tabs.product_intel.enrichment import normalize_trace_rich

        trace = {
            "id": "t3",
            "input": {"messages": [{"type": "human", "content": "Show forest loss"}]},
            "output": {"messages": [
                {"type": "ai", "content": "Here is the analysis", "tool_calls": []},
            ]},
        }
        result = normalize_trace_rich(trace)
        assert result is not None
        assert isinstance(result["tools_used"], list)
        assert isinstance(result["datasets_analysed"], list)
        assert isinstance(result["aoi_name"], str)


# ---------------------------------------------------------------------------
# Delta detection
# ---------------------------------------------------------------------------

class TestDeltaDetection:
    def test_needs_enrichment_for_new_trace(self):
        from tabs.product_intel.enrichment import _needs_enrichment

        row = {"trace_id": "new_trace"}
        existing = {"old_trace": {"trace_id": "old_trace", "topic": "deforestation"}}
        assert _needs_enrichment(row, existing) is True

    def test_no_enrichment_for_existing_trace(self):
        from tabs.product_intel.enrichment import _needs_enrichment

        row = {"trace_id": "old_trace"}
        existing = {"old_trace": {"trace_id": "old_trace", "topic": "deforestation"}}
        assert _needs_enrichment(row, existing) is False

    def test_trace_ids_hash_changes_with_different_traces(self):
        from tabs.product_intel.enrichment import _trace_ids_hash

        traces_a = [{"id": "t1"}, {"id": "t2"}]
        traces_b = [{"id": "t1"}, {"id": "t3"}]
        assert _trace_ids_hash(traces_a) != _trace_ids_hash(traces_b)

    def test_trace_ids_hash_stable_for_same_traces(self):
        from tabs.product_intel.enrichment import _trace_ids_hash

        traces = [{"id": "t2"}, {"id": "t1"}]  # different order
        h1 = _trace_ids_hash(traces)
        h2 = _trace_ids_hash(traces)
        assert h1 == h2  # sorted internally

    def test_trace_ids_hash_order_independent(self):
        from tabs.product_intel.enrichment import _trace_ids_hash

        traces_a = [{"id": "t1"}, {"id": "t2"}]
        traces_b = [{"id": "t2"}, {"id": "t1"}]
        assert _trace_ids_hash(traces_a) == _trace_ids_hash(traces_b)


# ---------------------------------------------------------------------------
# Batch payload building
# ---------------------------------------------------------------------------

class TestBuildBatchPayload:
    def test_builds_correct_payload(self):
        from tabs.product_intel.enrichment import _build_batch_payload

        rows = [
            {
                "trace_id": "t1",
                "prompt": "Show deforestation in Brazil",
                "answer": "Here is the data...",
                "outcome_heuristic": "ANSWER",
                "tools_used": ["pull_data", "generate_insight"],
                "datasets_analysed": ["tree_cover_loss"],
                "aoi_name": "Brazil",
                "aoi_type": "country",
            },
        ]
        payload = _build_batch_payload(rows, max_chars=4000)
        assert len(payload) == 1
        assert payload[0]["trace_id"] == "t1"
        assert "Show deforestation" in payload[0]["user_prompt"]
        assert payload[0]["outcome_heuristic"] == "ANSWER"
        assert "pull_data" in payload[0]["tools_used"]
        assert payload[0]["aoi_name"] == "Brazil"

    def test_truncates_long_text(self):
        from tabs.product_intel.enrichment import _build_batch_payload

        rows = [
            {
                "trace_id": "t1",
                "prompt": "x" * 10000,
                "answer": "y" * 10000,
                "outcome_heuristic": "ANSWER",
                "tools_used": [],
                "datasets_analysed": [],
                "aoi_name": "",
                "aoi_type": "",
            },
        ]
        payload = _build_batch_payload(rows, max_chars=100)
        assert len(payload[0]["user_prompt"]) <= 100
        assert len(payload[0]["assistant_response"]) <= 100

    def test_handles_empty_fields(self):
        from tabs.product_intel.enrichment import _build_batch_payload

        rows = [
            {
                "trace_id": "t1",
                "prompt": None,
                "answer": None,
                "outcome_heuristic": "",
                "tools_used": None,
                "datasets_analysed": None,
                "aoi_name": None,
                "aoi_type": None,
            },
        ]
        payload = _build_batch_payload(rows, max_chars=4000)
        assert payload[0]["trace_id"] == "t1"
        # Should not raise


# ---------------------------------------------------------------------------
# Model-aware batch size integration
# ---------------------------------------------------------------------------

class TestModelAwareBatchIntegration:
    def test_enrichment_uses_model_aware_sizing(self):
        from utils.llm_helpers import model_aware_batch_size

        # Flash lite should get large batches
        assert model_aware_batch_size("gemini-2.5-flash-lite") == 50
        # Older models capped
        assert model_aware_batch_size("gemini-2.0-flash") == 25


# ---------------------------------------------------------------------------
# Welcome prompt integration
# ---------------------------------------------------------------------------

class TestWelcomePromptIntegration:
    def test_welcome_prompt_detected_in_enrichment(self):
        from utils.enrichment_schema import is_welcome_prompt, load_starter_prompts

        starter_set = load_starter_prompts()
        # Known starter prompt
        assert is_welcome_prompt(
            "Show changes in grassland extent in Montana.",
            starter_set,
        )
        # Organic prompt
        assert not is_welcome_prompt(
            "What is the deforestation rate in the Amazon basin since 2010?",
            starter_set,
        )
