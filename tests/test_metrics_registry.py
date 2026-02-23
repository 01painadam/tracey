"""Tests for the metrics/page documentation registry.

These are governance tests: they are meant to fail fast when someone adds a KPI
or references a metric ID without documenting it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOT = Path(__file__).resolve().parents[1]
metrics_mod = _load_module("metrics_registry", ROOT / "utils" / "metrics_registry.py")
validation_mod = _load_module("metrics_validation", ROOT / "utils" / "metrics_validation.py")

METRICS = metrics_mod.METRICS
PAGES = metrics_mod.PAGES
validate_metrics_registry = validation_mod.validate_metrics_registry


def test_metrics_registry_is_valid() -> None:
    errors = validate_metrics_registry(METRICS, PAGES)
    assert not errors, "\n".join(["Metrics registry validation errors:"] + [f"- {e}" for e in errors])
