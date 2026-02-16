from __future__ import annotations

import json
from typing import Any

import streamlit as st

from utils.data_helpers import strip_code_fences


def get_gemini_model_options(api_key: str, cache_key: str = "gemini_model_options") -> list[str]:
    cached = st.session_state.get(cache_key)
    if isinstance(cached, list) and all(isinstance(x, str) for x in cached) and len(cached):
        return list(cached)

    fallback = [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    def _parse_version(name: str) -> tuple[int, int, int]:
        if not isinstance(name, str) or not name.startswith("gemini-"):
            return (0, 0, 0)
        rest = name[len("gemini-") :]
        ver = rest.split("-", 1)[0]
        parts = ver.split(".")
        out: list[int] = []
        for p in parts[:3]:
            try:
                out.append(int(p))
            except Exception:
                out.append(0)
        while len(out) < 3:
            out.append(0)
        return (out[0], out[1], out[2])

    try:
        models: list[str] = []

        from google import genai as _genai  # google-genai

        client = _genai.Client(api_key=api_key)
        for m in client.models.list():
            name = getattr(m, "name", None)
            if not isinstance(name, str) or not name.strip():
                continue
            cleaned = name.replace("models/", "")
            if not cleaned.startswith("gemini-"):
                continue
            if "image" in cleaned.lower():
                continue
            models.append(cleaned)

        models = sorted(set(models), key=lambda n: _parse_version(n), reverse=True)
        st.session_state[cache_key] = models if models else fallback
        return list(st.session_state[cache_key])
    except Exception:
        st.session_state[cache_key] = fallback
        return fallback


def chunked(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 1:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def truncate_text(txt: str, max_chars: int) -> str:
    if max_chars <= 0:
        return txt
    if len(txt) <= max_chars:
        return txt
    if max_chars <= 3:
        return txt[:max_chars]
    return txt[: max_chars - 3] + "..."


def parse_json_any(txt: str) -> Any:
    cleaned = strip_code_fences(str(txt or "")).strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def parse_json_dict(txt: str) -> dict[str, Any]:
    cleaned = strip_code_fences(str(txt or "")).strip()
    if not cleaned:
        return {}
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
    except Exception:
        return {"raw": cleaned}


def call_gemini(api_key: str, model_name: str, prompt: str) -> str:
    """Call Gemini API and return text response.
    
    Args:
        api_key: Google API key for Gemini
        model_name: Name of the Gemini model to use
        prompt: The prompt to send to the model
        
    Returns:
        The text response from the model, or empty string on error
    """
    try:
        from google import genai as _genai  # google-genai

        client = _genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return str(getattr(resp, "text", "") or "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Model-aware batch sizing
# ---------------------------------------------------------------------------

# Models with large output token limits (65K+) can handle bigger batches.
# Models with 8K output limits need smaller batches.
_LARGE_OUTPUT_MODELS = {"gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"}
_DEFAULT_BATCH_SIZE_LARGE = 50  # for 65K output models
_DEFAULT_BATCH_SIZE_SMALL = 25  # for 8K output models


def model_aware_batch_size(model_name: str, user_batch_size: int | None = None) -> int:
    """Return an appropriate batch size for the given model.

    If *user_batch_size* is provided and positive, it is used as an upper bound
    but capped to the model's safe maximum.
    """
    # Determine model tier
    is_large = any(model_name.startswith(prefix) for prefix in _LARGE_OUTPUT_MODELS)
    safe_max = _DEFAULT_BATCH_SIZE_LARGE if is_large else _DEFAULT_BATCH_SIZE_SMALL

    if user_batch_size is not None and user_batch_size > 0:
        return min(int(user_batch_size), safe_max)
    return safe_max


# ---------------------------------------------------------------------------
# Structured output (JSON mode with schema)
# ---------------------------------------------------------------------------


def call_gemini_structured(
    api_key: str,
    model_name: str,
    prompt: str,
    *,
    response_schema: Any = None,
    system_instruction: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any] | list[Any] | None:
    """Call Gemini with JSON mode and optional response schema.

    Uses ``GenerateContentConfig`` with ``response_mime_type="application/json"``
    so the model is forced to return valid JSON.  When *response_schema* is
    provided (a Pydantic model class, dict, or ``google.genai.types.Schema``),
    the output is further constrained to match that schema.

    Args:
        api_key: Google API key.
        model_name: Gemini model name.
        prompt: User prompt text.
        response_schema: Optional schema for structured output.
        system_instruction: Optional system instruction prepended to the request.
        temperature: Optional sampling temperature.

    Returns:
        Parsed JSON (dict or list) on success, ``None`` on error.
    """
    try:
        from google import genai as _genai
        from google.genai import types as _types

        client = _genai.Client(api_key=api_key)

        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
        }
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        config = _types.GenerateContentConfig(**config_kwargs)

        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        raw = str(getattr(resp, "text", "") or "").strip()
        if not raw:
            return None

        return json.loads(raw)
    except json.JSONDecodeError:
        # Model returned non-JSON despite json mode — try strip_code_fences
        try:
            cleaned = strip_code_fences(raw).strip()
            return json.loads(cleaned) if cleaned else None
        except Exception:
            return None
    except Exception:
        return None
