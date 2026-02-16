"""Shared UI components for Product Intelligence modes."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_trace_evidence(
    traces: list[dict[str, Any]],
    base_thread_url: str,
    title: str = "Supporting traces",
    max_display: int = 20,
) -> None:
    """Render expandable trace evidence with links to GNW."""
    with st.expander(f"📋 {title} ({len(traces)})", expanded=False):
        for t in traces[:max_display]:
            prompt_preview = str(t.get("prompt") or "")[:80]
            with st.expander(f"{prompt_preview}..."):
                st.markdown("**Prompt:**")
                st.code(t.get("prompt", ""), language=None)
                answer = str(t.get("answer") or "")
                st.markdown("**Answer:**")
                st.code(answer[:1000] + ("..." if len(answer) > 1000 else ""), language=None)

                # Enrichment metadata if available
                meta_parts = []
                if t.get("topic"):
                    meta_parts.append(f"Topic: {t['topic']}")
                if t.get("query_type"):
                    meta_parts.append(f"Type: {t['query_type']}")
                if t.get("outcome"):
                    meta_parts.append(f"Outcome: {t['outcome']}")
                if t.get("failure_mode"):
                    meta_parts.append(f"Failure: {t['failure_mode']}")
                if meta_parts:
                    st.caption(" · ".join(meta_parts))

                sid = t.get("session_id")
                if sid:
                    url = f"{base_thread_url.rstrip('/')}/{sid}"
                    st.link_button("🔗 View in GNW", url)

        if len(traces) > max_display:
            st.caption(f"Showing {max_display} of {len(traces)} traces.")
