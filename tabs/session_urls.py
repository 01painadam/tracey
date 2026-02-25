"""Session URLs tab."""

from typing import Any

import pandas as pd
import streamlit as st

from utils import (
    csv_bytes_any,
    normalize_trace_format,
    parse_trace_dt,
    first_human_prompt,
)

from utils.content_kpis import compute_thread_key


def render(
    base_thread_url: str,
) -> None:
    """Render the Session URLs tab."""
    st.subheader("🔗 Conversation Browser")
    st.caption(
        "Generate a deduplicated list of conversation-thread links from the currently loaded traces, then export them to CSV "
        "for review or sharing."
    )

    traces: list[dict[str, Any]] = st.session_state.get("stats_traces", [])

    if not traces:
        st.info(
            "This tab turns the currently loaded traces into a list of **unique session links** you can click to open the "
            "GNW Threads UI for each conversation.\n\n"
            "Use the sidebar **🚀 Fetch traces** button first, then come back here to export session URLs."
        )
        return

    normed = [normalize_trace_format(t) for t in traces]

    rows: list[dict[str, Any]] = []
    for n in normed:
        sid = n.get("sessionId") or n.get("session_id")
        trace_id = n.get("id") or n.get("trace_id") or n.get("traceId")
        url = f"{base_thread_url.rstrip('/')}/{sid}" if sid else ""

        dt = parse_trace_dt(n)
        prompt = first_human_prompt(n)
        prompt_one_line = " ".join(str(prompt or "").split())
        prompt_snippet = prompt_one_line[:120]
        if len(prompt_one_line) > len(prompt_snippet):
            prompt_snippet = f"{prompt_snippet}…"

        rows.append(
            {
                "timestamp": dt,
                "prompt_snippet": prompt_snippet,
                "session_id": sid,
                "trace_id": trace_id,
                "url": url,
            }
        )

    if not rows:
        st.warning("No sessions found in the fetched traces.")
        return

    df = pd.DataFrame(rows)
    df["thread_key"] = compute_thread_key(df)
    df = df.sort_values("timestamp", na_position="last")
    df = df.drop_duplicates(subset=["thread_key"], keep="first").reset_index(drop=True)

    st.write(f"**{len(df)}** unique conversation threads")

    df["link"] = df["url"].apply(
        lambda u: f'<a href="{u}" target="_blank">Open</a>' if isinstance(u, str) and u else ""
    )

    st.markdown(
        df[["timestamp", "prompt_snippet", "link"]].to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

    csv_data = csv_bytes_any(df.drop(columns=["link"]).to_dict("records"))

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="gnw_session_urls.csv",
        mime="text/csv",
        key="session_urls_csv",
    )
