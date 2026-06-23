"""Conversation Browser tab (reads sessions from the Zeno API)."""

from typing import Any

import pandas as pd
import streamlit as st

from utils import csv_bytes_any, zeno_api
from utils.data_helpers import format_report_date
from utils.shared_ui import _is_machine_user_id, _load_internal_user_ids


def render(
    base_thread_url: str,
    start_date=None,
    end_date=None,
    envs: list[str] | None = None,
    zeno_api_url: str = "",
    zeno_api_token: str = "",
) -> None:
    """Render the Conversation Browser tab."""
    st.subheader("🔗 Conversation Browser")
    st.caption(
        "List conversation threads (one row per session, deduped server-side by the Zeno API) "
        "with their first prompt and turn count, then export them to CSV for review or sharing."
    )

    zeno_url = (zeno_api_url or "").strip()
    zeno_token = (zeno_api_token or "").strip()

    st.caption(
        f"Reading from the **Zeno API** · environment: **{', '.join(envs) if envs else 'all'}** · "
        f"{format_report_date(start_date)} → {format_report_date(end_date)}"
    )
    c_fetch, _c_sp = st.columns([1, 3])
    with c_fetch:
        fetch_clicked = st.button(
            "🚀 Fetch sessions", type="primary", key="sessions_zeno_fetch"
        )

    if fetch_clicked:
        if not zeno_url or not zeno_token:
            st.error("Set ZENO_API_URL and ZENO_API_TOKEN in the sidebar (🔐 Credentials).")
        elif not (start_date and end_date):
            st.error("Select a date range in the sidebar first.")
        else:
            try:
                with st.spinner("Fetching sessions from the Zeno API…"):
                    sessions = zeno_api.fetch_sessions(
                        base_url=zeno_url,
                        token=zeno_token,
                        start_date=start_date,
                        end_date=end_date,
                        environment=(envs[0] if envs and len(envs) == 1 else None),
                    )
                st.session_state.zeno_sessions = sessions
                st.toast(f"Fetched {len(sessions):,} sessions from the Zeno API")
            except zeno_api.ZenoAPIError as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(f"Zeno API fetch failed: {e}")

    sessions: list[dict[str, Any]] = list(st.session_state.get("zeno_sessions") or [])
    if not sessions:
        st.info(
            "This tab turns Zeno-API sessions into a list of **unique conversation links** you can "
            "click to open the GNW Threads UI.\n\nSet the date range / environment in the sidebar, "
            "then click **🚀 Fetch sessions** above."
        )
        return

    # Client-side exclusion of machine + internal users (parity with analytics).
    internal_user_ids = _load_internal_user_ids()
    exclude_internal = bool(st.session_state.get("_shadow_exclude_internal", True))

    def _keep(s: dict[str, Any]) -> bool:
        uid = str(s.get("user_id") or "").strip()
        if _is_machine_user_id(uid):
            return False
        if exclude_internal and internal_user_ids and uid in internal_user_ids:
            return False
        return True

    sessions = [s for s in sessions if _keep(s)]
    if envs:
        sessions = [s for s in sessions if (s.get("environment") in envs)]

    rows: list[dict[str, Any]] = []
    for s in sessions:
        sid = s.get("session_id")
        if not sid:
            continue
        url = f"{base_thread_url.rstrip('/')}/{sid}"
        prompt = " ".join(str(s.get("first_prompt") or "").split())
        prompt_snippet = prompt[:120]
        if len(prompt) > len(prompt_snippet):
            prompt_snippet = f"{prompt_snippet}…"
        rows.append(
            {
                "first_timestamp": s.get("first_timestamp"),
                "last_timestamp": s.get("last_timestamp"),
                "turn_count": s.get("turn_count"),
                "prompt_snippet": prompt_snippet,
                "url": url,
            }
        )

    if not rows:
        st.warning("No sessions found for the selected window/filters.")
        return

    st.write(f"**{len(rows)}** unique conversation threads")

    df = pd.DataFrame(rows)
    df["link"] = df["url"].apply(lambda u: f'<a href="{u}" target="_blank">Open</a>')

    st.markdown(
        df[["first_timestamp", "last_timestamp", "turn_count", "prompt_snippet", "link"]].to_html(
            escape=False, index=False
        ),
        unsafe_allow_html=True,
    )

    csv_data = csv_bytes_any(rows)

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="gnw_session_urls.csv",
        mime="text/csv",
        key="session_urls_csv",
    )
