import os
import hmac
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
import inspect
from typing import Any

import streamlit as st

from utils import (
    as_float,
    classify_outcome,
    csv_bytes_any,
    fetch_traces_window,
    first_human_prompt,
    get_langfuse_headers,
    iso_utc,
    maybe_load_dotenv,
    normalize_trace_format,
    parse_trace_dt,
    final_ai_message,
)
from tabs import (
    render_session_urls,
    render_human_eval,
    render_product_dev,
    render_analytics,
    render_trace_explorer,
)


def main() -> None:
    st.set_page_config(page_title="GNW Langfuse Session Pull", layout="wide")

    maybe_load_dotenv()

    app_password = os.getenv("APP_PASSWORD", "")
    if "app_authenticated" not in st.session_state:
        st.session_state.app_authenticated = False

    if app_password and not st.session_state.app_authenticated:
        st.title("🔒 Tracey")
        st.caption("Enter the app password to continue.")

        pw = st.text_input("Password", type="password", key="_app_password_input", width=500)
        col_login, _ = st.columns([1, 3])
        with col_login:
            if st.button("Log in", type="primary"):
                if hmac.compare_digest(str(pw), str(app_password)):
                    st.session_state.app_authenticated = True
                    st.session_state.pop("_app_password_input", None)
                    st.rerun()
                else:
                    st.error("Incorrect password")
        return

    with st.sidebar:
        if app_password:
            if st.button("Log out", width="stretch"):
                st.session_state.app_authenticated = False
                st.rerun()

        ### title with small text to the right
        st.title("💬🧠📎 Tracey. `v0.1`")
        st.caption("Think: _Clippy_... but for GNW traces.")
        st.markdown(
            "**ℹ️ What this tool does**\n\n"
            "Tracey allows you quickly pull and explore traces from Langfuse.\n"
            "_Ta, Trace!_\n\n"
            "- **📥 Fetch** a single set of traces once\n"
            "- **📊 Explore** the same dataset across tabs\n"
            "- **📋 Generate** reports & understand user behaviour\n"
            "- **🧪 Sample** for human eval & product mining"
        )

        st.markdown("---")

        st.markdown("**🌍 Environment**")
        environment = st.selectbox(
            "",
            options=["production", "production,default", "staging", "all"],
            index=0,
            label_visibility="collapsed",
        )

        envs: list[str] | None
        if environment == "all":
            envs = None
        else:
            envs = [e.strip() for e in environment.split(",") if e.strip()]

        st.markdown("**📅 Date range**")
        date_preset = st.selectbox(
            "",
            options=["All", "Last day", "Last week", "Last month", "Custom"],
            index=1,
            label_visibility="collapsed",
        )

        default_end = date.today()
        default_start = default_end - timedelta(days=7)

        use_date_filter = date_preset != "All"
        if date_preset == "Last day":
            start_date = default_end - timedelta(days=1)
            end_date = default_end
            st.caption(f"Using {start_date} to {end_date}")
        elif date_preset == "Last week":
            start_date = default_end - timedelta(days=7)
            end_date = default_end
            st.caption(f"Using {start_date} to {end_date}")
        elif date_preset == "Last month":
            start_date = default_end - timedelta(days=30)
            end_date = default_end
            st.caption(f"Using {start_date} to {end_date}")
        elif date_preset == "Custom":
            start_date = st.date_input(
                "Start date",
                value=default_start,
            )
            end_date = st.date_input(
                "End date",
                value=default_end,
            )
        else:
            start_date = date(2025, 9, 17)  # Launch date
            end_date = default_end
            st.caption(f"Using {start_date} to {end_date}")

        if "stats_traces" not in st.session_state:
            st.session_state.stats_traces = []

        st.markdown(
            """
<style>
/* Sidebar button styling */
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
  background: #e5484d !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
  background: #d83f45 !important;
}

/* Make the raw CSV download button visually prominent when data is available */
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button {
  background: #1fb6a6 !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
}
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button:hover {
  background: #17a596 !important;
}
</style>
            """,
            unsafe_allow_html=True,
        )

        c_fetch, c_dl = st.columns(2)
        with c_fetch:
            fetch_clicked = st.button("🚀 Fetch traces", type="primary", width="stretch")
        with c_dl:
            traces_for_dl = st.session_state.get("stats_traces", [])
            if traces_for_dl:
                normed_for_dl = [normalize_trace_format(t) for t in traces_for_dl]
                out_rows = []
                for n in normed_for_dl:
                    prompt = first_human_prompt(n)
                    answer = final_ai_message(n)
                    dt = parse_trace_dt(n)
                    out_rows.append(
                        {
                            "trace_id": n.get("id"),
                            "timestamp": dt,
                            "date": dt.date() if dt else None,
                            "environment": n.get("environment"),
                            "session_id": n.get("sessionId"),
                            "user_id": n.get("userId")
                            or (n.get("metadata") or {}).get("user_id")
                            or (n.get("metadata") or {}).get("userId"),
                            "latency_seconds": as_float(n.get("latency")),
                            "total_cost": as_float(n.get("totalCost")),
                            "outcome": classify_outcome(n, answer or ""),
                            "prompt": prompt,
                            "answer": answer,
                        }
                    )

                raw_csv_bytes = csv_bytes_any(out_rows)

                st.download_button(
                    label="⬇️ Download csv",
                    data=raw_csv_bytes,
                    file_name="gnw_traces_raw.csv",
                    mime="text/csv",
                    key="raw_csv_download",
                    width="stretch",
                )
            else:
                st.button("⬇️ Download csv", disabled=True, width="stretch")

        fetch_status = st.empty()

        with st.expander("🔎 Debug Langfuse call", expanded=False):
            dbg = st.session_state.get("fetch_debug")
            if isinstance(dbg, dict) and dbg:
                st.json(dbg)
            else:
                st.caption("Fetch traces to populate request/response metadata.")

        st.markdown("---")

        with st.expander("🔐 Credentials", expanded=False):
            public_key = st.text_input(
                "LANGFUSE_PUBLIC_KEY",
                value=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            )
            secret_key = st.text_input(
                "LANGFUSE_SECRET_KEY",
                value=os.getenv("LANGFUSE_SECRET_KEY", ""),
                type="password",
            )
            base_url = st.text_input(
                "LANGFUSE_BASE_URL",
                value=os.getenv("LANGFUSE_BASE_URL", ""),
            )
            gemini_api_key = st.text_input(
                "GEMINI_API_KEY",
                value=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
                type="password",
            )

        with st.expander("⚠️ Limits", expanded=False):
            stats_page_limit = st.number_input(
                "Max pages",
                min_value=1,
                max_value=1000,
                value=500,
                key="stats_page_limit",
            )
            stats_http_timeout_s = st.number_input(
                "HTTP timeout (seconds)",
                min_value=5,
                max_value=600,
                value=60,
                key="stats_http_timeout_s",
                help="If multi-month fetches time out, increase this (e.g. 120-300s).",
            )
            stats_page_size = st.number_input(
                "Traces per page",
                min_value=1,
                max_value=100,
                value=100,
                key="stats_page_size",
                help="This is the API page size (per request). It is not the overall max; that's controlled by 'Max traces'.",
            )
            stats_max_traces = st.number_input(
                "Max traces",
                min_value=1,
                max_value=200000,
                value=25000,
                key="stats_max_traces",
            )
            stats_parallel_workers = st.number_input(
                "Parallel workers",
                min_value=1,
                max_value=5,
                value=3,
                key="stats_parallel_workers",
                help="Number of weekly chunks to fetch in parallel. Higher = faster but more server load.",
            )

            base_thread_url = f"https://www.{'staging.' if environment == 'staging' else ''}globalnaturewatch.org/app/threads"

        if fetch_clicked:
            if not public_key or not secret_key or not base_url:
                st.error("Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL")
            else:
                if start_date > end_date:
                    st.error("Start date must be on or before end date")
                else:
                    headers = get_langfuse_headers(public_key, secret_key)
                    
                    # Determine if we need to chunk (use 5-day blocks for ranges >5 days)
                    date_range_days = (end_date - start_date).days
                    chunk_by_block = date_range_days > 5
                    
                    if chunk_by_block:
                        # Split into 5-day chunks
                        chunks: list[tuple[int, date, date]] = []
                        current_start = start_date
                        idx = 1
                        while current_start <= end_date:
                            current_end = min(current_start + timedelta(days=4), end_date)
                            chunks.append((idx, current_start, current_end))
                            current_start = current_end + timedelta(days=1)
                            idx += 1
                        
                        # Detect function signature once
                        sig = None
                        try:
                            sig = inspect.signature(fetch_traces_window)
                        except Exception:
                            sig = None
                        supports_debug_out = bool(sig and "debug_out" in sig.parameters)
                        supports_page_size = bool(sig and "page_size" in sig.parameters)
                        supports_http_timeout = bool(sig and "http_timeout_s" in sig.parameters)
                        
                        def fetch_chunk(chunk_info: tuple[int, date, date]) -> tuple[int, date, date, list[dict[str, Any]], dict[str, Any]]:
                            """Fetch a single chunk - runs in thread pool."""
                            chunk_idx, chunk_start, chunk_end = chunk_info
                            from_iso = iso_utc(datetime.combine(chunk_start, time.min).replace(tzinfo=timezone.utc))
                            to_iso = iso_utc(datetime.combine(chunk_end, time.max).replace(tzinfo=timezone.utc))
                            
                            chunk_debug: dict[str, Any] = {}
                            call_kwargs: dict[str, Any] = {
                                "base_url": base_url,
                                "headers": headers,
                                "from_iso": from_iso,
                                "to_iso": to_iso,
                                "envs": envs,
                                "page_limit": int(stats_page_limit),
                                "max_traces": int(stats_max_traces),
                                "retry": 2,
                                "backoff": 0.5,
                            }
                            if supports_page_size:
                                call_kwargs["page_size"] = int(stats_page_size)
                            if supports_http_timeout:
                                call_kwargs["http_timeout_s"] = float(stats_http_timeout_s)
                            if supports_debug_out:
                                call_kwargs["debug_out"] = chunk_debug
                            
                            chunk_traces = fetch_traces_window(**call_kwargs)
                            return (chunk_idx, chunk_start, chunk_end, chunk_traces, chunk_debug)
                        
                        all_traces: list[dict[str, Any]] = []
                        seen_ids: set[str] = set()
                        fetch_debug: dict[str, Any] = {"chunks": [], "total_traces": 0, "parallel_workers": int(stats_parallel_workers)}
                        completed_count = 0
                        
                        with fetch_status.status(f"Fetching traces in {len(chunks)} 5-day chunks ({stats_parallel_workers} parallel)...", expanded=True) as status:
                            with ThreadPoolExecutor(max_workers=int(stats_parallel_workers)) as executor:
                                futures = {executor.submit(fetch_chunk, chunk): chunk for chunk in chunks}
                                
                                for future in as_completed(futures):
                                    completed_count += 1
                                    try:
                                        chunk_idx, chunk_start, chunk_end, chunk_traces, chunk_debug = future.result()
                                        
                                        # Deduplicate by trace ID
                                        new_traces = 0
                                        for trace in chunk_traces:
                                            trace_id = trace.get("id")
                                            if isinstance(trace_id, str) and trace_id not in seen_ids:
                                                seen_ids.add(trace_id)
                                                all_traces.append(trace)
                                                new_traces += 1
                                        
                                        fetch_debug["chunks"].append({
                                            "chunk": chunk_idx,
                                            "start": chunk_start.isoformat(),
                                            "end": chunk_end.isoformat(),
                                            "fetched": len(chunk_traces),
                                            "new": new_traces,
                                            "debug": chunk_debug,
                                        })
                                        
                                        status.update(label=f"Completed {completed_count}/{len(chunks)} chunks ({len(all_traces)} traces so far)")
                                    except Exception as e:
                                        chunk_info = futures[future]
                                        fetch_debug["chunks"].append({
                                            "chunk": chunk_info[0],
                                            "start": chunk_info[1].isoformat(),
                                            "end": chunk_info[2].isoformat(),
                                            "error": str(e),
                                        })
                                        status.update(label=f"Chunk {chunk_info[0]} failed: {e}")
                            
                            fetch_debug["total_traces"] = len(all_traces)
                            status.update(label=f"Fetched {len(all_traces)} total traces from {len(chunks)} chunks", state="complete")
                        
                        traces = all_traces
                    else:
                        # Single fetch for <=7 days
                        from_iso = iso_utc(datetime.combine(start_date, time.min).replace(tzinfo=timezone.utc))
                        to_iso = iso_utc(datetime.combine(end_date, time.max).replace(tzinfo=timezone.utc))
                        
                        with fetch_status.status("Fetching traces...", expanded=False):
                            fetch_debug: dict[str, Any] = {}
                            sig = None
                            try:
                                sig = inspect.signature(fetch_traces_window)
                            except Exception:
                                sig = None

                            supports_debug_out = bool(sig and "debug_out" in sig.parameters)
                            supports_page_size = bool(sig and "page_size" in sig.parameters)
                            supports_http_timeout = bool(sig and "http_timeout_s" in sig.parameters)

                            call_kwargs: dict[str, Any] = {
                                "base_url": base_url,
                                "headers": headers,
                                "from_iso": from_iso,
                                "to_iso": to_iso,
                                "envs": envs,
                                "page_limit": int(stats_page_limit),
                                "max_traces": int(stats_max_traces),
                                "retry": 2,
                                "backoff": 0.5,
                            }
                            if supports_page_size:
                                call_kwargs["page_size"] = int(stats_page_size)

                            if supports_http_timeout:
                                call_kwargs["http_timeout_s"] = float(stats_http_timeout_s)

                            if supports_debug_out:
                                call_kwargs["debug_out"] = fetch_debug

                            traces = fetch_traces_window(**call_kwargs)
                            fetch_status.status(f"Fetched {len(traces)} traces", state="complete", expanded=False)
                    
                    st.session_state.fetch_debug = fetch_debug
                    st.session_state.stats_traces = traces
                    st.rerun()

    tabs = st.tabs([
        "📊 Analytics Report",
        "✅ Human Eval tool",
        "🧠 Product Intelligence",
        "🔎 Trace Explorer",
        "🔗 Conversation Browser",
    ])

    with tabs[0]:
        render_analytics(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            base_thread_url=base_thread_url,
            gemini_api_key=gemini_api_key,
            use_date_filter=use_date_filter,
            start_date=start_date,
            end_date=end_date,
            envs=envs,
            stats_page_limit=stats_page_limit,
            stats_max_traces=stats_max_traces,
        )

    with tabs[1]:
        render_human_eval(
            base_thread_url=base_thread_url,
            gemini_api_key=gemini_api_key,
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
        )

    with tabs[2]:
        render_product_dev(
            base_thread_url=base_thread_url,
            gemini_api_key=gemini_api_key,
        )

    with tabs[3]:
        render_trace_explorer(
            base_thread_url=base_thread_url,
        )

    with tabs[4]:
        render_session_urls(
            base_thread_url=base_thread_url,
        )


if __name__ == "__main__":
    main()
