import json
from typing import Any

import streamlit as st


def _as_dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for c in content:
            if isinstance(c, dict):
                t = c.get("text")
                if isinstance(t, str) and t.strip():
                    out.append(t)
                elif isinstance(c.get("content"), str) and str(c.get("content") or "").strip():
                    out.append(str(c.get("content") or ""))
        return "\n".join(out)
    return ""


def _strip_noise(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "__gemini_function_call_thought_signatures__":
                continue
            cleaned[k] = _strip_noise(v)
        return cleaned
    if isinstance(obj, list):
        return [_strip_noise(x) for x in obj]
    return obj


def _first_user_prompt_snippet(t: dict[str, Any], max_len: int = 80) -> str:
    input_msgs = _as_list(_as_dict(t.get("input")).get("messages"))
    for m in input_msgs:
        md = _as_dict(m)
        if str(md.get("type") or "") != "human":
            continue
        text = _content_text(md.get("content")).strip()
        if not text:
            continue
        if len(text) > max_len:
            return text[: max_len - 1] + "…"
        return text
    return ""


def _current_user_prompt(t: dict[str, Any]) -> str:
    input_msgs = _as_list(_as_dict(t.get("input")).get("messages"))
    for m in reversed(input_msgs):
        md = _as_dict(m)
        if str(md.get("type") or "") != "human":
            continue
        text = _content_text(md.get("content")).strip()
        if text:
            return text
    return ""


def _mtype(m: dict[str, Any]) -> str:
    t = (m.get("type") or m.get("role") or "").lower()
    return {"assistant": "ai", "user": "human"}.get(t, t)


def _stop_reason(m: dict[str, Any]) -> str:
    meta = m.get("response_metadata") or {}
    sr = (
        meta.get("stop_reason")
        or meta.get("finish_reason")
        or m.get("stop_reason")
        or m.get("finish_reason")
        or ""
    )
    return str(sr).lower()


def _get_msg_content(m: dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out: list[str] = []
        for x in c:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                if isinstance(x.get("text"), str):
                    out.append(str(x.get("text") or ""))
                elif isinstance(x.get("content"), str):
                    out.append(str(x.get("content") or ""))
        return " ".join([o for o in out if o])
    if isinstance(c, dict):
        return str(c.get("text") or c.get("content") or "")
    return ""


def _is_meaningful_human(text: str) -> bool:
    if not text or not text.strip():
        return False
    t = text.strip().lower()
    prefixes = (
        "user selected",
        "user clicked",
        "user chose",
        "user set",
        "user changed",
        "user uploaded",
        "user drew",
        "user toggled",
        "selected aoi",
        "selected dataset",
        "user selected aoi",
    )
    if t.startswith(prefixes):
        return False
    return any(ch.isalnum() for ch in text)


def _find_active_turn_window(msgs: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    if not msgs:
        return None, None

    end_idxs: list[int] = []
    for i, m in enumerate(msgs):
        if _mtype(m) != "ai":
            continue

        sr = _stop_reason(m)
        if sr == "end_turn":
            end_idxs.append(i)
            continue

        if sr == "stop":
            next_idx = i + 1
            if next_idx >= len(msgs):
                end_idxs.append(i)
            elif _mtype(msgs[next_idx]) == "human":
                end_idxs.append(i)

    if not end_idxs:
        start: int | None = None
        for j in range(len(msgs) - 1, -1, -1):
            if _mtype(msgs[j]) == "human" and _is_meaningful_human(_get_msg_content(msgs[j])):
                start = j
                break
        return start, None

    end = end_idxs[-1]
    prev_end = end_idxs[-2] if len(end_idxs) > 1 else -1

    start: int | None = None
    for j in range(end - 1, prev_end, -1):
        if _mtype(msgs[j]) == "human" and _is_meaningful_human(_get_msg_content(msgs[j])):
            start = j
            break

    if start is None:
        start = prev_end + 1 if prev_end + 1 <= end else None

    return start, end


def _slice_output_to_current_turn(trace: dict[str, Any], output_msgs: list[Any]) -> list[Any]:
    """Return output messages starting from the current (last) human prompt.

    Some traces include full conversation history in output.messages. For debugging the current
    trace turn, we slice from the last occurrence of the current user prompt.
    """
    cur = _current_user_prompt(trace)
    if not cur:
        return output_msgs

    start_idx: int | None = None
    for i, m in enumerate(output_msgs):
        md = _as_dict(m)
        if str(md.get("type") or "") != "human":
            continue
        text = _content_text(md.get("content")).strip()
        if text == cur:
            start_idx = i

    if start_idx is None:
        return output_msgs
    return output_msgs[start_idx:]


def _trace_label(t: dict[str, Any]) -> str:
    snippet = _first_user_prompt_snippet(t)
    if snippet:
        return snippet
    return "(empty prompt)"


def render(base_thread_url: str) -> None:
    st.subheader("🔎 Trace Explorer")
    st.caption(
        "Inspect individual traces in detail: view the current user turn, assistant output, tool calls, metadata, and raw JSON. "
        "Useful for debugging odd outputs, latency/cost spikes, and tool failures."
    )

    traces: list[dict[str, Any]] = st.session_state.get("stats_traces", [])
    if not traces:
        st.info("Use the sidebar **🚀 Fetch traces** button first.")
        return

    with st.expander("🔽 Apply Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            session_id_filter = st.text_input(
                "Session id",
                value=str(st.session_state.get("trace_explorer_filter_session_id") or ""),
                key="trace_explorer_filter_session_id",
                placeholder="e.g. 7f6b…",
            )
        with f2:
            trace_id_filter = st.text_input(
                "Trace id",
                value=str(st.session_state.get("trace_explorer_filter_trace_id") or ""),
                key="trace_explorer_filter_trace_id",
                placeholder="e.g. 3a2c…",
            )
        with f3:
            prompt_substring_filter = st.text_input(
                "Prompt contains",
                value=str(st.session_state.get("trace_explorer_filter_prompt_substring") or ""),
                key="trace_explorer_filter_prompt_substring",
                placeholder="substring…",
            )

        session_id_filter_n = str(session_id_filter or "").strip().lower()
        trace_id_filter_n = str(trace_id_filter or "").strip().lower()
        prompt_substring_filter_n = str(prompt_substring_filter or "").strip().lower()

        filtered_idxs: list[int] = []
        for i, t in enumerate(traces):
            tid = str(t.get("id") or "").strip().lower()
            sid = str(t.get("sessionId") or "").strip().lower()
            prompt = str(_current_user_prompt(t) or "").strip().lower()

            if session_id_filter_n and session_id_filter_n not in sid:
                continue
            if trace_id_filter_n and trace_id_filter_n not in tid:
                continue
            if prompt_substring_filter_n and prompt_substring_filter_n not in prompt:
                continue

            filtered_idxs.append(i)

    if not filtered_idxs:
        st.warning("No traces match the current filters.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        prev_selected = st.session_state.get("trace_explorer_selected_idx")
        try:
            prev_selected_int = int(prev_selected) if prev_selected is not None else None
        except Exception:
            prev_selected_int = None

        default_pos = 0
        if isinstance(prev_selected_int, int) and prev_selected_int in filtered_idxs:
            default_pos = filtered_idxs.index(prev_selected_int)

        filtered_traces_str = f"(Showing {len(filtered_idxs):,} / {len(traces):,} traces)" if len(filtered_idxs) < len(traces) else ""
        idx = st.selectbox(
            f"Select trace {filtered_traces_str}",
            options=filtered_idxs,
            format_func=lambda i: _trace_label(traces[int(i)]),
            index=default_pos,
            key="trace_explorer_selected_idx",
        )

    with col2:
        hide_empty = st.checkbox("Hide empty messages", value=True, key="trace_explorer_hide_empty")
        show_raw = st.checkbox("Show raw JSON", value=False, key="trace_explorer_show_raw")

    trace = traces[int(idx)]
    trace_clean = _strip_noise(trace)

    tid = str(trace.get("id") or "")
    sid = str(trace.get("sessionId") or "")
    if sid:
        st.link_button("🔗 Open session in GNW", f"{base_thread_url.rstrip('/')}/{sid}")

    st.markdown("### Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Environment", str(trace.get("environment") or ""))
    with c2:
        st.metric("Latency (s)", str(trace.get("latency") or ""))
    with c3:
        st.metric("Total cost", str(trace.get("totalCost") or ""))
    with c4:
        obs = trace.get("observations")
        st.metric("Observations", str(len(obs)) if isinstance(obs, list) else "")

    st.markdown("### Input messages")
    current_prompt = _current_user_prompt(trace)
    if not current_prompt:
        st.info("No input.messages")
    else:
        st.code(current_prompt, language=None)

    output_msgs_all = _as_list(_as_dict(trace.get("output")).get("messages"))
    output_msgs = output_msgs_all
    output_msgs_clean = [_as_dict(m) for m in output_msgs if isinstance(m, dict)]
    window_start, window_end = _find_active_turn_window(output_msgs_clean)

    end_idx = window_end if window_end is not None else (len(output_msgs_clean) - 1)

    history_end = int(window_start) if isinstance(window_start, int) else 0
    has_history = bool(isinstance(window_start, int) and window_start > 0)

    with st.expander("### Tool calls", expanded=False):
        tool_results_by_call_id: dict[str, dict[str, Any]] = {}
        for m in output_msgs_clean:
            md = _as_dict(m)
            if str(md.get("type") or "") == "tool" and md.get("tool_call_id") is not None:
                tool_results_by_call_id[str(md.get("tool_call_id"))] = md

        if has_history:
            with st.expander("History (tool calls)", expanded=False):
                tool_calls_found_hist = 0
                for i in range(0, history_end):
                    md = output_msgs_clean[i]
                    if str(md.get("type") or "") != "ai":
                        continue
                    for tc in _as_list(md.get("tool_calls")):
                        tcd = _as_dict(tc)
                        call_id = str(tcd.get("id") or "")
                        name = str(tcd.get("name") or "")
                        args = tcd.get("args")
                        tool_calls_found_hist += 1

                        result = tool_results_by_call_id.get(call_id)
                        status = str(_as_dict(result).get("status") or "") if isinstance(result, dict) else ""
                        exp_label = f"{tool_calls_found_hist}. {name}"
                        if status:
                            exp_label = f"{exp_label} ({status})"
                        with st.expander(exp_label, expanded=False):
                            st.markdown("**Call**")
                            if call_id:
                                st.code(call_id, language=None)
                            st.json(_strip_noise({"name": name, "args": args}))

                            st.markdown("**Result**")
                            if isinstance(result, dict):
                                rtext = _content_text(result.get("content"))
                                if str(rtext or "").strip():
                                    st.code(rtext, language=None)
                                st.json(_strip_noise(result))
                            else:
                                st.info("No tool result message found for this tool_call_id")

                if tool_calls_found_hist == 0:
                    st.info("No tool calls found in history")

        tool_calls_found = 0
        window_started = False
        window_ended = False
        start_i = int(window_start) if isinstance(window_start, int) else 0
        for i in range(start_i, len(output_msgs_clean)):
            md = output_msgs_clean[i]
            if window_start is not None:
                if (not window_started) and i == window_start:
                    st.caption("Active window start 👇")
                    window_started = True
                if (not window_ended) and i == end_idx + 1:
                    st.caption("Active window end 👆")
                    window_ended = True

            if str(md.get("type") or "") != "ai":
                continue
            for tc in _as_list(md.get("tool_calls")):
                tcd = _as_dict(tc)
                call_id = str(tcd.get("id") or "")
                name = str(tcd.get("name") or "")
                args = tcd.get("args")
                tool_calls_found += 1

                result = tool_results_by_call_id.get(call_id)
                status = str(_as_dict(result).get("status") or "") if isinstance(result, dict) else ""
                exp_label = f"{tool_calls_found}. {name}"
                if status:
                    exp_label = f"{exp_label} ({status})"
                with st.expander(exp_label, expanded=False):
                    st.markdown("**Call**")
                    if call_id:
                        st.code(call_id, language=None)
                    st.json(_strip_noise({"name": name, "args": args}))

                    st.markdown("**Result**")
                    if isinstance(result, dict):
                        rtext = _content_text(result.get("content"))
                        if str(rtext or "").strip():
                            st.code(rtext, language=None)
                        st.json(_strip_noise(result))
                    else:
                        st.info("No tool result message found for this tool_call_id")

        if tool_calls_found == 0:
            st.info("No tool calls found in output.messages")

    with st.expander("### Output messages", expanded=False):
        if not output_msgs:
            st.info("No output.messages")
        else:
            if has_history:
                with st.expander("History (messages)", expanded=False):
                    any_shown_hist = False
                    for i in range(0, history_end):
                        md = output_msgs_clean[i]
                        mtype = str(md.get("type") or "")
                        name = str(md.get("name") or "")
                        text = _content_text(md.get("content"))
                        if hide_empty and not str(text or "").strip() and mtype not in {"ai", "tool"}:
                            continue
                        any_shown_hist = True
                        label = f"{i}: {mtype or 'message'}"
                        if name:
                            label = f"{label} ({name})"
                        with st.expander(label, expanded=False):
                            if str(text or "").strip():
                                st.code(text, language=None)
                            st.json(_strip_noise(md))

                    if not any_shown_hist:
                        st.info("No messages in history")

            window_started = False
            window_ended = False
            start_i = int(window_start) if isinstance(window_start, int) else 0
            for i in range(start_i, len(output_msgs_clean)):
                md = output_msgs_clean[i]
                if window_start is not None:
                    if (not window_started) and i == window_start:
                        st.caption("Active window start 👇")
                        window_started = True
                    if (not window_ended) and i == end_idx + 1:
                        st.caption("Active window end 👆")
                        window_ended = True

                mtype = str(md.get("type") or "")
                name = str(md.get("name") or "")
                text = _content_text(md.get("content"))
                if hide_empty and not str(text or "").strip() and mtype not in {"ai", "tool"}:
                    continue
                label = f"{i}: {mtype or 'message'}"
                if name:
                    label = f"{label} ({name})"
                with st.expander(label, expanded=False):
                    if str(text or "").strip():
                        st.code(text, language=None)
                    st.json(_strip_noise(md))

    st.markdown("### Cleaned trace JSON")
    with st.expander("View cleaned JSON (noise removed)", expanded=False):
        st.json(trace_clean)

    if show_raw:
        st.markdown("### Raw trace JSON")
        st.json(trace)

    st.download_button(
        "⬇️ Download trace JSON",
        data=json.dumps(trace, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=f"trace_{tid or 'unknown'}.json",
        mime="application/json",
        key=f"trace_explorer_dl_{tid or 'unknown'}",
    )
