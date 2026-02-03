"""Trace Explorer page."""

import streamlit as st

from utils.shared_ui import check_authentication, render_sidebar
from tabs.trace_explorer import render as render_trace_explorer


st.set_page_config(page_title="Trace Explorer - Tracey", page_icon="🔎", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_trace_explorer(
    base_thread_url=config["base_thread_url"],
)
