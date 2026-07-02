"""Conversation Browser page."""

import streamlit as st

from utils.shared_ui import check_authentication, render_sidebar
from tabs.session_urls import render as render_session_urls


st.set_page_config(page_title="Conversation Browser - Tracey", page_icon="🔗", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_session_urls(
    base_thread_url=config["base_thread_url"],
    start_date=config["start_date"],
    end_date=config["end_date"],
    envs=config["envs"],
    zeno_api_url=config["zeno_api_url"],
    zeno_api_token=config["zeno_api_token"],
)
