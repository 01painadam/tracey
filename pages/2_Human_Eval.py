"""Human Evaluation page."""

import streamlit as st

from utils.shared_ui import check_authentication, render_sidebar
from tabs.human_eval import render as render_human_eval


st.set_page_config(page_title="Human Eval - Tracey", page_icon="✅", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_human_eval(
    base_thread_url=config["base_thread_url"],
    gemini_api_key=config["gemini_api_key"],
    public_key=config["public_key"],
    secret_key=config["secret_key"],
    base_url=config["base_url"],
)
