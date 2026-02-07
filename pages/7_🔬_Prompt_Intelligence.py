"""Prompt Intelligence page."""

import streamlit as st

from utils.shared_ui import check_authentication, render_sidebar
from tabs.prompt_intelligence import render as render_prompt_intelligence


st.set_page_config(page_title="Prompt Intelligence - Tracey", page_icon="🔬", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_prompt_intelligence(
    gemini_api_key=config["gemini_api_key"],
    start_date=config["start_date"],
    end_date=config["end_date"],
)
