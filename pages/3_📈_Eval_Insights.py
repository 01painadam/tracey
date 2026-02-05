"""Eval Insights page."""

import streamlit as st

from utils.shared_ui import check_authentication, render_sidebar
from tabs.eval_insights import render as render_eval_insights


st.set_page_config(page_title="Eval Insights - Tracey", page_icon="📈", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_eval_insights(
    public_key=config["public_key"],
    secret_key=config["secret_key"],
    base_url=config["base_url"],
)
