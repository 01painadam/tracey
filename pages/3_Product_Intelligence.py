"""Product Intelligence page."""

import streamlit as st

from utils.shared_ui import check_authentication, render_sidebar
from tabs.product_dev import render as render_product_dev


st.set_page_config(page_title="Product Intelligence - Tracey", page_icon="🧠", layout="wide")

if not check_authentication():
    st.stop()

config = render_sidebar()

render_product_dev(
    base_thread_url=config["base_thread_url"],
    gemini_api_key=config["gemini_api_key"],
)
