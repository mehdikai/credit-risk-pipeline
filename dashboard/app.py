"""Streamlit entry point — Credit Risk Scorecard Dashboard.

Run with: streamlit run dashboard/app.py
Pages (auto-discovered from dashboard/pages/):
  1_portfolio.py     -- Mode A (passive): portfolio analysis, read-only
  2_live_scoring.py  -- Mode B (active): live applicant scoring (Week 9)
"""
import streamlit as st

st.set_page_config(page_title="Credit Risk Scorecard", page_icon="📊", layout="wide")

st.title("Credit Risk Scorecard — Dashboard")
st.markdown(
    """
This dashboard sits on top of the Bronze/Silver/Gold lakehouse pipeline and
the MLflow-tracked Logistic Regression scorecard.

- **Portfolio** (sidebar): passive view of Gold features, model performance,
  data-quality reports, and the scorecard table — the full-pipeline view.
- **Live Scoring** (Week 9): score a new applicant on the spot.

Use the sidebar to navigate between pages.
"""
)
st.info("Select a page from the sidebar to get started.")
