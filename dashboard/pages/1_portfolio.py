"""Mode A — Portfolio (passive).

Read-only view of Gold features, model performance, data-quality reports,
and the scorecard table. Nothing on this page writes to or re-runs any
part of the pipeline.
"""
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st
from deltalake import DeltaTable
from mlflow.tracking import MlflowClient

from config.config import GOLD_PATH, MLFLOW_URI
from ml.train import MODEL_PATH

st.set_page_config(page_title="Portfolio — Credit Risk Scorecard", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] { font-weight: 600; color: #5b6573; }
    [data-testid="stMetricValue"] { color: #1f2937; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    div[data-testid="stPlotlyChart"] {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CHART_TEMPLATE = "plotly_white"
COLOR_GOOD = "#3b82f6"
COLOR_BAD = "#ef4444"


def _style_chart(fig, legend_bottom=True):
    fig.update_layout(
        template=CHART_TEMPLATE,
        title_font=dict(size=20, color="#1f2937"),
        font=dict(family="sans-serif", size=13, color="#374151"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f3", zeroline=False)
    if legend_bottom:
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
    return fig


st.title("📊 Portfolio Analysis")
st.caption("Mode A — passive. Reads directly from Gold, MLflow, and DQ reports; nothing here modifies the pipeline.")
st.divider()


@st.cache_data
def load_gold_with_scores():
    gold = DeltaTable(f"{GOLD_PATH}gold_features").to_pandas()
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    gold["PD"] = model.predict_proba(gold[feature_columns])[:, 1]
    return gold


@st.cache_data
def load_latest_evaluation_metrics():
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    experiment = client.get_experiment_by_name("credit_risk_scorecard")
    if experiment is None:
        return {}
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'evaluation'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0].data.metrics if runs else {}


@st.cache_data
def load_iv_report():
    path = Path("docs/iv_report.json")
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_scorecard():
    path = Path("ml/scorecard.csv")
    return pd.read_csv(path) if path.exists() else None


tab_scores, tab_metrics, tab_dq, tab_scorecard = st.tabs(
    ["📈 Score Distribution", "🧮 Model Metrics", "✅ Data Quality", "🏦 Scorecard & IV"]
)

with tab_scores:
    try:
        gold = load_gold_with_scores()
        outcome = gold["TARGET"].map({0: "Good", 1: "Bad"})

        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Portfolio size", f"{len(gold):,}")
        col2.metric("⚠️ Observed default rate", f"{gold['TARGET'].mean() * 100:.2f}%")
        col3.metric("🎯 Avg. predicted PD", f"{gold['PD'].mean() * 100:.2f}%")

        st.write("")
        fig = px.histogram(
            gold, x="PD", color=outcome,
            nbins=40, barmode="overlay", opacity=0.7,
            labels={"PD": "Predicted Probability of Default", "color": "Outcome"},
            title="Predicted PD Distribution by Outcome",
            color_discrete_map={"Good": COLOR_GOOD, "Bad": COLOR_BAD},
        )
        fig = _style_chart(fig)
        fig.update_xaxes(rangeslider_visible=True, tickformat=".0%")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width="stretch")
    except FileNotFoundError:
        st.warning("gold_features or model.pkl not found — run the pipeline and ml/train.py first.")

with tab_metrics:
    metrics = load_latest_evaluation_metrics()
    if metrics:
        cols = st.columns(3)
        cols[0].metric("📐 AUC", f"{metrics.get('auc', 0):.3f}")
        cols[1].metric("📏 Gini", f"{metrics.get('gini', 0):.3f}")
        cols[2].metric("📊 KS", f"{metrics.get('ks', 0):.3f}")

        st.write("")
        eval_report_path = Path("docs/evaluation_report.json")
        if eval_report_path.exists():
            with st.expander("Full evaluation report (JSON)", expanded=False):
                st.json(json.loads(eval_report_path.read_text()))
    else:
        st.warning("No MLflow 'evaluation' run found yet — run ml/evaluate.py first.")

with tab_dq:
    reports_dir = Path("dq_checks/reports")
    html_reports = sorted(reports_dir.glob("*.html"), reverse=True) if reports_dir.exists() else []
    if html_reports:
        selected = st.selectbox("📄 Select a DQ report", html_reports, format_func=lambda p: p.name)
        st.markdown(
            '<div style="border:1px solid #e6e9ef; border-radius:12px; overflow:hidden; '
            'box-shadow:0 1px 3px rgba(0,0,0,0.06);">',
            unsafe_allow_html=True,
        )
        st.iframe(selected, height=600)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No DQ reports found yet — run the pipeline scripts first.")

with tab_scorecard:
    iv_report = load_iv_report()
    if iv_report:
        iv_df = pd.DataFrame(
            [
                {"feature": k, "iv": v, "selected": k in iv_report["selected_features"]}
                for k, v in iv_report["iv_by_feature"].items()
            ]
        ).sort_values("iv", ascending=False)
        fig = px.bar(
            iv_df, x="feature", y="iv", color="selected",
            title=(
                f"Information Value by Feature "
                f"(selection range [{iv_report['iv_min']}, {iv_report['iv_max']}])"
            ),
            color_discrete_map={True: COLOR_GOOD, False: "#d1d5db"},
            labels={"iv": "Information Value", "feature": "", "selected": "Selected"},
        )
        fig = _style_chart(fig, legend_bottom=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("docs/iv_report.json not found — run ingestion/silver_woe.py first.")

    scorecard_df = load_scorecard()
    if scorecard_df is not None:
        st.write("")
        st.subheader("🏦 Credit Scorecard")
        st.dataframe(scorecard_df, width="stretch", height=400)
    else:
        st.warning("ml/scorecard.csv not found — run ml/train.py first.")
