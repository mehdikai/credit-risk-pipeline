"""Mode A — Portfolio (passive).

Read-only view of Gold features, model performance, data-quality reports,
and the scorecard table. Nothing on this page writes to or re-runs any
part of the pipeline.
"""
import functools
import http.server
import json
import pickle
import sys
import threading
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
from dq_checks.ge_utils import DATA_DOCS_INDEX
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


DATA_DOCS_SERVER_PORT = 8765


@st.cache_resource
def _start_data_docs_server():
    """Serve the GE Data Docs directory over plain HTTP.

    Browsers block navigation to file:// URLs clicked from an http:// page
    (cross-origin file access is disallowed), so file:// link buttons never
    actually open. Serving the same directory over localhost HTTP sidesteps
    that restriction entirely. st.cache_resource starts this once per app
    process, not on every script rerun.
    """
    directory = str(DATA_DOCS_INDEX.parent)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    server = http.server.ThreadingHTTPServer(("localhost", DATA_DOCS_SERVER_PORT), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _data_docs_url(html_path=None):
    _start_data_docs_server()
    rel_path = "index.html" if html_path is None else html_path.relative_to(DATA_DOCS_INDEX.parent)
    return f"http://localhost:{DATA_DOCS_SERVER_PORT}/{rel_path}"


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


@st.cache_data
def load_suite_statuses():
    """Latest validation result (pass/fail + link to its Data Docs page) per suite."""
    validations_dir = DATA_DOCS_INDEX.parent.parent.parent / "validations"
    data_docs_validations_dir = DATA_DOCS_INDEX.parent / "validations"
    if not validations_dir.exists():
        return []

    statuses = []
    for suite_dir in sorted(p for p in validations_dir.iterdir() if p.is_dir()):
        json_files = sorted(suite_dir.rglob("*.json"), reverse=True)
        if not json_files:
            continue
        latest_json = json_files[0]
        data = json.loads(latest_json.read_text())
        html_path = data_docs_validations_dir / latest_json.relative_to(validations_dir).with_suffix(".html")
        statuses.append({
            "suite": suite_dir.name.replace("_suite", ""),
            "success": data.get("success", False),
            "success_pct": data.get("statistics", {}).get("success_percent", 0),
            "html_path": html_path,
        })
    return statuses


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
    if DATA_DOCS_INDEX.exists():
        st.markdown(
            "Every pipeline run validates each layer through a Great Expectations "
            "checkpoint, which builds the full **GE Data Docs** site below — "
            "per-suite pass/fail history, expectation-level drill-down, and charts."
        )
        st.link_button("🔍 Open Full Data Quality Report ↗", _data_docs_url(), width="stretch")
        st.divider()

        suite_statuses = load_suite_statuses()
        if suite_statuses:
            n_pass = sum(s["success"] for s in suite_statuses)
            st.subheader(f"Validated suites — {n_pass}/{len(suite_statuses)} passing")

            header = st.columns([3, 1, 1, 1.3])
            for col, label in zip(header, ["Suite", "Status", "Pass %", ""]):
                col.markdown(f"**{label}**")

            for s in suite_statuses:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1.3])
                col1.markdown(f"`{s['suite']}`")
                if s["success"]:
                    col2.markdown(":green[✅ Pass]")
                else:
                    col2.markdown(":red[❌ Fail]")
                col3.markdown(f"{s['success_pct']:.0f}%")
                if s["html_path"].exists():
                    col4.link_button("View ↗", _data_docs_url(s["html_path"]))
        else:
            st.info("No suite results found yet — run the pipeline scripts first.")
    else:
        st.warning("No Data Docs found yet — run the pipeline scripts first.")

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
