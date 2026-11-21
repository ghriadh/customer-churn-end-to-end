"""Shared helpers for every page: styles, layout blocks, cached loaders, labels and chart styling."""
import json
import os

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from churn.config import (
    COEFFICIENTS_PATH, IMPORTANCE_PATH, METRICS_PATH, RESULTS_PATH, TEST_PREDICTIONS_PATH, model_path,
)
from churn.data import ID_COL, TARGET, load_data
from i18n import get_lang, t

# ── Palette (validated for colour-vision deficiencies) ──────────────────────
CHURN_COLORS = {"No": "#2a78d6", "Yes": "#eb6834"}
HIGHLIGHT = "#2a78d6"   # the model in focus; other models are drawn in NEUTRAL (6 hues would not stay distinguishable)
NEUTRAL = "#a3a29c"
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "ridge": "Ridge Classifier",
    "sgd": "SGD Classifier",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting (hist)",
    "adaboost": "AdaBoost",
}
METRIC_NAMES = {
    "en": {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "precision": "Precision",
           "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "average_precision": "PR-AUC"},
    "fr": {"accuracy": "Accuracy", "balanced_accuracy": "Accuracy équilibrée", "precision": "Précision",
           "recall": "Rappel", "f1": "F1", "roc_auc": "ROC-AUC", "average_precision": "PR-AUC"},
}
METRICS = list(METRIC_NAMES["en"])

COMMON = {
    "en": {
        "no_artifacts": "No trained models found. Run `python run_pipeline.py` from the project root, then reload.",
        "code": "Show the code",
        "churn": "Churn", "no_churn": "No churn", "count": "Customers", "share": "Share",
        "actual": "Actual", "predicted": "Predicted",
        "cell_tn": "Correctly left alone", "cell_fp": "False alarm", "cell_fn": "Missed churner", "cell_tp": "Caught churner",
    },
    "fr": {
        "no_artifacts": "Aucun modèle entraîné. Lancez `python run_pipeline.py` depuis la racine du projet, puis rechargez.",
        "code": "Voir le code",
        "churn": "Churn", "no_churn": "Pas de churn", "count": "Clients", "share": "Part",
        "actual": "Réel", "predicted": "Prédit",
        "cell_tn": "À juste titre non signalé", "cell_fp": "Fausse alerte", "cell_fn": "Churner manqué", "cell_tp": "Churner détecté",
    },
}


def metric_name(key):
    return METRIC_NAMES[get_lang()].get(key, key)


# ── Styles & layout ─────────────────────────────────────────────────────────
_CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1200px; }
.eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
           color: #2a78d6; margin-bottom: 2px; }
.page-caption { color: #52514e; font-size: 15px; margin-top: -6px; margin-bottom: 18px; }
.kpi { border: 1px solid #e6e5e1; border-radius: 10px; padding: 14px 16px; background: #fcfcfb; height: 100%; }
.kpi-label { font-size: 12px; color: #52514e; text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { font-size: 28px; font-weight: 650; color: #0b0b0b; line-height: 1.25; }
.kpi-sub { font-size: 13px; color: #52514e; }
.finding { border-left: 3px solid #2a78d6; background: #f3f7fd; padding: 10px 14px;
           border-radius: 0 8px 8px 0; margin: 8px 0 14px 0; color: #1d1d1b; }
.finding p { margin: 0 0 4px 0; }
[data-testid="stExpander"] { border-radius: 10px; }
</style>
"""


def setup_page():
    st.markdown(_CSS, unsafe_allow_html=True)


def header(eyebrow, title, caption=None):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    if caption:
        st.markdown(f'<div class="page-caption">{caption}</div>', unsafe_allow_html=True)


def finding(markdown_text):
    """Highlighted comment / insight block (accepts Markdown)."""
    with st.container():
        st.markdown(f'<div class="finding">\n\n{markdown_text}\n\n</div>', unsafe_allow_html=True)


def kpis(items):
    """items: list of (label, value, sub-text)."""
    for col, (label, value, sub) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                     f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                     unsafe_allow_html=True)


def show_code(code):
    with st.expander(t(COMMON, "code")):
        st.code(code, language="python")


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


# ── Data & artifacts ────────────────────────────────────────────────────────
@st.cache_data
def raw_data():
    return load_data()


@st.cache_data
def eda_data():
    """Data used for exploration: customerID dropped, TotalCharges numeric, labels kept as text."""
    df = load_data().drop(columns=ID_COL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def artifacts_ready():
    return all(os.path.exists(p) for p in (METRICS_PATH, RESULTS_PATH, TEST_PREDICTIONS_PATH))


def require_artifacts():
    if not artifacts_ready():
        st.error(t(COMMON, "no_artifacts"))
        st.stop()


@st.cache_data
def metrics():
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cv_results():
    return pd.read_csv(RESULTS_PATH)


@st.cache_data
def test_predictions():
    return pd.read_csv(TEST_PREDICTIONS_PATH)


@st.cache_data
def importances():
    return pd.read_csv(IMPORTANCE_PATH)


@st.cache_data
def coefficients():
    return pd.read_csv(COEFFICIENTS_PATH)


@st.cache_resource
def model(name):
    return joblib.load(model_path(name))


# ── Charts ──────────────────────────────────────────────────────────────────
def style(chart, height=320):
    """Recessive axes and grid, readable text, consistent size."""
    return (chart.properties(height=height)
            .configure(background=SURFACE)
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=TEXT_2, titleColor=TEXT_2, gridColor=GRID, domainColor=GRID,
                            tickColor=GRID, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor=TEXT_2, titleColor=TEXT_2, labelFontSize=12, titleFontSize=12,
                              orient="top")
            .configure_title(color=TEXT, fontSize=14, anchor="start", fontWeight=600))


def churn_scale():
    return alt.Scale(domain=["No", "Yes"], range=[CHURN_COLORS["No"], CHURN_COLORS["Yes"]])


def highlight(field, value):
    """Colour the mark whose `field` equals `value`, grey for the others."""
    return alt.condition(alt.datum[field] == value, alt.value(HIGHLIGHT), alt.value(NEUTRAL))


def confusion_chart(tn, fp, fn, tp, height=240):
    """Confusion matrix as a heatmap, with plain-language names for the four cells in the tooltip."""
    labels = ["No", "Yes"]
    actual, predicted = t(COMMON, "actual"), t(COMMON, "predicted")
    cells = [("No", "No", tn, "cell_tn"), ("No", "Yes", fp, "cell_fp"), ("Yes", "No", fn, "cell_fn"), ("Yes", "Yes", tp, "cell_tp")]
    cm = pd.DataFrame([(a, p, int(v), t(COMMON, k)) for a, p, v, k in cells], columns=[actual, predicted, "n", "group"])
    heat = alt.Chart(cm).mark_rect(cornerRadius=4, stroke=SURFACE, strokeWidth=2).encode(
        x=alt.X(f"{predicted}:N", sort=labels, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y(f"{actual}:N", sort=labels),
        color=alt.Color("n:Q", scale=alt.Scale(range=["#cde2fb", "#184f95"]), legend=None),
        tooltip=["group", actual, predicted, "n"])
    text = heat.mark_text(fontSize=18, fontWeight=600).encode(
        text="n:Q", color=alt.condition(f"datum.n > {max(tn, fp, fn, tp) / 2}", alt.value("#ffffff"), alt.value(TEXT)))
    draw(heat + text, height)


def confusion_grid(items, per_row=3, height=190):
    """items: list of (title, (tn, fp, fn, tp), caption)."""
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, (title, (tn, fp, fn, tp), caption) in zip(cols, items[start:start + per_row]):
            with col:
                st.markdown(f"**{title}**")
                confusion_chart(tn, fp, fn, tp, height)
                st.caption(caption)


def draw(chart, height=320):
    st.altair_chart(style(chart, height), width="stretch", theme=None)
