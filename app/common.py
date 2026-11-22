"""Shared helpers for every page: styles, layout blocks, number formatting, cached loaders and charts."""
import inspect
import json
import os

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from churn.config import (
    COEFFICIENTS_PATH, IMPORTANCE_PATH, METRICS_PATH, RESULTS_PATH, TEST_PREDICTIONS_PATH, model_path,
)
from churn.data import ID_COL, load_data
from i18n import get_lang, t

# ── Palette (validated for colour-vision deficiencies) ──────────────────────
CHURN_COLORS = {"No": "#2a78d6", "Yes": "#eb6834"}
HIGHLIGHT = "#2a78d6"   # the model in focus; other models are drawn in NEUTRAL (6 hues would not stay distinguishable)
NEUTRAL = "#b4b3ad"
# Metric curves (recall / precision / F1): hues distinct from the churn colours, which mean classes
METRIC_COLORS = ["#6250d6", "#1baf7a", "#eda100"]
TEXT, TEXT_2, GRID, SURFACE = "#1a1a1a", "#3d3d3a", "#e6e5e1", "#ffffff"

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
        "code": "Show the code", "tech": "Technical details", "brief": "In short", "controls": "Your turn",
        "count": "Customers", "share": "Share", "actual": "Actual", "predicted": "Predicted",
        "cell_tn": "Correctly left alone", "cell_fp": "False alarm", "cell_fn": "Missed churner", "cell_tp": "Caught churner",
    },
    "fr": {
        "no_artifacts": "Aucun modèle entraîné. Lancez `python run_pipeline.py` depuis la racine du projet, puis rechargez.",
        "code": "Voir le code", "tech": "Détails techniques", "brief": "En bref", "controls": "À vous de jouer",
        "count": "Clients", "share": "Part", "actual": "Réel", "predicted": "Prédit",
        "cell_tn": "À juste titre non signalé", "cell_fp": "Fausse alerte", "cell_fn": "Churner manqué", "cell_tp": "Churner détecté",
    },
}


def metric_name(key):
    return METRIC_NAMES[get_lang()].get(key, key)


# ── Number formatting (French: 7 043 and 0,82) ──────────────────────────────
def num(x, decimals=0):
    s = f"{x:,.{decimals}f}"
    if get_lang() == "fr":
        s = s.replace(",", " ").replace(".", ",")
    return s


def pct(x, decimals=1):
    s = f"{x * 100:.{decimals}f}"
    return s.replace(".", ",") + " %" if get_lang() == "fr" else s + "%"


# ── Styles ──────────────────────────────────────────────────────────────────
_CSS = """
<style>
.block-container { padding-top: 3.2rem; max-width: 1250px; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { font-size: 1.03rem; line-height: 1.65; }
h2, h3 { color: #0d366b !important; }

/* Inline code: readable size and colour instead of small green text */
[data-testid="stMarkdownContainer"] code { color: #184f95; background: #e8f1fc; font-size: .95em;
    padding: .08em .42em; border-radius: 5px; font-weight: 500; }

/* Page banner and home hero */
.page-banner { background: linear-gradient(120deg, #eaf2fd 0%, #f3efff 100%); border: 1px solid #dbe7fa;
    border-radius: 18px; padding: 22px 28px 18px; margin-bottom: 18px; }
.page-banner .eyebrow, .hero .eyebrow { font-size: .78rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #2a78d6; margin-bottom: 4px; }
.page-banner h1 { margin: 0 !important; padding: 0 !important; font-size: 2.3rem !important; color: #0d366b !important; }
.page-banner .caption { font-size: 1.06rem; color: #3d3d3a; margin-top: 6px; }
.hero { background: linear-gradient(120deg, #1c5cab 0%, #4a52c9 55%, #6250d6 100%); border-radius: 22px;
    padding: 34px 38px 30px; color: #fff; margin-bottom: 22px; box-shadow: 0 8px 24px rgba(28, 92, 171, .25); }
.hero .eyebrow { color: #cde2fb; }
.hero h1 { color: #fff !important; font-size: 2.7rem !important; margin: 0 !important; padding: 0 !important; }
.hero .lead { font-size: 1.15rem; color: #eef4fd; margin-top: 10px; max-width: 760px; line-height: 1.6; }
.hero .author { font-size: .95rem; color: #cde2fb; margin-top: 14px; }

/* "In short" box (plain-language summary at the top of each page) */
.brief { background: #fff7e6; border: 1px solid #f6dca3; border-left: 6px solid #eda100; border-radius: 14px;
    padding: 14px 20px 10px; margin: 4px 0 24px; }
.brief-title { font-weight: 800; color: #8a5a00; font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; }
.brief p, .brief li { font-size: 1.07rem !important; color: #2b2410; }

/* Comment / insight box */
.finding { border-left: 5px solid #2a78d6; background: #f3f7fd; padding: 12px 18px 6px;
    border-radius: 0 12px 12px 0; margin: 10px 0 18px 0; }

/* Numbered steps */
.step { display: flex; align-items: center; gap: 14px; margin: 34px 0 10px; }
.step-num { background: linear-gradient(135deg, #2a78d6, #6250d6); color: #fff; min-width: 38px; height: 38px;
    border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.1rem; }
.step-title { font-size: 1.5rem; font-weight: 700; color: #0d366b; line-height: 1.2; }

/* KPI cards */
.kpi { background: #fff; border: 1px solid #e3e8f0; border-top: 5px solid #2a78d6; border-radius: 14px;
    padding: 14px 18px; height: 100%; box-shadow: 0 2px 8px rgba(16, 24, 40, .06); }
.kpi.orange { border-top-color: #eb6834; } .kpi.green { border-top-color: #1baf7a; } .kpi.violet { border-top-color: #6250d6; }
.kpi-label { font-size: .8rem; color: #3d3d3a; text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }
.kpi-value { font-size: 2.1rem; font-weight: 750; color: #1a1a1a; line-height: 1.25; }
.kpi-sub { font-size: .95rem; color: #3d3d3a; }

/* Widgets: make every control stand out */
[data-testid="stWidgetLabel"] p { font-weight: 650 !important; color: #0d366b !important; font-size: 1rem !important; }
/* Radio buttons as pills (Streamlit 1.67: stRadioGroup / stRadioOption with react-aria data-selected) */
[data-testid="stRadioGroup"] { gap: 8px !important; flex-wrap: wrap; }
[data-testid="stRadioOption"] { border: 1.5px solid #86b6ef; border-radius: 999px; padding: 5px 16px 5px 10px !important;
    background: #fff; margin-right: 4px !important; cursor: pointer; transition: background .15s, box-shadow .15s;
    box-shadow: 0 1px 3px rgba(16, 24, 40, .08); }
[data-testid="stRadioOption"][data-hovered] { background: #eaf2fd; }
[data-testid="stRadioOption"][data-selected] { background: #2a78d6; border-color: #2a78d6; box-shadow: 0 2px 6px rgba(42, 120, 214, .35); }
[data-testid="stRadioOption"][data-selected] p { color: #fff !important; font-weight: 650; }
/* Toggles and checkboxes: bold label, darker track when off */
[data-testid="stCheckbox"] p { font-weight: 600; color: #0d366b; }
[data-testid="stCheckbox"]:not([data-selected]) > div > div:first-child { background-color: #a9b8cc !important; }
/* Select boxes (Streamlit 1.67 react-aria combobox): tinted, bordered, value in bold, blue chevron */
[data-testid="stSelectbox"] div:has(> input) { border: 1.5px solid #6da7ec !important; background: #ffffff !important;
    border-radius: 10px !important; box-shadow: 0 1px 4px rgba(42, 120, 214, .15); transition: box-shadow .15s, border-color .15s; }
[data-testid="stSelectbox"] div:has(> input):hover { border-color: #2a78d6 !important; box-shadow: 0 3px 10px rgba(42, 120, 214, .28); }
[data-testid="stSelectbox"] input { background: transparent !important; font-weight: 650 !important; font-size: 1.04rem !important;
    color: #0d366b !important; -webkit-text-fill-color: #0d366b; cursor: pointer; }
[data-testid="stSelectbox"] div:has(> input) svg { color: #2a78d6 !important; fill: #2a78d6; }
[data-testid="stNumberInput"] input { background: #ffffff; font-weight: 600; }
.stButton > button, [data-testid="stFormSubmitButton"] > button { background: #2a78d6; color: #fff; border: none;
    font-weight: 650; padding: .55rem 1.4rem; border-radius: 10px; box-shadow: 0 2px 6px rgba(42, 120, 214, .35); }
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background: #1c5cab; color: #fff; }
.stButton > button p, [data-testid="stFormSubmitButton"] > button p { color: #fff; }

/* "Your turn" boxes that group the interactive controls */
[class*="st-key-ctl"] { background: #f5f9ff; border: 2px dashed #86b6ef !important; border-radius: 16px !important;
    padding: 12px 18px 4px; }
.ctl-title { font-weight: 800; color: #1c5cab; font-size: .82rem; letter-spacing: .1em; text-transform: uppercase; }

/* Tabs: every tab looks like a button, the active one in solid blue */
[data-testid="stTabs"] [role="tablist"] { gap: 8px; }
[data-testid="stTab"] { background: #e3edfa !important; border-radius: 10px 10px 0 0; padding: 0 20px !important;
    transition: background .15s; }
[data-testid="stTab"]:hover { background: #cfe0f7 !important; }
[data-testid="stTab"] p { font-size: 1.08rem !important; font-weight: 700 !important; color: #0d366b; }
[data-testid="stTab"][aria-selected="true"] { background: #2a78d6 !important; }
[data-testid="stTab"][aria-selected="true"] p { color: #fff !important; }

/* Expanders: raised and clearly clickable. "Show the code" = dark header, "Technical details" = violet */
[data-testid="stExpander"] details { border-radius: 12px; border: 1px solid #c9d8ee; border-left: 6px solid #2a78d6;
    background: #fff; box-shadow: 0 3px 10px rgba(16, 24, 40, .10); transition: box-shadow .15s, transform .15s; overflow: hidden; }
[data-testid="stExpander"] details:hover { box-shadow: 0 6px 16px rgba(16, 24, 40, .18); transform: translateY(-1px); }
[data-testid="stExpander"] summary { background: #eaf2fd; }
[data-testid="stExpander"] summary p { font-weight: 700; color: #0d366b; font-size: 1.04rem; }
[class*="st-key-xcode"] details { border-left-color: #0d366b; }
[class*="st-key-xcode"] summary { background: #1d2b45; }
[class*="st-key-xcode"] summary p, [class*="st-key-xcode"] summary svg { color: #fff !important; }
[class*="st-key-xcode"] summary:hover { background: #2a3d60; }
[class*="st-key-xtech"] details { border-left-color: #6250d6; border-color: #d9d3f6; }
[class*="st-key-xtech"] summary { background: #efecfd; }
[class*="st-key-xtech"] summary p, [class*="st-key-xtech"] summary svg { color: #3f31a8 !important; }
[class*="st-key-xtech"] summary:hover { background: #e3defb; }

/* Result column that stays visible while the form is scrolled (prediction page) */
[data-testid="stHorizontalBlock"]:has(.sticky-anchor) { align-items: flex-start; }
[data-testid="stColumn"]:has(.sticky-anchor) { position: sticky; top: 3.5rem; }

[data-testid="stCaptionContainer"] { opacity: 1 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: .95rem !important; color: #3d3d3a !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #eef4fd 0%, #f4f1ff 100%); }
</style>
"""


def setup_page():
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Layout blocks ───────────────────────────────────────────────────────────
def header(eyebrow, title, caption=None):
    cap = f'<div class="caption">{caption}</div>' if caption else ""
    st.markdown(f'<div class="page-banner"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>{cap}</div>',
                unsafe_allow_html=True)


def brief(markdown_text):
    """Plain-language summary for non-specialist readers."""
    st.markdown(f'<div class="brief"><div class="brief-title">💡 {t(COMMON, "brief")}</div>\n\n{markdown_text}\n\n</div>',
                unsafe_allow_html=True)


def finding(markdown_text):
    """Highlighted comment / insight block (accepts Markdown)."""
    st.markdown(f'<div class="finding">\n\n{markdown_text}\n\n</div>', unsafe_allow_html=True)


def step(n, title):
    st.markdown(f'<div class="step"><div class="step-num">{n}</div><div class="step-title">{title}</div></div>',
                unsafe_allow_html=True)


def controls(key, title=None):
    """Bordered box that groups interactive widgets, so visitors notice them."""
    box = st.container(border=True, key=f"ctl_{key}")
    box.markdown(f'<div class="ctl-title">👉 {title or t(COMMON, "controls")}</div>', unsafe_allow_html=True)
    return box


def kpis(items):
    """items: list of (label, value, sub-text[, colour]) with colour in '', 'orange', 'green', 'violet'."""
    for col, item in zip(st.columns(len(items)), items):
        label, value, sub = item[:3]
        color = item[3] if len(item) > 3 else ""
        col.markdown(f'<div class="kpi {color}"><div class="kpi-label">{label}</div>'
                     f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                     unsafe_allow_html=True)


def _caller_key(prefix):
    """Stable, unique key from the calling page and line (gives the CSS class st-key-<key>)."""
    frame = inspect.stack()[2]
    return f"{prefix}_{os.path.basename(frame.filename).split('.')[0]}_{frame.lineno}"


def show_code(code):
    with st.expander(f"</> {t(COMMON, 'code')}", key=_caller_key("xcode")):
        st.code(code, language="python")


def tech():
    """Collapsed section for content aimed at data scientists."""
    return st.expander(f"🔬 {t(COMMON, 'tech')}", key=_caller_key("xtech"))


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


def params_text(params):
    """{'model__estimator__alpha': 10} -> 'alpha=10'."""
    return ", ".join(f"{k.split('__')[-1]}={v}" for k, v in params.items())


# ── Charts ──────────────────────────────────────────────────────────────────
def style(chart, height=320):
    """Recessive axes and grid, readable text, consistent size."""
    return (chart.properties(height=height)
            .configure(background=SURFACE, font="Source Sans Pro, sans-serif")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=TEXT_2, titleColor=TEXT_2, gridColor=GRID, domainColor=GRID,
                            tickColor=GRID, labelFontSize=13, titleFontSize=13, titleFontWeight="normal",
                            titlePadding=10, labelLimit=260)
            .configure_legend(labelColor=TEXT_2, titleColor=TEXT_2, labelFontSize=13, titleFontSize=13,
                              orient="top")
            .configure_header(labelColor=TEXT, labelFontSize=14, titleFontSize=14)
            .configure_title(color=TEXT, fontSize=15, anchor="start", fontWeight=600))


def draw(chart, height=320):
    st.altair_chart(style(chart, height), width="stretch", theme=None)


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
    heat = alt.Chart(cm).mark_rect(cornerRadius=6, stroke=SURFACE, strokeWidth=3).encode(
        x=alt.X(f"{predicted}:N", sort=labels, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y(f"{actual}:N", sort=labels),
        color=alt.Color("n:Q", scale=alt.Scale(range=["#cde2fb", "#184f95"]), legend=None),
        tooltip=["group", actual, predicted, "n"])
    text = heat.mark_text(fontSize=20, fontWeight=700).encode(
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
