"""Entry point: page config, language toggle, navigation and the home page.

Run from the project root:  streamlit run app/Home.py
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Customer Churn", page_icon="📉", layout="wide")

from common import MODEL_NAMES, artifacts_ready, cv_results, finding, header, kpis, metric_name, metrics, pct, setup_page  # noqa: E402
from i18n import render_lang_toggle, t  # noqa: E402

render_lang_toggle()
setup_page()

STRINGS = {
    "en": {
        "nav_home": "Home", "nav_data": "Data", "nav_eda": "Exploration", "nav_prep": "Preparation",
        "nav_compare": "Model comparison", "nav_eval": "Final evaluation", "nav_predict": "Prediction",
        "section_analysis": "Analysis", "section_model": "Modelling",
        "eyebrow": "Machine learning project",
        "title": "Customer churn prediction",
        "caption": "Telecom customers · 7,043 customers · 19 features · binary classification",
        "author": "Author: Riadh Ghanmi",
        "problem_title": "The problem",
        "problem": (
            "A telecommunications company loses about **one customer in four**. Keeping a customer costs far "
            "less than acquiring a new one, so the goal is to **identify customers likely to leave** early "
            "enough to act (retention offer, contract upgrade, support call).\n\n"
            "This project explores the data, compares several preparation methods and models in a single "
            "reproducible pipeline, and serves the final models in this app."
        ),
        "results_title": "Key results (test set, customers never seen during training)",
        "kpi_recall": "Churners detected", "kpi_recall_sub": "recall · baseline 0%",
        "kpi_precision": "Precision", "kpi_precision_sub": "of customers flagged really leave",
        "kpi_auc": "ROC-AUC", "kpi_auc_sub": "baseline 0.50",
        "kpi_f1": "F1", "kpi_f1_sub": "threshold tuned to {thr:.2f}",
        "selected": "Selected model: **{model}** (best {metric} in cross-validation).",
        "insights_title": "What drives churn",
        "insights": (
            "- **Contract**: 43% of month-to-month customers leave, against 11% (one year) and 3% (two years).\n"
            "- **Tenure**: 54% of churners leave during their first year.\n"
            "- **Fiber optic** customers (42% churn) and **electronic check** payers (45%) are the most exposed.\n"
            "- Customers **without online security or tech support** churn about 2.8 times more than those who have them."
        ),
        "tour_title": "How to read this app",
        "tour": (
            "1. **Data**: the raw dataset and its cleaning.\n"
            "2. **Exploration**: distributions, statistical tests and the links between each feature and churn.\n"
            "3. **Preparation**: every step of the pipeline and its options.\n"
            "4. **Model comparison**: {n} combinations of preparation methods and models compared with the same cross-validation.\n"
            "5. **Final evaluation**: results on the test set, decision threshold, feature importance and recommendations.\n"
            "6. **Prediction**: estimate the churn risk of a new customer."
        ),
        "no_results": "Results will appear here once the models are trained (`python run_pipeline.py`).",
    },
    "fr": {
        "nav_home": "Accueil", "nav_data": "Données", "nav_eda": "Exploration", "nav_prep": "Préparation",
        "nav_compare": "Comparaison des modèles", "nav_eval": "Évaluation finale", "nav_predict": "Prédiction",
        "section_analysis": "Analyse", "section_model": "Modélisation",
        "eyebrow": "Projet de machine learning",
        "title": "Prédiction du churn client",
        "caption": "Clients télécom · 7 043 clients · 19 variables · classification binaire",
        "author": "Auteur : Riadh Ghanmi",
        "problem_title": "Le problème",
        "problem": (
            "Un opérateur télécom perd environ **un client sur quatre**. Garder un client coûte bien moins cher "
            "que d'en conquérir un nouveau : l'objectif est donc d'**identifier les clients susceptibles de "
            "partir** assez tôt pour agir (offre de fidélisation, changement de contrat, appel du support).\n\n"
            "Ce projet explore les données, compare plusieurs méthodes de préparation et plusieurs modèles dans "
            "un pipeline unique et reproductible, puis met les modèles finaux à disposition dans cette application."
        ),
        "results_title": "Résultats clés (jeu de test, clients jamais vus pendant l'entraînement)",
        "kpi_recall": "Churners détectés", "kpi_recall_sub": "rappel · baseline 0 %",
        "kpi_precision": "Précision", "kpi_precision_sub": "des clients signalés partent vraiment",
        "kpi_auc": "ROC-AUC", "kpi_auc_sub": "baseline 0,50",
        "kpi_f1": "F1", "kpi_f1_sub": "seuil réglé à {thr:.2f}",
        "selected": "Modèle retenu : **{model}** (meilleure {metric} en validation croisée).",
        "insights_title": "Ce qui explique le churn",
        "insights": (
            "- **Contrat** : 43 % des clients au mois partent, contre 11 % (un an) et 3 % (deux ans).\n"
            "- **Ancienneté** : 54 % des churners partent pendant leur première année.\n"
            "- Les clients **fibre optique** (42 % de churn) et ceux qui paient par **chèque électronique** (45 %) sont les plus exposés.\n"
            "- Les clients **sans sécurité en ligne ni support technique** partent environ 2,8 fois plus que ceux qui en ont."
        ),
        "tour_title": "Comment lire cette application",
        "tour": (
            "1. **Données** : le jeu de données brut et son nettoyage.\n"
            "2. **Exploration** : distributions, tests statistiques et lien entre chaque variable et le churn.\n"
            "3. **Préparation** : chaque étape du pipeline et ses options.\n"
            "4. **Comparaison des modèles** : {n} combinaisons de méthodes de préparation et de modèles comparées avec la même validation croisée.\n"
            "5. **Évaluation finale** : résultats sur le jeu de test, seuil de décision, importance des variables et recommandations.\n"
            "6. **Prédiction** : estimer le risque de churn d'un nouveau client."
        ),
        "no_results": "Les résultats apparaîtront ici une fois les modèles entraînés (`python run_pipeline.py`).",
    },
}


def home():
    header(t(STRINGS, "eyebrow"), t(STRINGS, "title"), t(STRINGS, "caption"))
    st.caption(t(STRINGS, "author"))

    st.subheader(t(STRINGS, "problem_title"))
    st.markdown(t(STRINGS, "problem"))

    st.subheader(t(STRINGS, "results_title"))
    if artifacts_ready():
        m = metrics()
        sel = m["selected_model"]
        test = m["models"][sel]["test"]
        kpis([
            (t(STRINGS, "kpi_recall"), pct(test["recall"], 0), t(STRINGS, "kpi_recall_sub")),
            (t(STRINGS, "kpi_precision"), pct(test["precision"], 0), t(STRINGS, "kpi_precision_sub")),
            (t(STRINGS, "kpi_auc"), f"{test['roc_auc']:.2f}", t(STRINGS, "kpi_auc_sub")),
            (t(STRINGS, "kpi_f1"), f"{test['f1']:.2f}",
             t(STRINGS, "kpi_f1_sub", thr=m["models"][sel]["threshold"])),
        ])
        st.write("")
        st.markdown(t(STRINGS, "selected", model=MODEL_NAMES[sel], metric=metric_name(m["selection_metric"])))
    else:
        st.info(t(STRINGS, "no_results"))

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(t(STRINGS, "insights_title"))
        finding(t(STRINGS, "insights"))
    with right:
        st.subheader(t(STRINGS, "tour_title"))
        st.markdown(t(STRINGS, "tour", n=len(cv_results()) if artifacts_ready() else "~80"))


pages = Path(__file__).parent / "pages"
nav = st.navigation({
    "": [st.Page(home, title=t(STRINGS, "nav_home"), icon="🏠", default=True)],
    t(STRINGS, "section_analysis"): [
        st.Page(str(pages / "1_Data.py"), title=t(STRINGS, "nav_data"), icon="🗂️"),
        st.Page(str(pages / "2_Exploration.py"), title=t(STRINGS, "nav_eda"), icon="🔎"),
    ],
    t(STRINGS, "section_model"): [
        st.Page(str(pages / "3_Preparation.py"), title=t(STRINGS, "nav_prep"), icon="🧰"),
        st.Page(str(pages / "4_Model_comparison.py"), title=t(STRINGS, "nav_compare"), icon="⚖️"),
        st.Page(str(pages / "5_Evaluation.py"), title=t(STRINGS, "nav_eval"), icon="🎯"),
        st.Page(str(pages / "6_Prediction.py"), title=t(STRINGS, "nav_predict"), icon="🔮"),
    ],
})
nav.run()
