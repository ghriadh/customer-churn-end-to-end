"""Entry point: page config, language toggle, glossary, navigation and the home page.

Run from the project root:  streamlit run app/Home.py
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Customer Churn", page_icon="📉", layout="wide")

from common import (  # noqa: E402
    MODEL_NAMES, artifacts_ready, brief, cv_results, eda_data, finding, kpis, metric_name, metrics, num, pct,
    setup_page,
)
from i18n import render_lang_toggle, t  # noqa: E402

render_lang_toggle()
setup_page()

STRINGS = {
    "en": {
        "nav_home": "Home", "nav_data": "Data", "nav_eda": "Exploration", "nav_prep": "Preparation",
        "nav_compare": "Model comparison", "nav_eval": "Results", "nav_predict": "Try the model",
        "section_analysis": "Analysis", "section_model": "Modelling",
        "eyebrow": "Machine learning project",
        "title": "Which customers are about to leave?",
        "lead": (
            "A telecom operator loses about one customer in four. This project predicts which customers are likely "
            "to cancel their subscription, so the company can act before they leave."
        ),
        "author": "Riadh Ghanmi · {n} customers · 19 features · binary classification",
        "brief": (
            "Out of **100 customers who are about to leave**, the model spots **{caught}** of them. When it flags a "
            "customer, it is right **{right} times out of 100**. Without a model, the company cannot tell which "
            "customers are at risk. The main reasons for leaving are clear: short contracts, the first months, "
            "fiber optic and missing support services."
        ),
        "kpi_recall": "Churners detected", "kpi_recall_sub": "recall · no model: 0%",
        "kpi_precision": "Correct alerts", "kpi_precision_sub": "precision of the alerts",
        "kpi_auc": "ROC-AUC", "kpi_auc_sub": "ranking quality · random: 0.50",
        "kpi_models": "Combinations tested", "kpi_models_sub": "6 models × preparation methods",
        "selected": "Final model: **{model}**, selected for the best {metric} in cross-validation, decision threshold {thr}.",
        "insights_title": "What drives churn",
        "insights": (
            "- **Contract**: {m2m} of month-to-month customers leave, against {one} (one year) and {two} (two years).\n"
            "- **First months**: {early} of the customers who leave do so during their first year.\n"
            "- **Fiber optic** customers ({fiber} churn) and **electronic check** payers ({echeck}) are the most exposed.\n"
            "- Customers **without online security** churn at {sec_no}, against {sec_yes} with it."
        ),
        "journey_title": "Explore the project",
        "journey_hint": "Follow the steps in order, or jump straight to the results.",
        "card_data": "The raw dataset, its columns and its cleaning.",
        "card_eda": "Distributions, statistical tests and the link between each feature and churn.",
        "card_prep": "Each step of the pipeline, with interactive illustrations.",
        "card_compare": "{n} combinations of methods and models, compared fairly.",
        "card_eval": "Results on unseen customers, decision threshold and business recommendations.",
        "card_predict": "Describe a customer and get their churn risk.",
        "hurry": "In a hurry? Go straight to the **results and recommendations**:",
        "no_results": "Results will appear here once the models are trained (`python run_pipeline.py`).",
        "glossary": "Glossary",
        "glossary_text": (
            "**Churn**: a customer who cancels their subscription.\n\n"
            "**Recall**: share of the customers who leave that the model detects.\n\n"
            "**Precision**: share of the flagged customers who really leave.\n\n"
            "**F1**: a single score that balances recall and precision.\n\n"
            "**ROC-AUC / PR-AUC**: how well the model ranks customers from most to least at risk (1 = perfect).\n\n"
            "**Decision threshold**: probability above which a customer is flagged.\n\n"
            "**Cross-validation**: the training data is split into 5 parts; each part is used in turn to check a "
            "model trained on the other 4.\n\n"
            "**Test set**: 20% of customers set aside at the start, used only once to measure the final result."
        ),
    },
    "fr": {
        "nav_home": "Accueil", "nav_data": "Données", "nav_eda": "Exploration", "nav_prep": "Préparation",
        "nav_compare": "Comparaison des modèles", "nav_eval": "Résultats", "nav_predict": "Tester le modèle",
        "section_analysis": "Analyse", "section_model": "Modélisation",
        "eyebrow": "Projet de machine learning",
        "title": "Quels clients sont sur le point de partir ?",
        "lead": (
            "Un opérateur télécom perd environ un client sur quatre. Ce projet prédit quels clients risquent de "
            "résilier leur abonnement, pour que l'entreprise puisse agir avant leur départ."
        ),
        "author": "Riadh Ghanmi · {n} clients · 19 variables · classification binaire",
        "brief": (
            "Sur **100 clients sur le point de partir**, le modèle en repère **{caught}**. Quand il signale un client, "
            "il a raison **{right} fois sur 100**. Sans modèle, l'entreprise ne sait pas quels clients sont à risque. "
            "Les principales raisons de départ sont claires : contrats courts, premiers mois, fibre optique et "
            "absence de services d'assistance."
        ),
        "kpi_recall": "Churners détectés", "kpi_recall_sub": "rappel · sans modèle : 0 %",
        "kpi_precision": "Alertes justes", "kpi_precision_sub": "précision des alertes",
        "kpi_auc": "ROC-AUC", "kpi_auc_sub": "qualité du classement · hasard : 0,50",
        "kpi_models": "Combinaisons testées", "kpi_models_sub": "6 modèles × méthodes de préparation",
        "selected": "Modèle final : **{model}**, retenu pour la meilleure {metric} en validation croisée, seuil de décision {thr}.",
        "insights_title": "Ce qui explique le churn",
        "insights": (
            "- **Contrat** : {m2m} des clients au mois partent, contre {one} (un an) et {two} (deux ans).\n"
            "- **Premiers mois** : {early} des clients qui partent le font pendant leur première année.\n"
            "- Les clients **fibre optique** ({fiber} de churn) et ceux qui paient par **chèque électronique** ({echeck}) sont les plus exposés.\n"
            "- Les clients **sans sécurité en ligne** partent à {sec_no}, contre {sec_yes} avec."
        ),
        "journey_title": "Explorer le projet",
        "journey_hint": "Suivez les étapes dans l'ordre, ou allez directement aux résultats.",
        "card_data": "Le jeu de données brut, ses colonnes et son nettoyage.",
        "card_eda": "Distributions, tests statistiques et lien entre chaque variable et le churn.",
        "card_prep": "Chaque étape du pipeline, avec des illustrations interactives.",
        "card_compare": "{n} combinaisons de méthodes et de modèles, comparées équitablement.",
        "card_eval": "Résultats sur des clients jamais vus, seuil de décision et recommandations métier.",
        "card_predict": "Décrivez un client et obtenez son risque de churn.",
        "hurry": "Pressé ? Allez directement aux **résultats et recommandations** :",
        "no_results": "Les résultats apparaîtront ici une fois les modèles entraînés (`python run_pipeline.py`).",
        "glossary": "Glossaire",
        "glossary_text": (
            "**Churn** : un client qui résilie son abonnement.\n\n"
            "**Rappel** : part des clients qui partent que le modèle détecte.\n\n"
            "**Précision** : part des clients signalés qui partent vraiment.\n\n"
            "**F1** : un score unique qui équilibre rappel et précision.\n\n"
            "**ROC-AUC / PR-AUC** : capacité du modèle à classer les clients du plus au moins risqué (1 = parfait).\n\n"
            "**Seuil de décision** : probabilité au-dessus de laquelle un client est signalé.\n\n"
            "**Validation croisée** : les données d'entraînement sont découpées en 5 parties ; chaque partie sert à "
            "tour de rôle à vérifier un modèle entraîné sur les 4 autres.\n\n"
            "**Jeu de test** : 20 % des clients mis de côté au départ, utilisés une seule fois pour mesurer le résultat final."
        ),
    },
}

pages = Path(__file__).parent / "pages"
PAGES = {
    "data": st.Page(str(pages / "1_Data.py"), title=t(STRINGS, "nav_data"), icon=":material/table_view:"),
    "eda": st.Page(str(pages / "2_Exploration.py"), title=t(STRINGS, "nav_eda"), icon=":material/query_stats:"),
    "prep": st.Page(str(pages / "3_Preparation.py"), title=t(STRINGS, "nav_prep"), icon=":material/build:"),
    "compare": st.Page(str(pages / "4_Model_comparison.py"), title=t(STRINGS, "nav_compare"), icon=":material/leaderboard:"),
    "eval": st.Page(str(pages / "5_Evaluation.py"), title=t(STRINGS, "nav_eval"), icon=":material/verified:"),
    "predict": st.Page(str(pages / "6_Prediction.py"), title=t(STRINGS, "nav_predict"), icon=":material/person_search:"),
}

with st.sidebar:
    with st.expander(f"📖 {t(STRINGS, 'glossary')}"):
        st.markdown(t(STRINGS, "glossary_text"))


def home():
    df = eda_data()
    n_combos = len(cv_results()) if artifacts_ready() else 83
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{t(STRINGS, "eyebrow")}</div><h1>{t(STRINGS, "title")}</h1>'
        f'<div class="lead">{t(STRINGS, "lead")}</div>'
        f'<div class="author">{t(STRINGS, "author", n=num(len(df)))}</div></div>', unsafe_allow_html=True)

    if artifacts_ready():
        m = metrics()
        sel = m["selected_model"]
        test = m["models"][sel]["test"]
        brief(t(STRINGS, "brief", caught=round(test["recall"] * 100), right=round(test["precision"] * 100)))
        kpis([
            (t(STRINGS, "kpi_recall"), pct(test["recall"], 0), t(STRINGS, "kpi_recall_sub"), "orange"),
            (t(STRINGS, "kpi_precision"), pct(test["precision"], 0), t(STRINGS, "kpi_precision_sub"), "green"),
            (t(STRINGS, "kpi_auc"), num(test["roc_auc"], 2), t(STRINGS, "kpi_auc_sub")),
            (t(STRINGS, "kpi_models"), str(n_combos), t(STRINGS, "kpi_models_sub"), "violet"),
        ])
        st.write("")
        st.markdown(t(STRINGS, "selected", model=MODEL_NAMES[sel], metric=metric_name(m["selection_metric"]),
                      thr=num(m["models"][sel]["threshold"], 2)))
    else:
        st.info(t(STRINGS, "no_results"))

    churn = df["Churn"] == "Yes"
    rate = lambda mask: pct(churn[mask].mean(), 0)  # noqa: E731
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.subheader(t(STRINGS, "insights_title"))
        finding(t(STRINGS, "insights",
                  m2m=rate(df.Contract == "Month-to-month"), one=rate(df.Contract == "One year"),
                  two=rate(df.Contract == "Two year"), early=pct((df.loc[churn, "tenure"] < 12).mean(), 0),
                  fiber=rate(df.InternetService == "Fiber optic"), echeck=rate(df.PaymentMethod == "Electronic check"),
                  sec_no=rate(df.OnlineSecurity == "No"), sec_yes=rate(df.OnlineSecurity == "Yes")))
    with right:
        st.subheader(t(STRINGS, "journey_title"))
        st.caption(t(STRINGS, "journey_hint"))
        cards = [("data", "card_data"), ("eda", "card_eda"), ("prep", "card_prep"),
                 ("compare", "card_compare"), ("eval", "card_eval"), ("predict", "card_predict")]
        for row in range(0, len(cards), 2):
            for col, (i, (key, text)) in zip(st.columns(2), enumerate(cards[row:row + 2], start=row + 1)):
                with col.container(border=True):
                    st.page_link(PAGES[key], label=f"**{i}. {PAGES[key].title}**", icon=PAGES[key].icon)
                    st.caption(t(STRINGS, text, n=n_combos))
        st.markdown(t(STRINGS, "hurry"))
        st.page_link(PAGES["eval"], label=f"**{PAGES['eval'].title} →**", icon=":material/verified:")


nav = st.navigation({
    "": [st.Page(home, title=t(STRINGS, "nav_home"), icon=":material/home:", default=True)],
    t(STRINGS, "section_analysis"): [PAGES["data"], PAGES["eda"]],
    t(STRINGS, "section_model"): [PAGES["prep"], PAGES["compare"], PAGES["eval"], PAGES["predict"]],
})
nav.run()
