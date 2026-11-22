import altair as alt
import pandas as pd
import streamlit as st

from churn.prediction import predict
from common import (
    MODEL_NAMES, TEXT, brief, controls, draw, finding, header, highlight, metric_name, metrics, model, num,
    params_text, pct, require_artifacts, setup_page, show_code,
)
from i18n import get_lang, t

setup_page()
require_artifacts()

STRINGS = {
    "en": {
        "eyebrow": "Modelling · 6", "title": "Try the model",
        "caption": "Describe a customer to estimate their churn risk.",
        "brief": (
            "Fill in the profile of a customer, or load one of the two examples, and the model estimates the "
            "**probability that this customer leaves**. Above the decision threshold, the customer would be passed "
            "on to the retention team."
        ),
        "examples": "Load an example", "risky": "Customer at risk", "loyal": "Loyal customer",
        "profile": "Profile", "services": "Services", "account": "Contract and billing",
        "senior": "Senior citizen (65 or older)", "partner": "Has a partner", "dependents": "Has dependents",
        "tenure": "Tenure (months)", "monthly": "Monthly charges", "total": "Total charges",
        "auto_total": "Total charges = tenure × monthly charges ({v})", "edit_total": "Enter the total charges manually",
        "choose_model": "Model used", "selected_tag": " (selected model)",
        "result": "Churn probability",
        "high": "**High risk**: above the decision threshold of {model} ({thr}). This customer would be flagged for the retention team.",
        "low": "**Low risk**: below the decision threshold of {model} ({thr}).",
        "all_models": "Probability given by each model", "probability": "Churn probability",
        "threshold_note": "Black tick: decision threshold of each model · dashed line: overall churn rate ({rate}).",
        "compare_note": (
            "The probabilities **are not directly comparable from one model to another**: models trained with "
            "resampling (such as SGD with SMOTE) give higher probabilities, and their threshold is higher to "
            "compensate. What counts is the position of each bar relative to its own black tick."
        ),
        "model_used": "Model used: preparation, hyperparameters and threshold",
        "model_used_text": (
            "The customer goes through the same pipeline as the training data: imputation, transformation, "
            "encoding and scaling, then the model computes a probability, compared with the threshold."
        ),
        "step": "Step", "value": "Value", "imputer": "Missing values", "transform": "Numeric transformation",
        "scaler": "Scaler", "imbalance": "Class imbalance", "params": "Hyperparameters", "threshold": "Decision threshold",
        "cv_score": "Cross-validation score ({metric})",
        "trees_note": "Tree-based model: the scaler has no effect on its predictions.",
    },
    "fr": {
        "eyebrow": "Modélisation · 6", "title": "Tester le modèle",
        "caption": "Décrivez un client pour estimer son risque de churn.",
        "brief": (
            "Renseignez le profil d'un client, ou chargez l'un des deux exemples, et le modèle estime la "
            "**probabilité que ce client parte**. Au-dessus du seuil de décision, le client serait transmis à "
            "l'équipe de fidélisation."
        ),
        "examples": "Charger un exemple", "risky": "Client à risque", "loyal": "Client fidèle",
        "profile": "Profil", "services": "Services", "account": "Contrat et facturation",
        "senior": "Senior (65 ans ou plus)", "partner": "A un conjoint", "dependents": "A des personnes à charge",
        "tenure": "Ancienneté (mois)", "monthly": "Montant mensuel", "total": "Montant total",
        "auto_total": "Montant total = ancienneté × montant mensuel ({v})", "edit_total": "Saisir le montant total manuellement",
        "choose_model": "Modèle utilisé", "selected_tag": " (modèle retenu)",
        "result": "Probabilité de churn",
        "high": "**Risque élevé** : au-dessus du seuil de décision de {model} ({thr}). Ce client serait signalé à l'équipe de fidélisation.",
        "low": "**Risque faible** : en dessous du seuil de décision de {model} ({thr}).",
        "all_models": "Probabilité donnée par chaque modèle", "probability": "Probabilité de churn",
        "threshold_note": "Trait noir : seuil de décision de chaque modèle · ligne pointillée : taux de churn global ({rate}).",
        "compare_note": (
            "Les probabilités **ne se comparent pas directement d'un modèle à l'autre** : les modèles entraînés avec "
            "un rééchantillonnage (comme SGD avec SMOTE) donnent des probabilités plus élevées, et leur seuil est "
            "plus haut pour compenser. Ce qui compte, c'est la position de chaque barre par rapport à son propre trait noir."
        ),
        "model_used": "Modèle utilisé : préparation, hyperparamètres et seuil",
        "model_used_text": (
            "Le client passe par le même pipeline que les données d'entraînement : imputation, transformation, "
            "encodage et mise à l'échelle, puis le modèle calcule une probabilité, comparée au seuil."
        ),
        "step": "Étape", "value": "Valeur", "imputer": "Valeurs manquantes", "transform": "Transformation numérique",
        "scaler": "Scaler", "imbalance": "Déséquilibre des classes", "params": "Hyperparamètres", "threshold": "Seuil de décision",
        "cv_score": "Score en validation croisée ({metric})",
        "trees_note": "Modèle à base d'arbres : le scaler n'a aucun effet sur ses prédictions.",
    },
}

LABELS = {
    "en": {"gender": "Gender", "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
           "InternetService": "Internet service", "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
           "DeviceProtection": "Device protection", "TechSupport": "Tech support", "StreamingTV": "TV streaming",
           "StreamingMovies": "Movie streaming", "Contract": "Contract", "PaperlessBilling": "Paperless billing",
           "PaymentMethod": "Payment method"},
    "fr": {"gender": "Genre", "PhoneService": "Service téléphonique", "MultipleLines": "Plusieurs lignes",
           "InternetService": "Service internet", "OnlineSecurity": "Sécurité en ligne", "OnlineBackup": "Sauvegarde en ligne",
           "DeviceProtection": "Protection de l'appareil", "TechSupport": "Support technique", "StreamingTV": "Streaming TV",
           "StreamingMovies": "Streaming films", "Contract": "Contrat", "PaperlessBilling": "Facture dématérialisée",
           "PaymentMethod": "Moyen de paiement"},
}
VALUES_FR = {
    "Yes": "Oui", "No": "Non", "Female": "Femme", "Male": "Homme", "No internet service": "Pas d'internet",
    "No phone service": "Pas de téléphone", "Fiber optic": "Fibre optique", "DSL": "ADSL",
    "Month-to-month": "Au mois", "One year": "Un an", "Two year": "Deux ans",
    "Electronic check": "Chèque électronique", "Mailed check": "Chèque postal",
    "Bank transfer (automatic)": "Virement (automatique)", "Credit card (automatic)": "Carte bancaire (automatique)",
}
SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
EXAMPLES = {
    "risky": {"gender": "Female", "senior": True, "partner": False, "dependents": False, "PhoneService": "Yes",
              "MultipleLines": "No", "InternetService": "Fiber optic", **{s: "No" for s in SERVICES},
              "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
              "tenure": 2, "monthly": 85.0},
    "loyal": {"gender": "Male", "senior": False, "partner": True, "dependents": True, "PhoneService": "Yes",
              "MultipleLines": "Yes", "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
              "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
              "Contract": "Two year", "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
              "tenure": 60, "monthly": 65.0},
}


def label(col):
    return LABELS[get_lang()][col]


def show(value):
    return VALUES_FR.get(value, value) if get_lang() == "fr" else value


def choice(col, options, container=st, horizontal=True):
    """Radio whose stored value is reset if it is no longer a valid option (e.g. after removing internet)."""
    key = f"pred_{col}"
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return container.radio(label(col), options, key=key, horizontal=horizontal, format_func=show)


def load_example(name):
    for k, v in EXAMPLES[name].items():
        st.session_state[f"pred_{k}"] = v
    st.session_state["pred_edit_total"] = False


# Defaults set through session state only, so the example buttons can overwrite them
st.session_state.setdefault("pred_tenure", 6)
st.session_state.setdefault("pred_monthly", 80.0)

m = metrics()
names = list(m["models"])
header(t(STRINGS, "eyebrow"), t(STRINGS, "title"), t(STRINGS, "caption"))
brief(t(STRINGS, "brief"))

with controls("examples", t(STRINGS, "examples")):
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button(t(STRINGS, "risky"), on_click=load_example, args=("risky",), icon=":material/warning:", width="stretch")
    b2.button(t(STRINGS, "loyal"), on_click=load_example, args=("loyal",), icon=":material/favorite:", width="stretch")

form, result = st.columns([1.55, 1], gap="large")
yes_no = ["No", "Yes"]
with form:
    with controls("form", t(STRINGS, "profile")):
        c1, c2 = st.columns(2)
        gender = choice("gender", ["Female", "Male"], c1)
        senior = c2.toggle(t(STRINGS, "senior"), key="pred_senior")
        partner = c2.toggle(t(STRINGS, "partner"), key="pred_partner")
        dependents = c2.toggle(t(STRINGS, "dependents"), key="pred_dependents")
    with controls("services", t(STRINGS, "services")):
        c1, c2 = st.columns(2)
        phone = choice("PhoneService", ["Yes", "No"], c1)
        multiple = choice("MultipleLines", yes_no if phone == "Yes" else ["No phone service"], c2)
        internet = choice("InternetService", ["Fiber optic", "DSL", "No"], c1)
        opts = yes_no if internet != "No" else ["No internet service"]
        services = {s: choice(s, opts, (c2 if i % 2 == 0 else c1)) for i, s in enumerate(SERVICES)}
    with controls("account", t(STRINGS, "account")):
        c1, c2 = st.columns(2)
        contract = c1.selectbox(label("Contract"), ["Month-to-month", "One year", "Two year"], key="pred_Contract", format_func=show)
        payment = c2.selectbox(label("PaymentMethod"), ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                                        "Credit card (automatic)"], key="pred_PaymentMethod", format_func=show)
        paperless = choice("PaperlessBilling", yes_no, c1)
        tenure = c1.slider(t(STRINGS, "tenure"), 0, 72, key="pred_tenure")
        monthly = c2.number_input(t(STRINGS, "monthly"), 18.0, 120.0, step=0.5, key="pred_monthly")
        if c2.checkbox(t(STRINGS, "edit_total"), key="pred_edit_total"):
            total = c2.number_input(t(STRINGS, "total"), 0.0, 9000.0, float(round(tenure * monthly, 2)), 10.0)
        else:
            total = round(tenure * monthly, 2)
            c2.caption(t(STRINGS, "auto_total", v=num(total, 2)))

customer = {
    "gender": gender, "SeniorCitizen": int(senior), "Partner": yes_no[partner], "Dependents": yes_no[dependents],
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet, **services,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total,
}
results = pd.DataFrame([{"name": n, "model": MODEL_NAMES[n], "probability": predict([customer], model(n)).iloc[0]["churn_probability"],
                         "threshold": m["models"][n]["threshold"]} for n in names])

with result:
    st.markdown('<div class="sticky-anchor"></div>', unsafe_allow_html=True)
    with controls("model", t(STRINGS, "choose_model")):
        sel = st.selectbox(t(STRINGS, "choose_model"), names, index=names.index(m["selected_model"]), label_visibility="collapsed",
                           format_func=lambda n: MODEL_NAMES[n] + (t(STRINGS, "selected_tag") if n == m["selected_model"] else ""))
    row = results.set_index("name").loc[sel]
    high = row.probability >= row.threshold
    color = "#eb6834" if high else "#1baf7a"
    st.markdown(f'<div class="kpi" style="border-top-color:{color}"><div class="kpi-label">{t(STRINGS, "result")}</div>'
                f'<div class="kpi-value" style="font-size:3.4rem;color:{color}">{pct(row.probability, 0)}</div>'
                f'<div class="kpi-sub">{MODEL_NAMES[sel]}</div></div>', unsafe_allow_html=True)
    finding(t(STRINGS, "high" if high else "low", model=MODEL_NAMES[sel], thr=pct(row.threshold, 0)))

    st.markdown(f"**{t(STRINGS, 'all_models')}**")
    bars = alt.Chart(results).mark_bar(cornerRadiusEnd=4, size=20).encode(
        y=alt.Y("model:N", title=None, sort=None),
        x=alt.X("probability:Q", title=t(STRINGS, "probability"), scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=highlight("name", sel),
        tooltip=["model", alt.Tooltip("probability:Q", format=".1%"), alt.Tooltip("threshold:Q", format=".2f")])
    ticks = alt.Chart(results).mark_tick(color=TEXT, thickness=3, size=28).encode(y=alt.Y("model:N", sort=None), x="threshold:Q")
    rule = alt.Chart(pd.DataFrame({"x": [m["churn_rate"]]})).mark_rule(color=TEXT, strokeDash=[5, 4], strokeWidth=1.5).encode(x="x:Q")
    labels = bars.mark_text(align="left", dx=6, fontSize=13).encode(color=alt.value(TEXT), text=alt.Text("probability:Q", format=".0%"))
    draw(bars + ticks + rule + labels, 230)
    st.caption(t(STRINGS, "threshold_note", rate=pct(m["churn_rate"])))
    finding(t(STRINGS, "compare_note"))

info = m["models"][sel]
with st.expander(t(STRINGS, "model_used")):
    st.markdown(t(STRINGS, "model_used_text"))
    cfg = info["config"]
    st.dataframe(pd.DataFrame({
        t(STRINGS, "step"): [t(STRINGS, k) for k in ("imputer", "transform", "scaler", "imbalance", "params", "threshold")]
                            + [t(STRINGS, "cv_score", metric=metric_name(m["selection_metric"]))],
        t(STRINGS, "value"): [cfg["imputer"], cfg["transform"], cfg["scaler"], cfg["imbalance"], params_text(info["params"]),
                              num(info["threshold"], 2), num(info["cv_score"], 3)],
    }), hide_index=True, width="stretch")
    if sel in ("random_forest", "hist_gradient_boosting", "adaboost"):
        st.caption(t(STRINGS, "trees_note"))

show_code("from churn.prediction import load_model, predict\n\n"
          f"model = load_model('{sel}')\npredict([customer], model)   # -> churn_probability, churn")
