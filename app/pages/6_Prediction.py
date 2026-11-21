import altair as alt
import pandas as pd
import streamlit as st

from churn.prediction import predict
from common import (
    MODEL_NAMES, draw, finding, header, metrics, model, model_scale, pct, require_artifacts, setup_page, show_code,
)
from i18n import t

setup_page()
require_artifacts()

STRINGS = {
    "en": {
        "eyebrow": "Modelling · 6", "title": "Prediction",
        "caption": "Describe a customer to estimate their churn risk.",
        "profile": "Profile", "services": "Services", "account": "Account",
        "senior": "Senior citizen (65 or older)", "partner": "Has a partner", "dependents": "Has dependents",
        "tenure": "Tenure (months)", "monthly": "Monthly charges", "total": "Total charges",
        "total_help": "Pre-filled with tenure × monthly charges; adjust if needed.",
        "predict": "Estimate the churn risk",
        "result": "Churn probability ({model})",
        "high": "**High risk**: the probability exceeds the decision threshold ({thr:.2f}). This customer would be flagged for the retention team.",
        "low": "**Low risk**: the probability is below the decision threshold ({thr:.2f}).",
        "all_models": "Probability given by each model", "probability": "Churn probability",
        "threshold_note": "Black tick: decision threshold of each model · dashed line: overall churn rate ({rate}).",
    },
    "fr": {
        "eyebrow": "Modélisation · 6", "title": "Prédiction",
        "caption": "Décrivez un client pour estimer son risque de churn.",
        "profile": "Profil", "services": "Services", "account": "Compte",
        "senior": "Senior (65 ans ou plus)", "partner": "A un conjoint", "dependents": "A des personnes à charge",
        "tenure": "Ancienneté (mois)", "monthly": "Montant mensuel", "total": "Montant total",
        "total_help": "Prérempli avec ancienneté × montant mensuel ; ajustez si besoin.",
        "predict": "Estimer le risque de churn",
        "result": "Probabilité de churn ({model})",
        "high": "**Risque élevé** : la probabilité dépasse le seuil de décision ({thr:.2f}). Ce client serait signalé à l'équipe de fidélisation.",
        "low": "**Risque faible** : la probabilité est inférieure au seuil de décision ({thr:.2f}).",
        "all_models": "Probabilité donnée par chaque modèle", "probability": "Probabilité de churn",
        "threshold_note": "Trait noir : seuil de décision de chaque modèle · ligne pointillée : taux de churn global ({rate}).",
    },
}

m = metrics()
header(t(STRINGS, "eyebrow"), t(STRINGS, "title"), t(STRINGS, "caption"))
yes_no = ["No", "Yes"]

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown(f"**{t(STRINGS, 'profile')}**")
    gender = st.radio("gender", ["Female", "Male"], horizontal=True)
    senior = st.toggle(t(STRINGS, "senior"))
    partner = st.toggle(t(STRINGS, "partner"))
    dependents = st.toggle(t(STRINGS, "dependents"))
with c2:
    st.markdown(f"**{t(STRINGS, 'services')}**")
    phone = st.radio("PhoneService", yes_no, index=1, horizontal=True)
    multiple = st.radio("MultipleLines", ["No", "Yes"] if phone == "Yes" else ["No phone service"], horizontal=True)
    internet = st.radio("InternetService", ["DSL", "Fiber optic", "No"], index=1, horizontal=True)
    opts = yes_no if internet != "No" else ["No internet service"]
    services = {s: st.radio(s, opts, horizontal=True) for s in
                ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]}
with c3:
    st.markdown(f"**{t(STRINGS, 'account')}**")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.radio("PaperlessBilling", yes_no, index=1, horizontal=True)
    payment = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                             "Credit card (automatic)"])
    tenure = st.slider(t(STRINGS, "tenure"), 0, 72, 6)
    monthly = st.number_input(t(STRINGS, "monthly"), 18.0, 120.0, 80.0, 0.5)
    total = st.number_input(t(STRINGS, "total"), 0.0, 9000.0, float(round(tenure * monthly, 2)), 10.0,
                            help=t(STRINGS, "total_help"))

customer = {
    "gender": gender, "SeniorCitizen": int(senior), "Partner": yes_no[partner], "Dependents": yes_no[dependents],
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet, **services,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total,
}

st.divider()
names = list(m["models"])
results = pd.DataFrame([{"name": n, "model": MODEL_NAMES[n], "probability": predict([customer], model(n)).iloc[0]["churn_probability"],
                         "threshold": m["models"][n]["threshold"]} for n in names])
sel = m["selected_model"]
row = results.set_index("name").loc[sel]

left, right = st.columns([1, 1.5], gap="large")
with left:
    st.markdown(f"**{t(STRINGS, 'result', model=MODEL_NAMES[sel])}**")
    st.markdown(f'<div class="kpi-value" style="font-size:44px">{pct(row.probability, 0)}</div>', unsafe_allow_html=True)
    finding(t(STRINGS, "high" if row.probability >= row.threshold else "low", thr=row.threshold))
with right:
    st.markdown(f"**{t(STRINGS, 'all_models')}**")
    bars = alt.Chart(results).mark_bar(cornerRadiusEnd=4, size=22).encode(
        y=alt.Y("model:N", title=None, sort=None),
        x=alt.X("probability:Q", title=t(STRINGS, "probability"), scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("model:N", scale=model_scale(names), legend=None),
        tooltip=["model", alt.Tooltip("probability:Q", format=".1%"), alt.Tooltip("threshold:Q", format=".2f")])
    ticks = alt.Chart(results).mark_tick(color="#0b0b0b", thickness=2, size=30).encode(y=alt.Y("model:N", sort=None), x="threshold:Q")
    rule = alt.Chart(pd.DataFrame({"x": [m["churn_rate"]]})).mark_rule(color="#a3a29c", strokeDash=[4, 4]).encode(x="x:Q")
    labels = bars.mark_text(align="left", dx=6, color="#0b0b0b").encode(text=alt.Text("probability:Q", format=".0%"))
    draw(bars + ticks + rule + labels, 200)
    st.caption(t(STRINGS, "threshold_note", rate=pct(m["churn_rate"])))

show_code("from churn.prediction import load_model, predict\n\n"
          f"model = load_model('{sel}')\npredict([customer], model)   # -> churn_probability, churn")
