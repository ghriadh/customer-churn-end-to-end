import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve

from churn.evaluation import test_metrics
from common import (
    CHURN_COLORS, MODEL_COLORS, MODEL_NAMES, METRICS, NEUTRAL, coefficients, draw, eda_data, finding, header,
    importances, kpis, metric_name, metrics, model_scale, pct, require_artifacts, setup_page, show_code,
    test_predictions,
)
from i18n import t

setup_page()
require_artifacts()

STRINGS = {
    "en": {
        "eyebrow": "Modelling · 5", "title": "Final evaluation",
        "caption": "Results on the test set: {n:,} customers set aside at the start and never used for training or model selection.",
        "choose_model": "Model", "selected_tag": " (selected)",
        "kpi_recall": "Recall", "kpi_recall_sub": "churners detected",
        "kpi_precision": "Precision", "kpi_precision_sub": "of alerts are real churners",
        "kpi_f1": "F1", "kpi_auc": "ROC-AUC", "kpi_ap": "PR-AUC", "baseline_sub": "baseline {v:.2f}",
        "all_models": "All final models on the test set (tuned threshold)",
        "threshold_title": "Decision threshold",
        "threshold_text": (
            "The model gives each customer a churn probability; a customer is flagged when the probability exceeds "
            "the threshold. The threshold tuned in cross-validation is **{thr:.2f}**. Move the slider to see the trade-off."
        ),
        "threshold": "Threshold",
        "cm_title": "Confusion matrix", "actual": "Actual", "predicted": "Predicted",
        "cm_text": (
            "With this threshold, the model detects **{tp}** of the **{pos}** churners ({rec}) and raises **{fp}** "
            "false alarms. **{fn}** churners are missed."
        ),
        "pr_title": "Precision and recall by threshold",
        "default_comment": (
            "With the default threshold of 0.5, the same model detects only **{r05}** of churners. Tuning the "
            "threshold raises recall to **{rt}** with a moderate loss of precision ({p05} → {pt}). For a retention "
            "campaign, missing a churner usually costs more than contacting a customer who would have stayed."
        ),
        "curves_title": "Precision-recall curves of the four models",
        "importance_title": "Which features matter",
        "importance_text": (
            "Permutation importance: drop in PR-AUC on the test set when the values of a feature are shuffled. "
            "It is computed the same way for every model, on the original features."
        ),
        "drop": "PR-AUC drop",
        "coef_title": "Direction of the effects (logistic regression coefficients)",
        "coef_text": "Positive coefficient = increases the churn risk; negative = reduces it (numeric features are transformed and scaled).",
        "coef": "Coefficient",
        "reco_title": "Business recommendations",
        "reco": (
            "1. **Target month-to-month contracts**: {m2m} of these customers leave, against {two} with a two-year "
            "contract. Offering a discount for switching to a one- or two-year contract is the strongest lever.\n"
            "2. **Take care of new customers**: {early} of churners leave during their first year. An onboarding "
            "programme for the first months (welcome call, check of the first bills) addresses the period of highest risk.\n"
            "3. **Promote online security and tech support**: customers without them churn {sec_no} against "
            "{sec_yes} with online security. Bundling these services into offers lowers the risk.\n"
            "4. **Look into the fiber optic offer**: {fiber} churn for fiber customers against {dsl} for DSL, despite "
            "a better service. Price and quality of the fiber offer deserve an analysis.\n"
            "5. **Encourage automatic payment**: {echeck} of customers paying by electronic check leave, against "
            "{auto} with automatic payment.\n"
            "6. **Use the model to prioritise**: with the tuned threshold, about {flag} of customers are flagged and "
            "{prec} of them really leave. The retention team can focus on this list, starting with the highest probabilities."
        ),
    },
    "fr": {
        "eyebrow": "Modélisation · 5", "title": "Évaluation finale",
        "caption": "Résultats sur le jeu de test : {n:,} clients mis de côté au départ et jamais utilisés pour l'entraînement ni pour le choix du modèle.",
        "choose_model": "Modèle", "selected_tag": " (retenu)",
        "kpi_recall": "Rappel", "kpi_recall_sub": "churners détectés",
        "kpi_precision": "Précision", "kpi_precision_sub": "des alertes sont de vrais churners",
        "kpi_f1": "F1", "kpi_auc": "ROC-AUC", "kpi_ap": "PR-AUC", "baseline_sub": "baseline {v:.2f}",
        "all_models": "Tous les modèles finaux sur le jeu de test (seuil réglé)",
        "threshold_title": "Seuil de décision",
        "threshold_text": (
            "Le modèle attribue à chaque client une probabilité de churn ; un client est signalé quand cette "
            "probabilité dépasse le seuil. Le seuil réglé en validation croisée est **{thr:.2f}**. Déplacez le "
            "curseur pour voir le compromis."
        ),
        "threshold": "Seuil",
        "cm_title": "Matrice de confusion", "actual": "Réel", "predicted": "Prédit",
        "cm_text": (
            "Avec ce seuil, le modèle détecte **{tp}** des **{pos}** churners ({rec}) et déclenche **{fp}** fausses "
            "alertes. **{fn}** churners ne sont pas détectés."
        ),
        "pr_title": "Précision et rappel selon le seuil",
        "default_comment": (
            "Avec le seuil par défaut de 0,5, le même modèle ne détecte que **{r05}** des churners. Le réglage du "
            "seuil fait monter le rappel à **{rt}** pour une perte de précision modérée ({p05} → {pt}). Pour une "
            "campagne de fidélisation, rater un churner coûte en général plus cher que contacter un client qui "
            "serait resté."
        ),
        "curves_title": "Courbes précision-rappel des quatre modèles",
        "importance_title": "Quelles variables comptent",
        "importance_text": (
            "Importance par permutation : baisse de la PR-AUC sur le jeu de test quand les valeurs d'une variable "
            "sont mélangées. Elle est calculée de la même façon pour tous les modèles, sur les variables d'origine."
        ),
        "drop": "Baisse de PR-AUC",
        "coef_title": "Sens des effets (coefficients de la régression logistique)",
        "coef_text": "Coefficient positif = augmente le risque de churn ; négatif = le réduit (les variables numériques sont transformées et mises à l'échelle).",
        "coef": "Coefficient",
        "reco_title": "Recommandations métier",
        "reco": (
            "1. **Cibler les contrats au mois** : {m2m} de ces clients partent, contre {two} avec un contrat de deux "
            "ans. Proposer une remise pour passer à un contrat d'un ou deux ans est le levier le plus fort.\n"
            "2. **Soigner les nouveaux clients** : {early} des churners partent pendant leur première année. Un "
            "programme d'accueil sur les premiers mois (appel de bienvenue, vérification des premières factures) "
            "cible la période la plus risquée.\n"
            "3. **Promouvoir la sécurité en ligne et le support technique** : les clients qui n'en ont pas partent à "
            "{sec_no}, contre {sec_yes} avec la sécurité en ligne. Intégrer ces services aux offres réduit le risque.\n"
            "4. **Analyser l'offre fibre** : {fiber} de churn pour les clients fibre contre {dsl} pour l'ADSL, malgré "
            "un meilleur service. Le prix et la qualité de l'offre fibre méritent une analyse.\n"
            "5. **Encourager le paiement automatique** : {echeck} des clients qui paient par chèque électronique "
            "partent, contre {auto} avec un paiement automatique.\n"
            "6. **Utiliser le modèle pour prioriser** : avec le seuil réglé, environ {flag} des clients sont signalés "
            "et {prec} d'entre eux partent vraiment. L'équipe de fidélisation peut se concentrer sur cette liste, en "
            "commençant par les probabilités les plus élevées."
        ),
    },
}

m = metrics()
preds = test_predictions()
y_true = preds["y_true"].to_numpy()
names = list(m["models"])
header(t(STRINGS, "eyebrow"), t(STRINGS, "title"), t(STRINGS, "caption", n=m["n_test"]))

name = st.selectbox(t(STRINGS, "choose_model"), names, index=names.index(m["selected_model"]),
                    format_func=lambda n: MODEL_NAMES[n] + (t(STRINGS, "selected_tag") if n == m["selected_model"] else ""))
info = m["models"][name]
test, base = info["test"], m["baseline_cv"]
kpis([
    (t(STRINGS, "kpi_recall"), pct(test["recall"], 0), t(STRINGS, "kpi_recall_sub")),
    (t(STRINGS, "kpi_precision"), pct(test["precision"], 0), t(STRINGS, "kpi_precision_sub")),
    (t(STRINGS, "kpi_f1"), f"{test['f1']:.2f}", t(STRINGS, "baseline_sub", v=base["f1"])),
    (t(STRINGS, "kpi_auc"), f"{test['roc_auc']:.2f}", t(STRINGS, "baseline_sub", v=base["roc_auc"])),
    (t(STRINGS, "kpi_ap"), f"{test['average_precision']:.2f}", t(STRINGS, "baseline_sub", v=base["average_precision"])),
])

st.subheader(t(STRINGS, "all_models"))
st.dataframe(pd.DataFrame([{"model": MODEL_NAMES[n], **{metric_name(k): round(s["test"][k], 3) for k in METRICS},
                            t(STRINGS, "threshold"): round(s["threshold"], 2)} for n, s in m["models"].items()]),
             hide_index=True)

# ── Threshold ───────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "threshold_title"))
st.markdown(t(STRINGS, "threshold_text", thr=info["threshold"]))
proba = preds[name].to_numpy()
thr = st.slider(t(STRINGS, "threshold"), 0.05, 0.95, float(round(info["threshold"], 2)), 0.01)
pred = (proba >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

left, right = st.columns([1, 1.4], gap="large")
with left:
    st.markdown(f"**{t(STRINGS, 'cm_title')}**")
    labels = ["No", "Yes"]
    cm = pd.DataFrame([(a, p, v) for (a, p), v in zip([(x, y) for x in labels for y in labels], [tn, fp, fn, tp])],
                      columns=[t(STRINGS, "actual"), t(STRINGS, "predicted"), "n"])
    heat = alt.Chart(cm).mark_rect(cornerRadius=4, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X(f"{t(STRINGS, 'predicted')}:N", sort=labels, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y(f"{t(STRINGS, 'actual')}:N", sort=labels),
        color=alt.Color("n:Q", scale=alt.Scale(range=["#cde2fb", "#184f95"]), legend=None),
        tooltip=[t(STRINGS, "actual"), t(STRINGS, "predicted"), "n"])
    text = heat.mark_text(fontSize=20, fontWeight=600).encode(
        text="n:Q", color=alt.condition("datum.n > 500", alt.value("#ffffff"), alt.value("#0b0b0b")))
    draw(heat + text, 260)
    st.markdown(t(STRINGS, "cm_text", tp=tp, pos=tp + fn, rec=pct(tp / (tp + fn), 0), fp=fp, fn=fn))
with right:
    st.markdown(f"**{t(STRINGS, 'pr_title')}**")
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curve = pd.DataFrame([{"threshold": g, **{k: v for k, v in test_metrics(y_true, (proba >= g).astype(int), proba).items()
                                              if k in ("precision", "recall", "f1")}} for g in grid])
    long = curve.melt("threshold", var_name="metric", value_name="value")
    long["metric"] = long["metric"].map(metric_name)
    order = [metric_name(k) for k in ("recall", "precision", "f1")]
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("threshold:Q", title=t(STRINGS, "threshold")),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("metric:N", title=None, sort=order,
                        scale=alt.Scale(domain=order, range=[CHURN_COLORS["Yes"], CHURN_COLORS["No"], "#1baf7a"])),
        tooltip=[alt.Tooltip("threshold:Q", format=".2f"), "metric", alt.Tooltip("value:Q", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"threshold": [thr]})).mark_rule(color=NEUTRAL, strokeDash=[4, 4]).encode(x="threshold:Q")
    draw(lines + rule, 300)

d05 = info["test_threshold_0.5"]
finding(t(STRINGS, "default_comment", r05=pct(d05["recall"], 0), rt=pct(test["recall"], 0),
          p05=pct(d05["precision"], 0), pt=pct(test["precision"], 0)))

st.subheader(t(STRINGS, "curves_title"))
rows = []
for n in names:
    p, r, _ = precision_recall_curve(y_true, preds[n])
    step = max(1, len(p) // 300)
    rows.append(pd.DataFrame({"recall": r[::step], "precision": p[::step], "model": MODEL_NAMES[n]}))
draw(alt.Chart(pd.concat(rows)).mark_line(strokeWidth=2).encode(
    x=alt.X("recall:Q", title=metric_name("recall")), y=alt.Y("precision:Q", title=metric_name("precision"), scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("model:N", scale=model_scale(names), title=None), order="recall:Q",
    tooltip=["model", alt.Tooltip("recall:Q", format=".2f"), alt.Tooltip("precision:Q", format=".2f")]), 320)

# ── Importance ──────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "importance_title"))
st.markdown(t(STRINGS, "importance_text"))
imp = importances()
imp = imp[imp.model == name].sort_values("importance", ascending=False)
imp["low"], imp["high"] = imp.importance - imp["std"], imp.importance + imp["std"]
base_chart = alt.Chart(imp).encode(y=alt.Y("feature:N", sort=None, title=None))
draw(base_chart.mark_bar(cornerRadiusEnd=4, color=MODEL_COLORS[name], size=14).encode(
    x=alt.X("importance:Q", title=t(STRINGS, "drop")),
    tooltip=["feature", alt.Tooltip("importance:Q", format=".4f"), alt.Tooltip("std:Q", format=".4f")])
     + base_chart.mark_rule(color="#52514e").encode(x="low:Q", x2="high:Q"), 460)
show_code("permutation_importance(model, X_test, y_test, scoring='average_precision', n_repeats=10)")

if name == "logistic_regression":
    st.subheader(t(STRINGS, "coef_title"))
    st.markdown(t(STRINGS, "coef_text"))
    coef = coefficients()
    coef = coef.reindex(coef.coefficient.abs().sort_values(ascending=False).index).head(16)
    draw(alt.Chart(coef).mark_bar(cornerRadiusEnd=4, size=14).encode(
        y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("coefficient:Q", title=t(STRINGS, "coef")),
        color=alt.condition("datum.coefficient > 0", alt.value(CHURN_COLORS["Yes"]), alt.value(CHURN_COLORS["No"])),
        tooltip=["feature", alt.Tooltip("coefficient:Q", format=".3f")]), 440)

# ── Recommendations ─────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "reco_title"))
df = eda_data()
rate = lambda mask: pct((df.loc[mask, "Churn"] == "Yes").mean(), 0)  # noqa: E731
sel = m["models"][m["selected_model"]]
sel_pred = preds[m["selected_model"]] >= sel["threshold"]
finding(t(STRINGS, "reco",
          m2m=rate(df.Contract == "Month-to-month"), two=rate(df.Contract == "Two year"),
          early=pct((df.loc[df.Churn == "Yes", "tenure"] < 12).mean(), 0),
          sec_no=rate(df.OnlineSecurity == "No"), sec_yes=rate(df.OnlineSecurity == "Yes"),
          fiber=rate(df.InternetService == "Fiber optic"), dsl=rate(df.InternetService == "DSL"),
          echeck=rate(df.PaymentMethod == "Electronic check"), auto=rate(df.PaymentMethod.str.contains("automatic")),
          flag=pct(sel_pred.mean(), 0), prec=pct(sel["test"]["precision"], 0)))
