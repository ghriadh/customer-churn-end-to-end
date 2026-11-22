import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve

from churn.evaluation import test_metrics
from common import (
    CHURN_COLORS, HIGHLIGHT, METRIC_COLORS, MODEL_NAMES, METRICS, NEUTRAL, TEXT, brief, coefficients, confusion_chart,
    confusion_grid, controls, draw, eda_data, finding, header, importances, kpis, metric_name, metrics, num, pct,
    require_artifacts, setup_page, show_code, test_predictions,
)
from i18n import t

setup_page()
require_artifacts()

STRINGS = {
    "en": {
        "eyebrow": "Modelling · 5", "title": "Results & recommendations",
        "caption": "Results on the test set: {n} customers set aside at the start and never used for training or model selection.",
        "brief": (
            "On customers it has never seen, the model detects **{recall} of the customers who leave**, and "
            "**{precision} of its alerts are correct**. The most useful lever is the **decision threshold**: flagging "
            "customers from a {thr} probability instead of 50% catches many more leavers. The page ends with "
            "**concrete recommendations** for the retention team."
        ),
        "choose_model": "Model", "selected_tag": " (selected)",
        "kpi_recall": "Recall", "kpi_recall_sub": "churners detected",
        "kpi_precision": "Precision", "kpi_precision_sub": "correct alerts",
        "kpi_f1": "F1", "kpi_auc": "ROC-AUC", "kpi_ap": "PR-AUC", "baseline_sub": "no model: {v}",
        "all_models": "All final models on the test set (tuned threshold)",
        "cm_all_title": "Confusion matrices of the final models",
        "cm_all_text": (
            "Each matrix splits the {n} test customers into four groups: **caught churners** (bottom right), "
            "**missed churners** (bottom left), **false alarms** (top right) and customers **correctly left alone** (top left)."
        ),
        "threshold_title": "Decision threshold",
        "threshold_text": (
            "The model gives each customer a churn probability; a customer is flagged when the probability exceeds "
            "the threshold. The threshold tuned in cross-validation is **{thr}**. Move the slider to see the trade-off."
        ),
        "threshold": "Decision threshold",
        "cm_title": "Confusion matrix at this threshold",
        "cm_text": (
            "With this threshold, the model detects **{tp}** of the **{pos}** churners ({rec}) and raises **{fp}** "
            "false alarms. **{fn}** churners are missed."
        ),
        "pr_title": "Precision and recall by threshold",
        "pr_note": "The curves stop where fewer than 20 customers are flagged: beyond, precision is not meaningful.",
        "default_comment": (
            "With the default threshold of 0.5, the same model detects only **{r05}** of churners. Tuning the "
            "threshold raises recall to **{rt}** with a moderate loss of precision ({p05} → {pt}). For a retention "
            "campaign, missing a churner usually costs more than contacting a customer who would have stayed."
        ),
        "curves_title": "Precision-recall curves of the final models",
        "curves_comment": (
            "The six curves almost overlap: **the models are practically equivalent** on this dataset. All of them are "
            "far above the dashed line, which is what a random ranking would give ({base})."
        ),
        "importance_title": "Which features matter",
        "importance_text": (
            "Permutation importance: drop in PR-AUC on the test set when the values of a feature are shuffled. "
            "It is computed the same way for every model, on the original features."
        ),
        "importance_comment": (
            "- **{top}** carry most of the predictive power of this model.\n"
            "- Features close to zero, or slightly negative, such as {low}, do not help the model: shuffling them "
            "changes nothing (negative values are just noise)."
        ),
        "drop": "PR-AUC drop",
        "coef_title": "Direction of the effects (logistic regression coefficients)",
        "coef_text": "Orange = increases the churn risk, blue = reduces it (numeric features are transformed and scaled).",
        "coef": "Coefficient",
        "reco_title": "Business recommendations",
        "reco": (
            "1. **Target month-to-month contracts**: {m2m} of these customers leave, against {two} with a two-year "
            "contract. Offering a discount for switching to a one- or two-year contract is the strongest lever.\n"
            "2. **Take care of new customers**: {early} of churners leave during their first year. An onboarding "
            "programme for the first months (welcome call, check of the first bills) addresses the period of highest risk.\n"
            "3. **Promote online security and tech support**: customers without online security churn at {sec_no}, "
            "against {sec_yes} with it. Bundling these services into offers lowers the risk.\n"
            "4. **Look into the fiber optic offer**: {fiber} churn for fiber customers against {dsl} for DSL, despite "
            "a better service. Price and quality of the fiber offer deserve an analysis.\n"
            "5. **Encourage automatic payment**: {echeck} of customers paying by electronic check leave, against "
            "{auto} with automatic payment.\n"
            "6. **Use the model to prioritise**: with the tuned threshold, about {flag} of customers are flagged and "
            "{prec} of them really leave. The retention team can focus on this list, starting with the highest probabilities."
        ),
        "model": "Model",
    },
    "fr": {
        "eyebrow": "Modélisation · 5", "title": "Résultats et recommandations",
        "caption": "Résultats sur le jeu de test : {n} clients mis de côté au départ et jamais utilisés pour l'entraînement ni pour le choix du modèle.",
        "brief": (
            "Sur des clients qu'il n'a jamais vus, le modèle détecte **{recall} des clients qui partent**, et "
            "**{precision} de ses alertes sont justes**. Le levier le plus utile est le **seuil de décision** : "
            "signaler les clients à partir d'une probabilité de {thr} au lieu de 50 % permet d'en repérer beaucoup "
            "plus. La page se termine par des **recommandations concrètes** pour l'équipe de fidélisation."
        ),
        "choose_model": "Modèle", "selected_tag": " (retenu)",
        "kpi_recall": "Rappel", "kpi_recall_sub": "churners détectés",
        "kpi_precision": "Précision", "kpi_precision_sub": "alertes justes",
        "kpi_f1": "F1", "kpi_auc": "ROC-AUC", "kpi_ap": "PR-AUC", "baseline_sub": "sans modèle : {v}",
        "all_models": "Tous les modèles finaux sur le jeu de test (seuil réglé)",
        "cm_all_title": "Matrices de confusion des modèles finaux",
        "cm_all_text": (
            "Chaque matrice répartit les {n} clients du test en quatre groupes : **churners détectés** (en bas à "
            "droite), **churners manqués** (en bas à gauche), **fausses alertes** (en haut à droite) et clients "
            "**à juste titre non signalés** (en haut à gauche)."
        ),
        "threshold_title": "Seuil de décision",
        "threshold_text": (
            "Le modèle attribue à chaque client une probabilité de churn ; un client est signalé quand cette "
            "probabilité dépasse le seuil. Le seuil réglé en validation croisée est **{thr}**. Déplacez le curseur "
            "pour voir le compromis."
        ),
        "threshold": "Seuil de décision",
        "cm_title": "Matrice de confusion à ce seuil",
        "cm_text": (
            "Avec ce seuil, le modèle détecte **{tp}** des **{pos}** churners ({rec}) et déclenche **{fp}** fausses "
            "alertes. **{fn}** churners ne sont pas détectés."
        ),
        "pr_title": "Précision et rappel selon le seuil",
        "pr_note": "Les courbes s'arrêtent quand moins de 20 clients sont signalés : au-delà, la précision n'a plus de sens.",
        "default_comment": (
            "Avec le seuil par défaut de 0,5, le même modèle ne détecte que **{r05}** des churners. Le réglage du "
            "seuil fait monter le rappel à **{rt}** pour une perte de précision modérée ({p05} → {pt}). Pour une "
            "campagne de fidélisation, rater un churner coûte en général plus cher que contacter un client qui "
            "serait resté."
        ),
        "curves_title": "Courbes précision-rappel des modèles finaux",
        "curves_comment": (
            "Les six courbes se superposent presque : **les modèles sont pratiquement équivalents** sur ce jeu de "
            "données. Toutes sont très au-dessus de la ligne pointillée, qui correspond à un classement au hasard ({base})."
        ),
        "importance_title": "Quelles variables comptent",
        "importance_text": (
            "Importance par permutation : baisse de la PR-AUC sur le jeu de test quand les valeurs d'une variable "
            "sont mélangées. Elle est calculée de la même façon pour tous les modèles, sur les variables d'origine."
        ),
        "importance_comment": (
            "- **{top}** portent l'essentiel du pouvoir prédictif de ce modèle.\n"
            "- Les variables proches de zéro, voire légèrement négatives, comme {low}, n'aident pas le modèle : les "
            "mélanger ne change rien (les valeurs négatives ne sont que du bruit)."
        ),
        "drop": "Baisse de PR-AUC",
        "coef_title": "Sens des effets (coefficients de la régression logistique)",
        "coef_text": "Orange = augmente le risque de churn, bleu = le réduit (les variables numériques sont transformées et mises à l'échelle).",
        "coef": "Coefficient",
        "reco_title": "Recommandations métier",
        "reco": (
            "1. **Cibler les contrats au mois** : {m2m} de ces clients partent, contre {two} avec un contrat de deux "
            "ans. Proposer une remise pour passer à un contrat d'un ou deux ans est le levier le plus fort.\n"
            "2. **Soigner les nouveaux clients** : {early} des churners partent pendant leur première année. Un "
            "programme d'accueil sur les premiers mois (appel de bienvenue, vérification des premières factures) "
            "cible la période la plus risquée.\n"
            "3. **Promouvoir la sécurité en ligne et le support technique** : les clients sans sécurité en ligne "
            "partent à {sec_no}, contre {sec_yes} avec. Intégrer ces services aux offres réduit le risque.\n"
            "4. **Analyser l'offre fibre** : {fiber} de churn pour les clients fibre contre {dsl} pour l'ADSL, malgré "
            "un meilleur service. Le prix et la qualité de l'offre fibre méritent une analyse.\n"
            "5. **Encourager le paiement automatique** : {echeck} des clients qui paient par chèque électronique "
            "partent, contre {auto} avec un paiement automatique.\n"
            "6. **Utiliser le modèle pour prioriser** : avec le seuil réglé, environ {flag} des clients sont signalés "
            "et {prec} d'entre eux partent vraiment. L'équipe de fidélisation peut se concentrer sur cette liste, en "
            "commençant par les probabilités les plus élevées."
        ),
        "model": "Modèle",
    },
}

m = metrics()
preds = test_predictions()
y_true = preds["y_true"].to_numpy()
names = list(m["models"])
sel_name = m["selected_model"]
sel_info = m["models"][sel_name]

header(t(STRINGS, "eyebrow"), t(STRINGS, "title"), t(STRINGS, "caption", n=num(m["n_test"])))
brief(t(STRINGS, "brief", recall=pct(sel_info["test"]["recall"], 0), precision=pct(sel_info["test"]["precision"], 0),
        thr=pct(sel_info["threshold"], 0)))

with controls("model"):
    name = st.selectbox(t(STRINGS, "choose_model"), names, index=names.index(sel_name),
                        format_func=lambda n: MODEL_NAMES[n] + (t(STRINGS, "selected_tag") if n == sel_name else ""))
info = m["models"][name]
test, base = info["test"], m["baseline_cv"]
kpis([
    (t(STRINGS, "kpi_recall"), pct(test["recall"], 0), t(STRINGS, "kpi_recall_sub"), "orange"),
    (t(STRINGS, "kpi_precision"), pct(test["precision"], 0), t(STRINGS, "kpi_precision_sub"), "green"),
    (t(STRINGS, "kpi_f1"), num(test["f1"], 2), t(STRINGS, "baseline_sub", v=num(base["f1"], 2)), "violet"),
    (t(STRINGS, "kpi_auc"), num(test["roc_auc"], 2), t(STRINGS, "baseline_sub", v=num(base["roc_auc"], 2))),
    (t(STRINGS, "kpi_ap"), num(test["average_precision"], 2), t(STRINGS, "baseline_sub", v=num(base["average_precision"], 2))),
])

st.subheader(t(STRINGS, "all_models"))
table = pd.DataFrame([{t(STRINGS, "model"): MODEL_NAMES[n] + (" ★" if n == sel_name else ""),
                       **{metric_name(k): s["test"][k] for k in METRICS}, t(STRINGS, "threshold"): s["threshold"]}
                      for n, s in m["models"].items()])
st.dataframe(table.style.format(precision=3).apply(
    lambda r: ["background-color: #e8f1fc; font-weight: 600" if "★" in str(r.iloc[0]) else "" for _ in r], axis=1),
    hide_index=True, width="stretch")

# ── Confusion matrices of the six final models ──────────────────────────────
st.subheader(t(STRINGS, "cm_all_title"))
st.markdown(t(STRINGS, "cm_all_text", n=num(m["n_test"])))
items = []
for n in names:
    (a, b), (c, d) = m["models"][n]["confusion_matrix"]
    items.append((MODEL_NAMES[n], (a, b, c, d),
                  f"{metric_name('recall')} {pct(d / (c + d), 0)} · {metric_name('precision')} {pct(d / (b + d), 0)} · "
                  f"{t(STRINGS, 'threshold')} {num(m['models'][n]['threshold'], 2)}"))
confusion_grid(items)

# ── Threshold ───────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "threshold_title"))
st.markdown(t(STRINGS, "threshold_text", thr=num(info["threshold"], 2)))
proba = preds[name].to_numpy()
with controls("threshold"):
    # starts exactly at the tuned threshold, so this matrix matches the one above
    thr = st.slider(t(STRINGS, "threshold"), 0.05, 0.95, float(round(info["threshold"], 3)), 0.005, format="%.3f")
pred = (proba >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

left, right = st.columns([1, 1.4], gap="large")
with left:
    st.markdown(f"**{t(STRINGS, 'cm_title')}**")
    confusion_chart(tn, fp, fn, tp, 270)
    st.markdown(t(STRINGS, "cm_text", tp=tp, pos=tp + fn, rec=pct(tp / (tp + fn), 0), fp=fp, fn=fn))
with right:
    st.markdown(f"**{t(STRINGS, 'pr_title')}**")
    rows = []
    for g in np.round(np.arange(0.05, 0.96, 0.01), 2):
        flagged = proba >= g
        if flagged.sum() < 20:
            break
        scores = test_metrics(y_true, flagged.astype(int), proba)
        rows.append({"threshold": g, **{k: scores[k] for k in ("recall", "precision", "f1")}})
    long = pd.DataFrame(rows).melt("threshold", var_name="metric", value_name="value")
    long["metric"] = long["metric"].map(metric_name)
    order = [metric_name(k) for k in ("recall", "precision", "f1")]
    lines = alt.Chart(long).mark_line(strokeWidth=3).encode(
        x=alt.X("threshold:Q", title=t(STRINGS, "threshold"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("metric:N", title=None, sort=order, scale=alt.Scale(domain=order, range=METRIC_COLORS)),
        tooltip=[alt.Tooltip("threshold:Q", format=".2f"), "metric", alt.Tooltip("value:Q", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"threshold": [thr]})).mark_rule(color=TEXT, strokeDash=[5, 4], strokeWidth=1.5).encode(x="threshold:Q")
    draw(lines + rule, 310)
    st.caption(t(STRINGS, "pr_note"))

d05 = info["test_threshold_0.5"]
finding(t(STRINGS, "default_comment", r05=pct(d05["recall"], 0), rt=pct(test["recall"], 0),
          p05=pct(d05["precision"], 0), pt=pct(test["precision"], 0)))

# ── Precision-recall curves ─────────────────────────────────────────────────
st.subheader(t(STRINGS, "curves_title"))
curves = []
for n in names:
    p, r, _ = precision_recall_curve(y_true, preds[n])
    step = max(1, len(p) // 300)
    curves.append(pd.DataFrame({"recall": r[::step], "precision": p[::step], "model": MODEL_NAMES[n],
                                "focus": int(n == name)}))
curves = pd.concat(curves).sort_values("focus")   # model in focus drawn last, on top
churn_rate = y_true.mean()
draw(alt.Chart(curves).mark_line().encode(
    x=alt.X("recall:Q", title=metric_name("recall")),
    y=alt.Y("precision:Q", title=metric_name("precision"), scale=alt.Scale(domain=[0, 1])),
    color=alt.condition(alt.datum.focus == 1, alt.value(HIGHLIGHT), alt.value(NEUTRAL)),
    strokeWidth=alt.condition(alt.datum.focus == 1, alt.value(3.5), alt.value(1.2)),
    detail="model:N", order="recall:Q",
    tooltip=["model", alt.Tooltip("recall:Q", format=".2f"), alt.Tooltip("precision:Q", format=".2f")])
     + alt.Chart(pd.DataFrame({"y": [churn_rate]})).mark_rule(color=TEXT, strokeDash=[5, 4]).encode(y="y:Q"), 320)
finding(t(STRINGS, "curves_comment", base=num(churn_rate, 3)))

# ── Importance ──────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "importance_title"))
st.markdown(t(STRINGS, "importance_text"))
imp = importances()
imp = imp[imp.model == name].sort_values("importance", ascending=False)
imp["low"], imp["high"] = imp.importance - imp["std"], imp.importance + imp["std"]
base_chart = alt.Chart(imp).encode(y=alt.Y("feature:N", sort=None, title=None))
draw(base_chart.mark_bar(cornerRadiusEnd=4, color=HIGHLIGHT, size=15).encode(
    x=alt.X("importance:Q", title=t(STRINGS, "drop")),
    tooltip=["feature", alt.Tooltip("importance:Q", format=".4f"), alt.Tooltip("std:Q", format=".4f")])
     + base_chart.mark_rule(color="#52514e").encode(x="low:Q", x2="high:Q"), 480)
top = ", ".join(f"`{f}`" for f in imp.feature.head(3))
low = ", ".join(f"`{f}`" for f in imp.feature.tail(3))
finding(t(STRINGS, "importance_comment", top=top, low=low))
show_code("permutation_importance(model, X_test, y_test, scoring='average_precision', n_repeats=10)")

if name == "logistic_regression":
    st.subheader(t(STRINGS, "coef_title"))
    st.markdown(t(STRINGS, "coef_text"))
    coef = coefficients()
    coef = coef.reindex(coef.coefficient.abs().sort_values(ascending=False).index).head(16)
    draw(alt.Chart(coef).mark_bar(cornerRadiusEnd=4, size=15).encode(
        y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("coefficient:Q", title=t(STRINGS, "coef")),
        color=alt.condition("datum.coefficient > 0", alt.value(CHURN_COLORS["Yes"]), alt.value(CHURN_COLORS["No"])),
        tooltip=["feature", alt.Tooltip("coefficient:Q", format=".3f")]), 460)

# ── Recommendations ─────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "reco_title"))
df = eda_data()
churn = df["Churn"] == "Yes"
rate = lambda mask: pct(churn[mask].mean(), 0)  # noqa: E731
sel_pred = preds[sel_name] >= sel_info["threshold"]
finding(t(STRINGS, "reco",
          m2m=rate(df.Contract == "Month-to-month"), two=rate(df.Contract == "Two year"),
          early=pct((df.loc[churn, "tenure"] < 12).mean(), 0),
          sec_no=rate(df.OnlineSecurity == "No"), sec_yes=rate(df.OnlineSecurity == "Yes"),
          fiber=rate(df.InternetService == "Fiber optic"), dsl=rate(df.InternetService == "DSL"),
          echeck=rate(df.PaymentMethod == "Electronic check"), auto=rate(df.PaymentMethod.str.contains("automatic")),
          flag=pct(sel_pred.mean(), 0), prec=pct(sel_info["test"]["precision"], 0)))
