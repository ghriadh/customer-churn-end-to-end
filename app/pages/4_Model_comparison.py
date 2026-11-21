import altair as alt
import pandas as pd
import streamlit as st

from churn.data import clean_data, load_data, split_data
from churn.evaluation import cross_validate_config, make_cv
from churn.pipeline import IMBALANCE, IMPUTERS, MODELS, SCALERS, TRANSFORMS, build_pipeline
from common import (
    CHURN_COLORS, MODEL_NAMES, METRICS, confusion_chart, confusion_grid, cv_results, draw, finding, header, highlight,
    metric_name, metrics,
    require_artifacts, setup_page, show_code,
)
from i18n import t

setup_page()
require_artifacts()

STRINGS = {
    "en": {
        "eyebrow": "Modelling · 4", "title": "Model comparison",
        "caption": "{n} combinations of preparation methods and models, all evaluated with the same 5-fold cross-validation on the training set.",
        "protocol": (
            "Each combination = imputation × numeric transformation × scaler × imbalance strategy × model. "
            "Tree-based models are insensitive to monotonic transformations and to scaling, so only the imputation "
            "and imbalance strategies are varied for them. Every metric is computed on the validation folds: the "
            "test set is not used on this page."
        ),
        "metric": "Rank by", "baseline": "Baseline (always predicts \"no churn\"): {name} = {v:.3f}",
        "best_per_model": "Best combination of each model ({metric}, mean ± std over the 5 folds)",
        "table": "All combinations",
        "cm_title": "Confusion matrices in cross-validation (threshold 0.5)",
        "cm_text": (
            "Each customer of the training set is predicted by the model of the fold where it was held out, so every "
            "matrix covers the {n:,} training customers. Best combination of each model for {metric}:"
        ),
        "cm_pick": "Confusion matrix of any combination", "cm_combo": "Combination",
        "cm_note": "These matrices use the default threshold of 0.5; the final models use a tuned threshold (see *Final evaluation*).",
        "imbalance_title": "Effect of the class imbalance strategy",
        "imbalance_sub": "Mean over all combinations using each strategy",
        "imbalance_comment": (
            "- The imbalance strategy barely changes **PR-AUC and ROC-AUC**: the models rank customers the same way.\n"
            "- It mainly moves the **balance between recall and precision**: `class_weight`, `over` and `smote` detect "
            "far more churners (recall {r_bal:.2f} against {r_none:.2f}) at the cost of more false alarms (precision "
            "{p_bal:.2f} against {p_none:.2f}).\n"
            "- In practice it acts like lowering the decision threshold, which is why the threshold is tuned for "
            "every final model (see *Final evaluation*)."
        ),
        "transform_title": "Effect of the transformation and scaler (logistic regression)",
        "transform_comment": "The `log` transformation gives the best results for logistic regression; the choice of scaler has almost no effect.",
        "winners_title": "The best model depends on the metric",
        "winners_comment": (
            "Differences between models are small (the best PR-AUC of each model lies between {lo:.3f} and {hi:.3f}), "
            "so the winner changes with the metric. Gradient boosting and random forest do not beat a well-prepared "
            "logistic regression on this dataset: the relationships are mostly additive."
        ),
        "winner": "Best combination",
        "tuned_title": "Final models",
        "tuned_text": (
            "For each model, the best combination ({metric}) is kept, its hyperparameters are tuned with a grid search "
            "(same cross-validation), then its decision threshold is tuned to maximise {thr_metric}. "
            "The selected model is **{selected}**."
        ),
        "try_title": "Try your own combination",
        "try_text": "Choose a combination: it is evaluated live with the same 5-fold cross-validation on the training set.",
        "run": "Evaluate", "running": "Cross-validation in progress…",
        "class_weight_error": "AdaBoost does not support `class_weight`: choose another strategy.",
        "retrain": "To retrain everything with another selection metric:",
        "model": "Model", "imputer": "Imputation", "transform": "Transformation", "scaler": "Scaler",
        "imbalance": "Imbalance", "params": "Hyperparameters", "threshold": "Threshold", "cv": "CV score",
    },
    "fr": {
        "eyebrow": "Modélisation · 4", "title": "Comparaison des modèles",
        "caption": "{n} combinaisons de méthodes de préparation et de modèles, toutes évaluées avec la même validation croisée à 5 folds sur le jeu d'entraînement.",
        "protocol": (
            "Chaque combinaison = imputation × transformation numérique × scaler × stratégie de déséquilibre × modèle. "
            "Les modèles à base d'arbres ne sont pas sensibles aux transformations monotones ni à la mise à l'échelle : "
            "pour eux, seules l'imputation et la stratégie de déséquilibre varient. Toutes les métriques sont "
            "calculées sur les folds de validation : le jeu de test n'est pas utilisé sur cette page."
        ),
        "metric": "Classer par", "baseline": "Baseline (prédit toujours « pas de churn ») : {name} = {v:.3f}",
        "best_per_model": "Meilleure combinaison de chaque modèle ({metric}, moyenne ± écart-type sur les 5 folds)",
        "table": "Toutes les combinaisons",
        "cm_title": "Matrices de confusion en validation croisée (seuil 0,5)",
        "cm_text": (
            "Chaque client du jeu d'entraînement est prédit par le modèle du fold où il a été mis de côté : chaque "
            "matrice couvre donc les {n:,} clients d'entraînement. Meilleure combinaison de chaque modèle pour {metric} :"
        ),
        "cm_pick": "Matrice de confusion de n'importe quelle combinaison", "cm_combo": "Combinaison",
        "cm_note": "Ces matrices utilisent le seuil par défaut de 0,5 ; les modèles finaux utilisent un seuil réglé (voir *Évaluation finale*).",
        "imbalance_title": "Effet de la stratégie de déséquilibre",
        "imbalance_sub": "Moyenne sur toutes les combinaisons utilisant chaque stratégie",
        "imbalance_comment": (
            "- La stratégie de déséquilibre ne change presque pas la **PR-AUC ni la ROC-AUC** : les modèles classent "
            "les clients de la même façon.\n"
            "- Elle déplace surtout **l'équilibre entre rappel et précision** : `class_weight`, `over` et `smote` "
            "détectent beaucoup plus de churners (rappel {r_bal:.2f} contre {r_none:.2f}) au prix de plus de fausses "
            "alertes (précision {p_bal:.2f} contre {p_none:.2f}).\n"
            "- En pratique, cela revient à baisser le seuil de décision : c'est pourquoi le seuil est réglé pour "
            "chaque modèle final (voir *Évaluation finale*)."
        ),
        "transform_title": "Effet de la transformation et du scaler (régression logistique)",
        "transform_comment": "La transformation `log` donne les meilleurs résultats pour la régression logistique ; le choix du scaler n'a presque pas d'effet.",
        "winners_title": "Le meilleur modèle dépend de la métrique",
        "winners_comment": (
            "Les écarts entre modèles sont faibles (la meilleure PR-AUC de chaque modèle est comprise entre {lo:.3f} "
            "et {hi:.3f}) : le gagnant change donc selon la métrique. Le gradient boosting et la forêt aléatoire ne "
            "battent pas une régression logistique bien préparée sur ce jeu de données : les relations sont surtout additives."
        ),
        "winner": "Meilleure combinaison",
        "tuned_title": "Modèles finaux",
        "tuned_text": (
            "Pour chaque modèle, la meilleure combinaison ({metric}) est conservée, ses hyperparamètres sont réglés "
            "par recherche sur grille (même validation croisée), puis son seuil de décision est réglé pour "
            "maximiser le {thr_metric}. Le modèle retenu est **{selected}**."
        ),
        "try_title": "Testez votre propre combinaison",
        "try_text": "Choisissez une combinaison : elle est évaluée en direct avec la même validation croisée à 5 folds sur le jeu d'entraînement.",
        "run": "Évaluer", "running": "Validation croisée en cours…",
        "class_weight_error": "AdaBoost ne gère pas `class_weight` : choisissez une autre stratégie.",
        "retrain": "Pour tout réentraîner avec une autre métrique de sélection :",
        "model": "Modèle", "imputer": "Imputation", "transform": "Transformation", "scaler": "Scaler",
        "imbalance": "Déséquilibre", "params": "Hyperparamètres", "threshold": "Seuil", "cv": "Score CV",
    },
}

m = metrics()
results = cv_results().copy()
results["model_name"] = results["model"].map(MODEL_NAMES)

header(t(STRINGS, "eyebrow"), t(STRINGS, "title"), t(STRINGS, "caption", n=len(results)))
st.markdown(t(STRINGS, "protocol"))

metric = st.selectbox(t(STRINGS, "metric"), METRICS, index=METRICS.index(m["selection_metric"]), format_func=metric_name)
st.caption(t(STRINGS, "baseline", name=metric_name(metric), v=m["baseline_cv"][metric]))

# ── Best combination per model ──────────────────────────────────────────────
st.subheader(t(STRINGS, "best_per_model", metric=metric_name(metric)))
best = results.sort_values(metric, ascending=False).groupby("model", sort=False).head(1).copy()
best["low"], best["high"] = best[metric] - best[f"{metric}_std"], best[metric] + best[f"{metric}_std"]
best["combination"] = best[["imputer", "transform", "scaler", "imbalance"]].agg(" · ".join, axis=1)
base = alt.Chart(best).encode(y=alt.Y("model_name:N", sort=None, title=None))
bars = base.mark_bar(cornerRadiusEnd=4, size=26).encode(
    x=alt.X(f"{metric}:Q", title=metric_name(metric), scale=alt.Scale(zero=False, padding=20)),
    color=highlight("model", best.iloc[0]["model"]),
    tooltip=[alt.Tooltip("model_name:N", title=t(STRINGS, "model")), alt.Tooltip("combination:N"),
             alt.Tooltip(f"{metric}:Q", format=".3f"), alt.Tooltip(f"{metric}_std:Q", format=".3f", title="std")])
errors = base.mark_rule(color="#52514e").encode(x="low:Q", x2="high:Q")
labels = base.mark_text(align="left", dx=6, color="#0b0b0b").encode(x="high:Q", text=alt.Text(f"{metric}:Q", format=".3f"))
draw(bars + errors + labels, 220)

# ── Confusion matrices (cross-validation) ───────────────────────────────────
def caption(row):
    return (f"{metric_name('recall')} {row.tp / (row.tp + row.fn):.0%} · {metric_name('precision')} "
            f"{row.tp / (row.tp + row.fp):.0%} · {row.imputer} · {row.transform} · {row.scaler} · {row.imbalance}")


st.subheader(t(STRINGS, "cm_title"))
st.markdown(t(STRINGS, "cm_text", n=m["n_train"], metric=metric_name(metric)))
confusion_grid([(r.model_name, (r.tn, r.fp, r.fn, r.tp), caption(r)) for r in best.itertuples()])
st.caption(t(STRINGS, "cm_note"))

st.markdown(f"**{t(STRINGS, 'cm_pick')}**")
ranked = results.sort_values(metric, ascending=False).reset_index(drop=True)
labels = [f"{r.model_name} · {r.imputer} · {r.transform} · {r.scaler} · {r.imbalance}  ({metric_name(metric)} {getattr(r, metric):.3f})"
          for r in ranked.itertuples()]
pick = st.selectbox(t(STRINGS, "cm_combo"), range(len(ranked)), format_func=labels.__getitem__)
row = ranked.iloc[pick]
left, right = st.columns([1, 1.5], gap="large")
with left:
    confusion_chart(row.tn, row.fp, row.fn, row.tp, 240)
with right:
    st.dataframe(pd.DataFrame([{metric_name(k): f"{row[k]:.3f} ± {row[k + '_std']:.3f}" for k in METRICS}]).T
                 .rename(columns={0: "CV"}))

# ── All combinations ────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "table"))
cols = ["model_name", "imputer", "transform", "scaler", "imbalance"] + METRICS
table = results.sort_values(metric, ascending=False)[cols].rename(
    columns={"model_name": t(STRINGS, "model"), "imputer": t(STRINGS, "imputer"), "transform": t(STRINGS, "transform"),
             "scaler": t(STRINGS, "scaler"), "imbalance": t(STRINGS, "imbalance"), **{k: metric_name(k) for k in METRICS}})
st.dataframe(table.style.format({metric_name(k): "{:.3f}" for k in METRICS})
             .background_gradient(cmap="Blues", subset=[metric_name(metric)]), hide_index=True, height=360)

# ── Imbalance strategies ────────────────────────────────────────────────────
st.subheader(t(STRINGS, "imbalance_title"))
st.caption(t(STRINGS, "imbalance_sub"))
shown = ["recall", "precision", "f1", "average_precision"]
imb = results.groupby("imbalance")[shown].mean().reindex(["none", "class_weight", "over", "smote"]).reset_index()
long = imb.melt("imbalance", var_name="metric", value_name="value")
long["metric"] = long["metric"].map(metric_name)
chart = alt.Chart(long).mark_bar(cornerRadiusEnd=4, color=CHURN_COLORS["No"]).encode(
    y=alt.Y("imbalance:N", sort=["none", "class_weight", "over", "smote"], title=None),
    x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
    tooltip=["imbalance", "metric", alt.Tooltip("value:Q", format=".3f")])
chart = (chart + chart.mark_text(align="left", dx=4, color="#0b0b0b").encode(text=alt.Text("value:Q", format=".2f"))
         ).properties(width=220, height=140).facet(column=alt.Column("metric:N", title=None, sort=[metric_name(k) for k in shown]))
st.altair_chart(chart.configure(background="#fcfcfb").configure_view(strokeWidth=0)
                .configure_axis(labelColor="#52514e", gridColor="#e6e5e1", domainColor="#e6e5e1")
                .configure_header(labelColor="#0b0b0b", labelFontSize=13), theme=None)
bal = results[results.imbalance.isin(["class_weight", "over", "smote"])]
none = results[results.imbalance == "none"]
finding(t(STRINGS, "imbalance_comment", r_bal=bal.recall.mean(), r_none=none.recall.mean(),
          p_bal=bal.precision.mean(), p_none=none.precision.mean()))

st.subheader(t(STRINGS, "transform_title"))
lr = results[results.model == "logistic_regression"]
st.dataframe(lr.pivot_table(index="transform", columns="scaler", values=metric).style.format("{:.3f}")
             .background_gradient(cmap="Blues", axis=None))
st.markdown(t(STRINGS, "transform_comment"))

# ── Winner per metric ───────────────────────────────────────────────────────
st.subheader(t(STRINGS, "winners_title"))
winners = []
for k in METRICS:
    row = results.sort_values(k, ascending=False).iloc[0]
    winners.append({"metric": metric_name(k), t(STRINGS, "winner"): f"{row.model_name} · {row.transform} · {row.scaler} · {row.imbalance}",
                    "score": round(row[k], 3)})
st.dataframe(pd.DataFrame(winners), hide_index=True)
best_ap = results.groupby("model")["average_precision"].max()
finding(t(STRINGS, "winners_comment", lo=best_ap.min(), hi=best_ap.max()))

# ── Final tuned models ──────────────────────────────────────────────────────
st.subheader(t(STRINGS, "tuned_title"))
st.markdown(t(STRINGS, "tuned_text", metric=metric_name(m["selection_metric"]), thr_metric=metric_name(m["threshold_metric"]),
              selected=MODEL_NAMES[m["selected_model"]]))
st.dataframe(pd.DataFrame([{
    t(STRINGS, "model"): MODEL_NAMES[name], t(STRINGS, "transform"): s["config"]["transform"],
    t(STRINGS, "scaler"): s["config"]["scaler"], t(STRINGS, "imbalance"): s["config"]["imbalance"],
    t(STRINGS, "params"): ", ".join(f"{k.replace('model__', '')}={v}" for k, v in s["params"].items()),
    f"{t(STRINGS, 'cv')} ({metric_name(m['selection_metric'])})": round(s["cv_score"], 3),
    t(STRINGS, "threshold"): round(s["threshold"], 2),
} for name, s in m["models"].items()]), hide_index=True)
show_code(
    "search = GridSearchCV(build_pipeline(**best_config), PARAM_GRIDS[model_name],\n"
    "                      scoring=metric, cv=StratifiedKFold(5, shuffle=True, random_state=1234))\n"
    "search.fit(X_train, y_train)\n"
    "model = TunedThresholdClassifierCV(search.best_estimator_, scoring='f1', cv=cv)\n"
    "model.fit(X_train, y_train)"
)

# ── Try a combination ───────────────────────────────────────────────────────
st.subheader(t(STRINGS, "try_title"))
st.markdown(t(STRINGS, "try_text"))


@st.cache_data(show_spinner=False)
def evaluate(config):
    X, y = clean_data(load_data())
    X_train, _, y_train, _ = split_data(X, y)
    return cross_validate_config(build_pipeline(**dict(config)), X_train, y_train, make_cv())


with st.form("try"):
    c = st.columns(5)
    choice = {
        "model": c[0].selectbox(t(STRINGS, "model"), list(MODELS), format_func=MODEL_NAMES.get),
        "imputer": c[1].selectbox(t(STRINGS, "imputer"), IMPUTERS),
        "transform": c[2].selectbox(t(STRINGS, "transform"), TRANSFORMS, index=1),
        "scaler": c[3].selectbox(t(STRINGS, "scaler"), SCALERS),
        "imbalance": c[4].selectbox(t(STRINGS, "imbalance"), IMBALANCE),
    }
    submitted = st.form_submit_button(t(STRINGS, "run"))
if submitted:
    if choice["model"] == "adaboost" and choice["imbalance"] == "class_weight":
        st.warning(t(STRINGS, "class_weight_error"))
    else:
        with st.spinner(t(STRINGS, "running")):
            scores = evaluate(tuple(choice.items()))
        st.dataframe(pd.DataFrame([{metric_name(k): f"{scores[k]:.3f} ± {scores[k + '_std']:.3f}" for k in METRICS}]), hide_index=True)

st.markdown(t(STRINGS, "retrain"))
st.code("python run_pipeline.py --metric f1 --threshold-metric f2", language="bash")
