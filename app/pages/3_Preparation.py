import altair as alt
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import estimator_html_repr

from churn.config import SEED
from churn.data import NUM_COLS, clean_data, load_data, split_data
from churn.pipeline import build_pipeline
from churn.transformers import GroupMedianImputer
from common import CHURN_COLORS, COMMON, churn_scale, draw, finding, header, pct, setup_page, show_code
from i18n import t

setup_page()

STRINGS = {
    "en": {
        "eyebrow": "Modelling · 3", "title": "Preparation",
        "caption": "Every step of the pipeline, in order. Each step that learns something from the data is fitted on the training set only.",
        "s1": "1. Train / test split",
        "s1_text": (
            "The data is split once: **80% for training** ({n_train:,} customers) and **20% for the final test** "
            "({n_test:,} customers), stratified so both sets keep the same churn rate. Model selection uses "
            "5-fold cross-validation inside the training set; the test set is used only once, for the final evaluation."
        ),
        "set": "Set", "churn_rate": "Churn rate", "rows": "Customers",
        "s2": "2. Missing values of `TotalCharges`",
        "s2_text": "The {n} missing values (customers with `tenure = 0`) can be filled in three ways:",
        "s2_options": (
            "- `group_median`: median of `TotalCharges` per **contract type × online security**, learned on the training data.\n"
            "- `median`: overall median, learned on the training data.\n"
            "- `zero`: 0, the real value for customers who have not been billed yet."
        ),
        "s2_medians": "Medians learned on the training set (`group_median`)",
        "s3": "3. Transformation of numeric features",
        "s3_text": "Option `transform`: `none`, `log` (log(1 + x), defined at 0), `sqrt` or `yeo-johnson` (power transform whose parameter is learned on the training data).",
        "s3_choose": "Transformation", "s3_feature": "Feature",
        "before": "Before", "after": "After", "skew_line": "Skewness: {b:.2f} before, {a:.2f} after.",
        "s3_comment": (
            "Transformations reduce the skewness of `TotalCharges` but none makes the features normal. They matter "
            "for linear models only: tree-based models are insensitive to monotonic transformations."
        ),
        "s4": "4. Encoding of categorical features",
        "s4_text": (
            "One-hot encoding: one 0/1 column per value. Binary features (`Yes`/`No`) keep a single column. "
            "Values never seen during training are ignored instead of causing an error. Result: **{n} columns** for the model."
        ),
        "s5": "5. Scaling",
        "s5_text": "Option `scaler`: `standard` (mean 0, standard deviation 1), `minmax` (between 0 and 1) or `robust` (median and interquartile range, less sensitive to extreme values). Applied to numeric features, fitted on the training data.",
        "s6": "6. Class imbalance",
        "s6_text": (
            "Option `imbalance`: `none`, `class_weight` (the model gives more weight to churners), `over` (random "
            "duplication of churners), `under` (random removal of non-churners) or `smote` (synthetic churners). "
            "Resampling is applied **only to the training data of each fold**: validation and test data keep the real churn rate."
        ),
        "s6_choose": "Strategy", "s6_result": "Training set after resampling",
        "s6_comment": (
            "Resampling changes what the model sees during training, not the real proportion of churners. "
            "The *Model comparison* page shows its effect: it mostly moves the balance between recall and precision."
        ),
        "s7": "7. The complete pipeline",
        "s7_text": "All the steps are chained in a single scikit-learn / imbalanced-learn pipeline. Click on the blocks to see their parameters.",
    },
    "fr": {
        "eyebrow": "Modélisation · 3", "title": "Préparation",
        "caption": "Chaque étape du pipeline, dans l'ordre. Toute étape qui apprend quelque chose des données est ajustée sur le jeu d'entraînement uniquement.",
        "s1": "1. Séparation entraînement / test",
        "s1_text": (
            "Les données sont séparées une seule fois : **80 % pour l'entraînement** ({n_train:,} clients) et "
            "**20 % pour le test final** ({n_test:,} clients), de façon stratifiée pour que les deux jeux gardent le "
            "même taux de churn. Le choix du modèle se fait par validation croisée à 5 folds dans le jeu "
            "d'entraînement ; le jeu de test ne sert qu'une fois, pour l'évaluation finale."
        ),
        "set": "Jeu", "churn_rate": "Taux de churn", "rows": "Clients",
        "s2": "2. Valeurs manquantes de `TotalCharges`",
        "s2_text": "Les {n} valeurs manquantes (clients avec `tenure = 0`) peuvent être complétées de trois façons :",
        "s2_options": (
            "- `group_median` : médiane de `TotalCharges` par **type de contrat × sécurité en ligne**, apprise sur les données d'entraînement.\n"
            "- `median` : médiane globale, apprise sur les données d'entraînement.\n"
            "- `zero` : 0, la vraie valeur pour des clients qui n'ont pas encore été facturés."
        ),
        "s2_medians": "Médianes apprises sur le jeu d'entraînement (`group_median`)",
        "s3": "3. Transformation des variables numériques",
        "s3_text": "Option `transform` : `none`, `log` (log(1 + x), défini en 0), `sqrt` ou `yeo-johnson` (transformation de puissance dont le paramètre est appris sur les données d'entraînement).",
        "s3_choose": "Transformation", "s3_feature": "Variable",
        "before": "Avant", "after": "Après", "skew_line": "Asymétrie : {b:.2f} avant, {a:.2f} après.",
        "s3_comment": (
            "Les transformations réduisent l'asymétrie de `TotalCharges` mais aucune ne rend les variables normales. "
            "Elles comptent seulement pour les modèles linéaires : les modèles à base d'arbres ne sont pas sensibles "
            "aux transformations monotones."
        ),
        "s4": "4. Encodage des variables catégorielles",
        "s4_text": (
            "Encodage one-hot : une colonne 0/1 par valeur. Les variables binaires (`Yes`/`No`) gardent une seule "
            "colonne. Les valeurs jamais vues à l'entraînement sont ignorées au lieu de provoquer une erreur. "
            "Résultat : **{n} colonnes** pour le modèle."
        ),
        "s5": "5. Mise à l'échelle",
        "s5_text": "Option `scaler` : `standard` (moyenne 0, écart-type 1), `minmax` (entre 0 et 1) ou `robust` (médiane et écart interquartile, moins sensible aux valeurs extrêmes). Appliquée aux variables numériques, ajustée sur les données d'entraînement.",
        "s6": "6. Déséquilibre des classes",
        "s6_text": (
            "Option `imbalance` : `none`, `class_weight` (le modèle donne plus de poids aux churners), `over` "
            "(duplication aléatoire de churners), `under` (suppression aléatoire de non-churners) ou `smote` "
            "(churners synthétiques). Le rééchantillonnage est appliqué **uniquement aux données d'entraînement de "
            "chaque fold** : les données de validation et de test gardent le vrai taux de churn."
        ),
        "s6_choose": "Stratégie", "s6_result": "Jeu d'entraînement après rééchantillonnage",
        "s6_comment": (
            "Le rééchantillonnage change ce que le modèle voit pendant l'entraînement, pas la vraie proportion de "
            "churners. La page *Comparaison des modèles* montre son effet : il déplace surtout l'équilibre entre "
            "rappel et précision."
        ),
        "s7": "7. Le pipeline complet",
        "s7_text": "Toutes les étapes sont enchaînées dans un seul pipeline scikit-learn / imbalanced-learn. Cliquez sur les blocs pour voir leurs paramètres.",
    },
}


@st.cache_data
def split():
    X, y = clean_data(load_data())
    return split_data(X, y)


header(t(STRINGS, "eyebrow"), t(STRINGS, "title"), t(STRINGS, "caption"))
X_train, X_test, y_train, y_test = split()

# ── 1. Split ────────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "s1"))
st.markdown(t(STRINGS, "s1_text", n_train=len(X_train), n_test=len(X_test)))
st.dataframe(pd.DataFrame({
    t(STRINGS, "set"): ["train", "test"],
    t(STRINGS, "rows"): [len(X_train), len(X_test)],
    t(STRINGS, "churn_rate"): [pct(y_train.mean()), pct(y_test.mean())],
}), hide_index=True)
show_code("X, y = clean_data(load_data())   # drop customerID, TotalCharges -> number, Churn -> 0/1\n"
          "X_train, X_test, y_train, y_test = train_test_split(\n"
          "    X, y, test_size=0.2, stratify=y, random_state=1234)")

# ── 2. Missing values ───────────────────────────────────────────────────────
st.subheader(t(STRINGS, "s2"))
st.markdown(t(STRINGS, "s2_text", n=int(pd.concat([X_train, X_test]).TotalCharges.isna().sum())))
st.markdown(t(STRINGS, "s2_options"))
imputer = GroupMedianImputer().fit(X_train)
st.markdown(f"**{t(STRINGS, 's2_medians')}**")
st.dataframe(imputer.medians_.unstack().round(1))
show_code(
    "class GroupMedianImputer(BaseEstimator, TransformerMixin):\n"
    "    def fit(self, X, y=None):\n"
    "        self.medians_ = X.groupby(['Contract', 'OnlineSecurity'])['TotalCharges'].median()\n"
    "        self.global_median_ = X['TotalCharges'].median()\n"
    "        return self\n\n"
    "    def transform(self, X):\n"
    "        X = X.copy()\n"
    "        # fill each missing value with the median of its group (global median if unseen group)\n"
    "        ...\n        return X"
)

# ── 3. Transformation ───────────────────────────────────────────────────────
st.subheader(t(STRINGS, "s3"))
st.markdown(t(STRINGS, "s3_text"))
c1, c2 = st.columns(2)
transform = c1.radio(t(STRINGS, "s3_choose"), ["log", "sqrt", "yeo-johnson"], horizontal=True)
feature = c2.radio(t(STRINGS, "s3_feature"), NUM_COLS, index=2, horizontal=True)
columns = build_pipeline(transform=transform, imputer="zero").named_steps["preprocess"]
num_step = {name: step for name, step, _ in columns.transformers}["num"]
num_step = num_step.set_params(scale="passthrough").fit(X_train[NUM_COLS])
after = pd.DataFrame(num_step.transform(X_train[NUM_COLS]), columns=NUM_COLS)[feature]
before = X_train[feature].fillna(0)
dist = pd.concat([pd.DataFrame({"value": before.to_numpy(), "step": t(STRINGS, "before")}),
                  pd.DataFrame({"value": after.to_numpy(), "step": t(STRINGS, "after")})])
cols = st.columns(2, gap="large")
for col, step in zip(cols, [t(STRINGS, "before"), t(STRINGS, "after")]):
    with col:
        st.markdown(f"**{step}**")
        draw(alt.Chart(dist[dist.step == step]).mark_bar(color=CHURN_COLORS["No"], cornerRadiusEnd=2).encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=30), title=feature),
            y=alt.Y("count():Q", title=t(COMMON, "count"))), 220)
st.markdown(t(STRINGS, "skew_line", b=before.skew(), a=after.skew()))
finding(t(STRINGS, "s3_comment"))
show_code("{'log': FunctionTransformer(np.log1p),\n 'sqrt': FunctionTransformer(np.sqrt),\n"
          " 'yeo-johnson': PowerTransformer(method='yeo-johnson', standardize=False)}[transform]")

# ── 4. Encoding ─────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "s4"))
prep = build_pipeline()[:-2].fit(X_train, y_train)
names = prep.named_steps["preprocess"].get_feature_names_out()
st.markdown(t(STRINGS, "s4_text", n=len(names)))
st.dataframe(pd.DataFrame(prep.transform(X_train.head(5)), columns=names).round(2), hide_index=True)
show_code("OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)")

# ── 5. Scaling ──────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "s5"))
st.markdown(t(STRINGS, "s5_text"))
show_code("{'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}[scaler]")

# ── 6. Imbalance ────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "s6"))
st.markdown(t(STRINGS, "s6_text"))
strategy = st.radio(t(STRINGS, "s6_choose"), ["none", "over", "under", "smote"], horizontal=True)
X_prep = prep.transform(X_train)
sampler = {"over": RandomOverSampler(random_state=SEED), "under": RandomUnderSampler(random_state=SEED),
           "smote": SMOTE(random_state=SEED)}.get(strategy)
y_res = sampler.fit_resample(X_prep, y_train)[1] if sampler else y_train
counts = y_res.map({0: "No", 1: "Yes"}).value_counts().rename_axis("Churn").reset_index(name="n")
left, right = st.columns([1, 1.3], gap="large")
with left:
    st.markdown(f"**{t(STRINGS, 's6_result')}**")
    bars = alt.Chart(counts).mark_bar(cornerRadiusEnd=4, size=50).encode(
        x=alt.X("Churn:N", sort=["No", "Yes"], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("n:Q", title=t(COMMON, "count")), color=alt.Color("Churn:N", scale=churn_scale(), legend=None),
        tooltip=["Churn", "n"])
    draw(bars + bars.mark_text(dy=-8, color="#0b0b0b").encode(text="n:Q"), 240)
with right:
    finding(t(STRINGS, "s6_comment"))
show_code("from imblearn.pipeline import Pipeline   # resamples during fit only\n"
          "{'over': RandomOverSampler(), 'under': RandomUnderSampler(), 'smote': SMOTE()}[imbalance]\n"
          "# 'class_weight' -> LogisticRegression(class_weight='balanced'), ...")

# ── 7. Full pipeline ────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "s7"))
st.markdown(t(STRINGS, "s7_text"))
st.html(estimator_html_repr(build_pipeline(model="logistic_regression", imputer="group_median",
                                           transform="log", scaler="robust", imbalance="smote")))
show_code("pipeline = build_pipeline(model='logistic_regression', imputer='group_median',\n"
          "                          transform='log', scaler='robust', imbalance='smote')\n"
          "pipeline.fit(X_train, y_train)")
