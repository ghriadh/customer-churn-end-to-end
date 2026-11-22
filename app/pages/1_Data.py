import pandas as pd
import streamlit as st

from common import brief, controls, finding, header, num, raw_data, setup_page, show_code, tech
from i18n import t

setup_page()

STRINGS = {
    "en": {
        "eyebrow": "Analysis · 1", "title": "Data",
        "caption": "IBM Telco Customer Churn dataset: one row per customer.",
        "brief": (
            "The dataset describes **{rows} customers** of a telecom operator: their profile, the services they use, "
            "their contract and bills, and whether they **left in the last month**. Two small cleaning steps are "
            "needed before modelling: removing the customer identifier and fixing 11 empty values."
        ),
        "sample": "Sample of the raw data", "resample": "Show another sample",
        "shape": "**{rows}** customers and **{cols}** columns.",
        "columns_title": "Columns",
        "columns": (
            "- `customerID`: customer identifier, dropped (see below).\n"
            "- **Target** `Churn`: `Yes` if the customer left during the last month, `No` otherwise.\n"
            "- **Profile**: `gender`, `SeniorCitizen` (coded 0/1), `Partner`, `Dependents`.\n"
            "- **Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, "
            "`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.\n"
            "- **Account**: `tenure` (months as a customer), `Contract`, `PaperlessBilling`, `PaymentMethod`, "
            "`MonthlyCharges`, `TotalCharges`."
        ),
        "types_title": "Types and missing values",
        "types_comment": (
            "- No column contains null values when the file is read.\n"
            "- Only `SeniorCitizen`, `tenure` and `MonthlyCharges` are read as numbers; the 18 other columns are text.\n"
            "- `SeniorCitizen` is numeric but is a categorical variable (0/1).\n"
            "- `TotalCharges` should be numeric but is read as text: some values cannot be converted (see below)."
        ),
        "describe_num": "Numeric columns", "describe_cat": "Text columns",
        "id_title": "Dropping `customerID`",
        "id_comment": (
            "`customerID` takes **{n_unique} different values for {n_rows} rows**: it is a unique identifier. "
            "It says nothing about the customer's behaviour, and a model could only learn it by heart, so it is "
            "**dropped** before modelling."
        ),
        "tc_title": "Cleaning `TotalCharges`",
        "tc_comment": (
            "Converting `TotalCharges` to a number fails because **{n} values are a blank space** (`' '`, shown "
            "below). They are converted to missing values (NaN).\n\n"
            "**These {n} customers all have `tenure = 0`**: they joined during the current month and have not been "
            "billed yet. Their real total is therefore 0. The pipeline offers three ways to fill these values "
            "(0, median, or median by contract type and online security): see the *Preparation* page."
        ),
        "dup_title": "Duplicated rows",
        "dup_comment": "Once `customerID` is removed, **{n} rows** are identical to another row on every feature, for example:",
        "dup_comment_2": (
            "These are **different customers** (different IDs) who share the same profile: all are in their first "
            "month (`tenure = 1`), most with a phone-only plan. With so little history, identical profiles are "
            "expected. They are real observations, not data-entry errors, so they are **kept** by default. Removing "
            "them is available as an option (`python run_pipeline.py --drop-duplicates`)."
        ),
    },
    "fr": {
        "eyebrow": "Analyse · 1", "title": "Données",
        "caption": "Jeu de données IBM Telco Customer Churn : une ligne par client.",
        "brief": (
            "Le jeu de données décrit **{rows} clients** d'un opérateur télécom : leur profil, les services utilisés, "
            "leur contrat et leurs factures, et s'ils sont **partis au cours du dernier mois**. Deux petits "
            "nettoyages sont nécessaires avant la modélisation : supprimer l'identifiant client et corriger 11 valeurs vides."
        ),
        "sample": "Échantillon des données brutes", "resample": "Afficher un autre échantillon",
        "shape": "**{rows}** clients et **{cols}** colonnes.",
        "columns_title": "Colonnes",
        "columns": (
            "- `customerID` : identifiant du client, supprimé (voir plus bas).\n"
            "- **Cible** `Churn` : `Yes` si le client est parti au cours du dernier mois, `No` sinon.\n"
            "- **Profil** : `gender`, `SeniorCitizen` (codé 0/1), `Partner`, `Dependents`.\n"
            "- **Services** : `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, "
            "`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.\n"
            "- **Compte** : `tenure` (ancienneté en mois), `Contract`, `PaperlessBilling`, `PaymentMethod`, "
            "`MonthlyCharges`, `TotalCharges`."
        ),
        "types_title": "Types et valeurs manquantes",
        "types_comment": (
            "- Aucune colonne ne contient de valeur nulle à la lecture du fichier.\n"
            "- Seules `SeniorCitizen`, `tenure` et `MonthlyCharges` sont lues comme des nombres ; les 18 autres colonnes sont du texte.\n"
            "- `SeniorCitizen` est numérique mais c'est une variable catégorielle (0/1).\n"
            "- `TotalCharges` devrait être numérique mais elle est lue comme du texte : certaines valeurs ne se convertissent pas (voir plus bas)."
        ),
        "describe_num": "Colonnes numériques", "describe_cat": "Colonnes texte",
        "id_title": "Suppression de `customerID`",
        "id_comment": (
            "`customerID` prend **{n_unique} valeurs différentes pour {n_rows} lignes** : c'est un identifiant "
            "unique. Il ne dit rien du comportement du client, et un modèle ne pourrait que l'apprendre par cœur : "
            "il est donc **supprimé** avant la modélisation."
        ),
        "tc_title": "Nettoyage de `TotalCharges`",
        "tc_comment": (
            "La conversion de `TotalCharges` en nombre échoue car **{n} valeurs sont un espace** (`' '`, visible "
            "ci-dessous). Elles sont converties en valeurs manquantes (NaN).\n\n"
            "**Ces {n} clients ont tous `tenure = 0`** : ils sont arrivés dans le mois et n'ont pas encore été "
            "facturés. Leur total réel est donc 0. Le pipeline propose trois façons de compléter ces valeurs "
            "(0, médiane, ou médiane par type de contrat et sécurité en ligne) : voir la page *Préparation*."
        ),
        "dup_title": "Lignes dupliquées",
        "dup_comment": "Une fois `customerID` supprimé, **{n} lignes** sont identiques à une autre sur toutes les variables, par exemple :",
        "dup_comment_2": (
            "Ce sont des **clients différents** (identifiants différents) qui ont le même profil : tous en sont à leur "
            "premier mois (`tenure = 1`), la plupart avec une offre téléphone seule. Avec si peu d'historique, des "
            "profils identiques sont attendus. Ce sont de vraies observations, pas des erreurs de saisie : elles sont "
            "donc **conservées** par défaut. Leur suppression reste disponible en option "
            "(`python run_pipeline.py --drop-duplicates`)."
        ),
    },
}

header(t(STRINGS, "eyebrow"), t(STRINGS, "title"), t(STRINGS, "caption"))
df = raw_data()
brief(t(STRINGS, "brief", rows=num(len(df))))

# ── Sample ──────────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "sample"))
with controls("sample"):
    if st.button(t(STRINGS, "resample"), icon=":material/shuffle:"):
        st.session_state["data_seed"] = st.session_state.get("data_seed", 0) + 1
st.dataframe(df.sample(10, random_state=st.session_state.get("data_seed", 0)), hide_index=True)
st.markdown(t(STRINGS, "shape", rows=num(df.shape[0]), cols=df.shape[1]))
show_code("df = pd.read_csv('data/customer_churn.csv')\ndf.sample(10)\ndf.shape")

# ── Columns & types ─────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "columns_title"))
st.markdown(t(STRINGS, "columns"))

st.subheader(t(STRINGS, "types_title"))
finding(t(STRINGS, "types_comment"))
with tech():
    info = pd.DataFrame({"dtype": df.dtypes.astype(str), "non-null": df.notna().sum(), "unique": df.nunique()})
    st.dataframe(info, width="stretch")
    st.markdown(f"**{t(STRINGS, 'describe_num')}**")
    st.dataframe(df.describe().round(2).T, width="stretch")
    st.markdown(f"**{t(STRINGS, 'describe_cat')}**")
    st.dataframe(df.drop(columns="customerID").describe(include="str").T, width="stretch")
    show_code("df.info()\ndf.describe().T\ndf.describe(include='str').T")

# ── customerID ──────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "id_title"))
finding(t(STRINGS, "id_comment", n_unique=num(df["customerID"].nunique()), n_rows=num(len(df))))
show_code("df['customerID'].nunique() == len(df)   # True: one ID per row\n"
          "df = df.drop(columns='customerID')\n"
          "\n# In the pipeline (src/churn/data.py, clean_data)\n"
          "df = df.drop(columns=ID_COL)")

# ── TotalCharges ────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "tc_title"))
total = pd.to_numeric(df["TotalCharges"], errors="coerce")
bad = df.loc[total.isna(), ["customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges"]].copy()
bad["TotalCharges"] = bad["TotalCharges"].map(repr)   # make the blank value visible: ' '
finding(t(STRINGS, "tc_comment", n=len(bad)))
st.dataframe(bad, hide_index=True, width="stretch")
show_code(
    "total = pd.to_numeric(df['TotalCharges'], errors='coerce')  # ' ' -> NaN\n"
    "df.loc[total.isna(), ['tenure', 'MonthlyCharges', 'TotalCharges']]  # tenure is always 0\n"
    "\n# In the pipeline (src/churn/data.py)\n"
    "X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')"
)

# ── Duplicates ──────────────────────────────────────────────────────────────
st.subheader(t(STRINGS, "dup_title"))
features = df.drop(columns="customerID")
dup_mask = features.duplicated(keep=False)
st.markdown(t(STRINGS, "dup_comment", n=int(features.duplicated().sum())))
example = features[dup_mask].sort_values(list(features.columns)).head(4)
st.dataframe(df.loc[example.index], hide_index=True)
finding(t(STRINGS, "dup_comment_2"))
show_code("features = df.drop(columns='customerID')\nfeatures.duplicated().sum()  # 22")
