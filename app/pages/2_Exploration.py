import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import CHURN_COLORS, COMMON, NEUTRAL, churn_scale, draw, eda_data, finding, header, pct, setup_page, show_code
from i18n import t
from stats import association, correlation, cramer_ranking, normality, qq_points

setup_page()

STRINGS = {
    "en": {
        "eyebrow": "Analysis · 2", "title": "Exploration",
        "caption": "How each feature is distributed and how strongly it is linked to churn.",
        "tab_overview": "Overview", "tab_target": "Target", "tab_cat": "Categorical features",
        "tab_num": "Numeric features", "tab_corr": "Correlations",
        "overview_comment": (
            "- `tenure`, `MonthlyCharges` and `TotalCharges` have very different scales and large spreads "
            "(`TotalCharges` ranges from {tc_min:,.0f} to {tc_max:,.0f}): linear models will need **scaling**, and "
            "skewed features may benefit from a **transformation**.\n"
            "- `SeniorCitizen` is coded 0/1 but is treated as a categorical feature.\n"
            "- Categorical features have a **low cardinality** (2 to 4 values): one-hot encoding stays compact."
        ),
        "target_comment": (
            "**{no:,}** customers stayed ({no_pct}) and **{yes:,}** left ({yes_pct}). The classes are **imbalanced**: "
            "a model that always predicts \"no churn\" is already {no_pct} accurate while detecting no churner. "
            "Accuracy alone is therefore misleading; recall, precision, F1 and PR-AUC are used to compare models."
        ),
        "choose_cat": "Categorical feature",
        "cat_counts": "Customers per value", "cat_rate": "Churn rate per value",
        "crosstab": "Cross-tabulation with the target",
        "test_line": "Chi-squared test: p-value = {p:.2e} · Cramér's V = {v:.2f} · Tschuprow's T = {tt:.2f}",
        "test_dependent": "The p-value is below 0.05: churn **depends** on `{f}`.",
        "test_independent": "The p-value is above 0.05: no significant link between `{f}` and churn.",
        "cramer_title": "Strength of the link with churn (Cramér's V)",
        "cramer_comment": (
            "- `Contract`, `OnlineSecurity` and `TechSupport` have the **strongest** link with churn.\n"
            "- `gender`, `PhoneService` and `MultipleLines` have a **negligible** link: they bring almost no "
            "information on their own.\n"
            "- `PhoneService` is itself imbalanced: {phone} of customers have a phone service."
        ),
        "pairs_title": "Links between features",
        "pairs_intro": "Some features have a similar link with churn. Are they related to each other?",
        "pairs_comment": (
            "- `StreamingTV` and `StreamingMovies` are **strongly associated** (Cramér's V = {v1:.2f}): customers "
            "usually take both or neither. The information is largely redundant.\n"
            "- `Partner` and `SeniorCitizen` are **almost independent** (V = {v2:.2f}, p = {p2:.2f}): their similar link "
            "with churn is a coincidence."
        ),
        "choose_num": "Numeric feature", "bins": "Number of bins",
        "dist_title": "Distribution", "norm_title": "Normality",
        "norm_table": "Skewness = {s:.2f} · Kurtosis = {k:.2f} · D'Agostino p = {d:.1e} · Kolmogorov-Smirnov p = {ks:.1e} · Shapiro-Wilk p = {sw:.1e}",
        "norm_comment": (
            "All p-values are far below 0.05: **none of the three numeric features is normally distributed**. "
            "`tenure` is U-shaped (many very new and very long-standing customers) and `TotalCharges` is right-skewed. "
            "This does not prevent modelling, but it motivates testing transformations (log, square root, Yeo-Johnson)."
        ),
        "qq_title": "Q-Q plot against a normal distribution",
        "box_title": "Outliers (box plot)",
        "box_comment": "No point lies beyond the whiskers: **no outliers** to handle.",
        "by_target_title": "Distribution by churn",
        "by_target_comment": (
            "- Median `tenure`: **{t_no:.0f}** months for customers who stay, **{t_yes:.0f}** for those who leave. "
            "**{early} of churners leave during their first year.**\n"
            "- Median `MonthlyCharges`: **{m_yes:.1f}** for churners, **{m_no:.1f}** for the others: customers who "
            "leave pay more each month.\n"
            "- Median `TotalCharges`: **{tc_no:,.0f}** for customers who stay, **{tc_yes:,.0f}** for churners (a "
            "consequence of their shorter tenure)."
        ),
        "by_cat_title": "By churn and a categorical feature", "choose_cat2": "Split by",
        "by_cat_comment": (
            "The contract changes the picture. With a **month-to-month** contract, churners are much more recent "
            "(median {m2m_yes:.0f} months against {m2m_no:.0f}). With a **one- or two-year** contract, churners have the "
            "same tenure as the others, or even more: they tend to leave **when their contract ends**, which is the "
            "right moment for a retention offer."
        ),
        "corr_test": "Correlation coefficient", "corr_line": "`{f}`: coefficient = {v:.3f}, p-value = {p:.1e}",
        "corr_title": "Correlation of numeric features with churn",
        "corr_comment": (
            "- `tenure` and `TotalCharges` are **negatively** correlated with churn: the longer a customer has been "
            "there, the less likely they are to leave.\n"
            "- `MonthlyCharges` is **positively** correlated: higher bills, more churn.\n"
            "- `tenure` has the strongest correlation ({v:.2f}).\n"
            "- `TotalCharges` is strongly correlated with `tenure` ({tt:.2f}), since it is roughly tenure × monthly charges."
        ),
        "value": "Value", "churn_rate": "Churn rate", "feature": "Feature",
    },
    "fr": {
        "eyebrow": "Analyse · 2", "title": "Exploration",
        "caption": "Comment chaque variable est distribuée et à quel point elle est liée au churn.",
        "tab_overview": "Vue d'ensemble", "tab_target": "Cible", "tab_cat": "Variables catégorielles",
        "tab_num": "Variables numériques", "tab_corr": "Corrélations",
        "overview_comment": (
            "- `tenure`, `MonthlyCharges` et `TotalCharges` ont des échelles très différentes et une forte dispersion "
            "(`TotalCharges` va de {tc_min:,.0f} à {tc_max:,.0f}) : les modèles linéaires auront besoin d'une "
            "**mise à l'échelle**, et les variables asymétriques peuvent bénéficier d'une **transformation**.\n"
            "- `SeniorCitizen` est codée 0/1 mais elle est traitée comme une variable catégorielle.\n"
            "- Les variables catégorielles ont une **faible cardinalité** (2 à 4 valeurs) : l'encodage one-hot reste compact."
        ),
        "target_comment": (
            "**{no:,}** clients sont restés ({no_pct}) et **{yes:,}** sont partis ({yes_pct}). Les classes sont "
            "**déséquilibrées** : un modèle qui prédit toujours « pas de churn » a déjà {no_pct} d'accuracy sans "
            "détecter un seul churner. L'accuracy seule est donc trompeuse ; les modèles sont comparés avec le "
            "rappel, la précision, le F1 et la PR-AUC."
        ),
        "choose_cat": "Variable catégorielle",
        "cat_counts": "Clients par valeur", "cat_rate": "Taux de churn par valeur",
        "crosstab": "Tableau croisé avec la cible",
        "test_line": "Test du chi² : p-value = {p:.2e} · V de Cramér = {v:.2f} · T de Tschuprow = {tt:.2f}",
        "test_dependent": "La p-value est inférieure à 0,05 : le churn **dépend** de `{f}`.",
        "test_independent": "La p-value est supérieure à 0,05 : pas de lien significatif entre `{f}` et le churn.",
        "cramer_title": "Force du lien avec le churn (V de Cramér)",
        "cramer_comment": (
            "- `Contract`, `OnlineSecurity` et `TechSupport` ont le lien **le plus fort** avec le churn.\n"
            "- `gender`, `PhoneService` et `MultipleLines` ont un lien **négligeable** : seules, elles n'apportent "
            "presque aucune information.\n"
            "- `PhoneService` est elle-même déséquilibrée : {phone} des clients ont un service téléphonique."
        ),
        "pairs_title": "Liens entre variables",
        "pairs_intro": "Certaines variables ont un lien similaire avec le churn. Sont-elles liées entre elles ?",
        "pairs_comment": (
            "- `StreamingTV` et `StreamingMovies` sont **fortement associées** (V de Cramér = {v1:.2f}) : les clients "
            "prennent en général les deux ou aucun. L'information est en grande partie redondante.\n"
            "- `Partner` et `SeniorCitizen` sont **presque indépendantes** (V = {v2:.2f}, p = {p2:.2f}) : leur lien "
            "similaire avec le churn est une coïncidence."
        ),
        "choose_num": "Variable numérique", "bins": "Nombre de classes",
        "dist_title": "Distribution", "norm_title": "Normalité",
        "norm_table": "Asymétrie = {s:.2f} · Aplatissement = {k:.2f} · D'Agostino p = {d:.1e} · Kolmogorov-Smirnov p = {ks:.1e} · Shapiro-Wilk p = {sw:.1e}",
        "norm_comment": (
            "Toutes les p-values sont très inférieures à 0,05 : **aucune des trois variables numériques ne suit une "
            "loi normale**. `tenure` a une forme en U (beaucoup de clients très récents et très anciens) et "
            "`TotalCharges` est asymétrique à droite. Cela n'empêche pas la modélisation, mais justifie de tester "
            "des transformations (log, racine carrée, Yeo-Johnson)."
        ),
        "qq_title": "Diagramme Q-Q par rapport à une loi normale",
        "box_title": "Valeurs aberrantes (boîte à moustaches)",
        "box_comment": "Aucun point au-delà des moustaches : **pas de valeurs aberrantes** à traiter.",
        "by_target_title": "Distribution selon le churn",
        "by_target_comment": (
            "- `tenure` médiane : **{t_no:.0f}** mois pour les clients qui restent, **{t_yes:.0f}** pour ceux qui partent. "
            "**{early} des churners partent pendant leur première année.**\n"
            "- `MonthlyCharges` médiane : **{m_yes:.1f}** pour les churners, **{m_no:.1f}** pour les autres : les "
            "clients qui partent paient plus chaque mois.\n"
            "- `TotalCharges` médiane : **{tc_no:,.0f}** pour les clients qui restent, **{tc_yes:,.0f}** pour les "
            "churners (conséquence de leur ancienneté plus courte)."
        ),
        "by_cat_title": "Selon le churn et une variable catégorielle", "choose_cat2": "Découper par",
        "by_cat_comment": (
            "Le contrat change la lecture. Avec un contrat **au mois**, les churners sont beaucoup plus récents "
            "(médiane de {m2m_yes:.0f} mois contre {m2m_no:.0f}). Avec un contrat **d'un ou deux ans**, les churners ont "
            "la même ancienneté que les autres, voire plus : ils partent plutôt **à l'échéance de leur contrat**, "
            "le bon moment pour une offre de fidélisation."
        ),
        "corr_test": "Coefficient de corrélation", "corr_line": "`{f}` : coefficient = {v:.3f}, p-value = {p:.1e}",
        "corr_title": "Corrélation des variables numériques avec le churn",
        "corr_comment": (
            "- `tenure` et `TotalCharges` sont corrélées **négativement** au churn : plus un client est ancien, "
            "moins il risque de partir.\n"
            "- `MonthlyCharges` est corrélée **positivement** : factures plus élevées, plus de churn.\n"
            "- `tenure` a la corrélation la plus forte ({v:.2f}).\n"
            "- `TotalCharges` est fortement corrélée à `tenure` ({tt:.2f}), car elle vaut à peu près ancienneté × montant mensuel."
        ),
        "value": "Valeur", "churn_rate": "Taux de churn", "feature": "Variable",
    },
}

NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
header(t(STRINGS, "eyebrow"), t(STRINGS, "title"), t(STRINGS, "caption"))
df = eda_data()
CAT = [c for c in df.columns if c not in NUM + ["Churn"]]
churned = df["Churn"] == "Yes"

tabs = st.tabs([t(STRINGS, k) for k in ("tab_overview", "tab_target", "tab_cat", "tab_num", "tab_corr")])

# ── Overview ────────────────────────────────────────────────────────────────
with tabs[0]:
    c1, c2 = st.columns(2, gap="large")
    c1.dataframe(df[NUM].describe().round(2).T)
    c2.dataframe(df[CAT].describe().T)
    finding(t(STRINGS, "overview_comment", tc_min=df.TotalCharges.min(), tc_max=df.TotalCharges.max()))
    show_code("num_col = ['tenure', 'MonthlyCharges', 'TotalCharges']\n"
              "cat_col = [c for c in df.columns if c not in num_col + ['Churn']]\n"
              "df[num_col].describe().T\ndf[cat_col].describe().T")

# ── Target ──────────────────────────────────────────────────────────────────
with tabs[1]:
    counts = df["Churn"].value_counts().rename_axis("Churn").reset_index(name="n")
    counts["share"] = counts["n"] / counts["n"].sum()
    bars = alt.Chart(counts).mark_bar(cornerRadiusEnd=4, size=60).encode(
        x=alt.X("Churn:N", title=None, sort=["No", "Yes"], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("n:Q", title=t(COMMON, "count")),
        color=alt.Color("Churn:N", scale=churn_scale(), legend=None),
        tooltip=["Churn", alt.Tooltip("n:Q", title=t(COMMON, "count")), alt.Tooltip("share:Q", format=".1%")],
    )
    labels = bars.mark_text(dy=-8, color="#0b0b0b").encode(text=alt.Text("share:Q", format=".1%"))
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        draw(bars + labels, 280)
    with right:
        no, yes = int((~churned).sum()), int(churned.sum())
        finding(t(STRINGS, "target_comment", no=no, yes=yes, no_pct=pct(no / len(df)), yes_pct=pct(yes / len(df))))
    show_code("df['Churn'].value_counts(normalize=True)")

# ── Categorical features ────────────────────────────────────────────────────
with tabs[2]:
    feature = st.selectbox(t(STRINGS, "choose_cat"), CAT, index=CAT.index("Contract"))
    per_value = df.groupby(feature).agg(n=("Churn", "size"), rate=("Churn", lambda s: (s == "Yes").mean())).reset_index()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"**{t(STRINGS, 'cat_counts')}**")
        stacked = df.groupby([feature, "Churn"]).size().reset_index(name="n")
        draw(alt.Chart(stacked).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y(f"{feature}:N", title=None),
            x=alt.X("n:Q", title=t(COMMON, "count"), stack=True),
            color=alt.Color("Churn:N", scale=churn_scale(), title="Churn"),
            order=alt.Order("Churn:N", sort="descending"),
            tooltip=[feature, "Churn", alt.Tooltip("n:Q", title=t(COMMON, "count"))],
        ), 60 + 45 * len(per_value))
    with c2:
        st.markdown(f"**{t(STRINGS, 'cat_rate')}**")
        rate = alt.Chart(per_value).mark_bar(cornerRadiusEnd=4, color=CHURN_COLORS["Yes"]).encode(
            y=alt.Y(f"{feature}:N", title=None),
            x=alt.X("rate:Q", title=t(STRINGS, "churn_rate"), axis=alt.Axis(format="%")),
            tooltip=[feature, alt.Tooltip("rate:Q", format=".1%", title=t(STRINGS, "churn_rate")),
                     alt.Tooltip("n:Q", title=t(COMMON, "count"))],
        )
        overall = alt.Chart(pd.DataFrame({"x": [churned.mean()]})).mark_rule(color=NEUTRAL, strokeDash=[4, 4]).encode(x="x:Q")
        draw(rate + rate.mark_text(align="left", dx=4, color="#0b0b0b").encode(text=alt.Text("rate:Q", format=".0%")) + overall,
             60 + 45 * len(per_value))

    st.markdown(f"**{t(STRINGS, 'crosstab')}**")
    st.dataframe(pd.crosstab(df[feature], df["Churn"], margins=True))
    a = association(df, feature, "Churn")
    st.markdown(t(STRINGS, "test_line", p=a["p_value"], v=a["cramer_v"], tt=a["tschuprow_t"]))
    st.markdown(t(STRINGS, "test_dependent" if a["p_value"] < 0.05 else "test_independent", f=feature))
    show_code(
        "cross_tab = pd.crosstab(df[feature], df['Churn'])\n"
        "chi2, p_value, _, _ = chi2_contingency(cross_tab)\n"
        "n = cross_tab.to_numpy().sum()\nr, k = cross_tab.shape\n"
        "cramer_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))\n"
        "tschuprow_t = np.sqrt(chi2 / (n * np.sqrt((r - 1) * (k - 1))))"
    )

    st.subheader(t(STRINGS, "cramer_title"))
    ranking = cramer_ranking(df, CAT, "Churn")
    chart = alt.Chart(ranking).mark_bar(cornerRadiusEnd=4, color=CHURN_COLORS["No"]).encode(
        x=alt.X("feature:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("cramer_v:Q", title="Cramér's V"),
        tooltip=[alt.Tooltip("feature:N", title=t(STRINGS, "feature")), alt.Tooltip("cramer_v:Q", format=".2f")],
    )
    draw(chart + chart.mark_text(dy=-8, color="#0b0b0b").encode(text=alt.Text("cramer_v:Q", format=".2f")), 340)
    finding(t(STRINGS, "cramer_comment", phone=pct((df.PhoneService == "Yes").mean(), 0)))

    st.subheader(t(STRINGS, "pairs_title"))
    st.markdown(t(STRINGS, "pairs_intro"))
    a1, a2 = association(df, "StreamingMovies", "StreamingTV"), association(df, "Partner", "SeniorCitizen")
    finding(t(STRINGS, "pairs_comment", v1=a1["cramer_v"], v2=a2["cramer_v"], p2=a2["p_value"]))
    show_code("association(df, 'StreamingMovies', 'StreamingTV')\nassociation(df, 'Partner', 'SeniorCitizen')")

# ── Numeric features ────────────────────────────────────────────────────────
with tabs[3]:
    c1, c2 = st.columns([2, 1])
    feature = c1.selectbox(t(STRINGS, "choose_num"), NUM)
    bins = c2.slider(t(STRINGS, "bins"), 4, 30, 12)
    data = df[[feature, "Churn"]].dropna()
    st.dataframe(data[feature].describe().to_frame().T.round(2))

    st.subheader(t(STRINGS, "dist_title"))
    hist = alt.Chart(data).mark_bar(color=CHURN_COLORS["No"], cornerRadiusEnd=2).encode(
        x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=bins), title=feature),
        y=alt.Y("count():Q", title=t(COMMON, "count")),
        tooltip=[alt.Tooltip(f"{feature}:Q", bin=alt.Bin(maxbins=bins)), alt.Tooltip("count():Q", title=t(COMMON, "count"))],
    )
    draw(hist, 260)

    st.subheader(t(STRINGS, "norm_title"))
    nm = normality(data[feature])
    st.markdown(t(STRINGS, "norm_table", s=nm["skewness"], k=nm["kurtosis"], d=nm["dagostino_p"], ks=nm["ks_p"], sw=nm["shapiro_p"]))
    qq, slope, intercept = qq_points(data[feature])
    qq_chart = alt.Chart(qq.sample(min(len(qq), 2000), random_state=0)).mark_circle(size=12, color=CHURN_COLORS["No"], opacity=0.6).encode(
        x=alt.X("theoretical:Q", title="Theoretical quantiles"), y=alt.Y("observed:Q", title=feature))
    line_df = pd.DataFrame({"theoretical": [qq.theoretical.min(), qq.theoretical.max()]})
    line_df["observed"] = intercept + slope * line_df["theoretical"]
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown(f"**{t(STRINGS, 'qq_title')}**")
        draw(qq_chart + alt.Chart(line_df).mark_line(color=CHURN_COLORS["Yes"], strokeWidth=2).encode(x="theoretical:Q", y="observed:Q"), 280)
    with right:
        finding(t(STRINGS, "norm_comment"))
    show_code("skew, kurt = x.skew(), x.kurt()\nstats.normaltest(x).pvalue\n"
              "stats.kstest((x - x.mean()) / x.std(), 'norm').pvalue\nstats.shapiro(x.sample(5000)).pvalue\n"
              "stats.probplot(x, dist='norm')")

    st.subheader(t(STRINGS, "box_title"))
    draw(alt.Chart(data).mark_boxplot(color=CHURN_COLORS["No"], size=40, ticks=True).encode(x=alt.X(f"{feature}:Q", title=feature)), 120)
    st.markdown(t(STRINGS, "box_comment"))

    st.subheader(t(STRINGS, "by_target_title"))
    left, right = st.columns(2, gap="large")
    with left:
        draw(alt.Chart(data).mark_boxplot(size=50).encode(
            x=alt.X("Churn:N", title="Churn", sort=["No", "Yes"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{feature}:Q", title=feature),
            color=alt.Color("Churn:N", scale=churn_scale(), legend=None)), 300)
    with right:
        draw(alt.Chart(data).transform_bin("bin", feature, bin=alt.Bin(maxbins=bins)).transform_aggregate(
            n="count()", groupby=["bin", "Churn"]).transform_joinaggregate(total="sum(n)", groupby=["Churn"]
        ).transform_calculate(share="datum.n / datum.total").mark_bar(opacity=0.85, cornerRadiusEnd=2).encode(
            x=alt.X("bin:Q", bin="binned", title=feature),
            y=alt.Y("share:Q", title=t(COMMON, "share"), axis=alt.Axis(format="%"), stack=None),
            xOffset="Churn:N",
            color=alt.Color("Churn:N", scale=churn_scale()),
            tooltip=["Churn", alt.Tooltip("bin:Q", title=feature), alt.Tooltip("share:Q", format=".1%"), alt.Tooltip("n:Q", title=t(COMMON, "count"))],
        ), 300)
    med = df.groupby("Churn")[NUM].median()
    finding(t(STRINGS, "by_target_comment", t_no=med.loc["No", "tenure"], t_yes=med.loc["Yes", "tenure"],
              m_no=med.loc["No", "MonthlyCharges"], m_yes=med.loc["Yes", "MonthlyCharges"],
              tc_no=med.loc["No", "TotalCharges"], tc_yes=med.loc["Yes", "TotalCharges"],
              early=pct((df.loc[churned, "tenure"] < 12).mean(), 0)))

    st.subheader(t(STRINGS, "by_cat_title"))
    split = st.selectbox(t(STRINGS, "choose_cat2"), CAT, index=CAT.index("Contract"))
    draw(alt.Chart(data.join(df[split])).mark_boxplot(size=26).encode(
        x=alt.X(f"{split}:N", title=split, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Churn:N", sort=["No", "Yes"]),
        y=alt.Y(f"{feature}:Q", title=feature),
        color=alt.Color("Churn:N", scale=churn_scale())), 320)
    if split == "Contract" and feature == "tenure":
        m2m = df[df.Contract == "Month-to-month"].groupby("Churn")["tenure"].median()
        finding(t(STRINGS, "by_cat_comment", m2m_yes=m2m["Yes"], m2m_no=m2m["No"]))

# ── Correlations ────────────────────────────────────────────────────────────
with tabs[4]:
    method = st.radio(t(STRINGS, "corr_test"), ["pearson", "spearman", "kendall"], horizontal=True)
    y01 = churned.astype(int)
    rows = []
    for f in NUM:
        mask = df[f].notna()
        v, p = correlation(df.loc[mask, f], y01[mask], method)
        rows.append({"feature": f, "coefficient": v, "p": p})
        st.markdown(t(STRINGS, "corr_line", f=f, v=v, p=p))
    corr = pd.DataFrame(rows)
    st.subheader(t(STRINGS, "corr_title"))
    chart = alt.Chart(corr).mark_bar(cornerRadiusEnd=4, size=50).encode(
        x=alt.X("feature:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("coefficient:Q", title=method.capitalize(), scale=alt.Scale(domain=[-0.5, 0.5])),
        color=alt.condition("datum.coefficient > 0", alt.value(CHURN_COLORS["Yes"]), alt.value(CHURN_COLORS["No"])),
        tooltip=["feature", alt.Tooltip("coefficient:Q", format=".3f"), alt.Tooltip("p:Q", format=".1e")],
    )
    text = chart.mark_text(dy=alt.expr("datum.coefficient > 0 ? -10 : 12"), color="#0b0b0b").encode(
        text=alt.Text("coefficient:Q", format=".2f"))
    draw(chart + text + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=NEUTRAL).encode(y="y:Q"), 300)
    tt = df[["tenure", "TotalCharges"]].corr(method=method).iloc[0, 1]
    finding(t(STRINGS, "corr_comment", v=corr.loc[corr.feature == "tenure", "coefficient"].iloc[0], tt=tt))
    show_code("df[num_col + ['Churn01']].corr(method='pearson')['Churn01']\nstats.pearsonr(df['tenure'], df['Churn01'])")
