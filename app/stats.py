"""Statistical tests used in the exploration page."""
import numpy as np
import pandas as pd
from scipy import stats


def association(df, a, b):
    """Chi-squared test of independence between two categorical variables, with Cramér's V and Tschuprow's T."""
    table = pd.crosstab(df[a], df[b])
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    cramer_v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    tschuprow_t = np.sqrt(chi2 / (n * np.sqrt((r - 1) * (k - 1))))
    return {"chi2": chi2, "p_value": p_value, "cramer_v": cramer_v, "tschuprow_t": tschuprow_t}


def cramer_ranking(df, features, target):
    rows = [{"feature": f, "cramer_v": association(df, f, target)["cramer_v"]} for f in features]
    return pd.DataFrame(rows).sort_values("cramer_v", ascending=False).reset_index(drop=True)


def correlation(x, y, method):
    """Correlation between a numeric feature and the 0/1 target. method: pearson | spearman | kendall."""
    func = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    res = func(x, y)
    return float(res[0]), float(res[1])


def normality(x):
    """Shape statistics and normality tests. Values are standardised for the KS test (it compares to N(0, 1))."""
    x = pd.Series(x).dropna()
    z = (x - x.mean()) / x.std()
    return {
        "skewness": float(x.skew()),
        "kurtosis": float(x.kurt()),
        "dagostino_p": float(stats.normaltest(x).pvalue),
        "ks_p": float(stats.kstest(z, "norm").pvalue),
        # Shapiro-Wilk is designed for n <= 5000: computed on a reproducible sample of 5000 values
        "shapiro_p": float(stats.shapiro(x.sample(min(len(x), 5000), random_state=0)).pvalue),
    }


def qq_points(x):
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(pd.Series(x).dropna(), dist="norm")
    return pd.DataFrame({"theoretical": theoretical, "observed": ordered}), slope, intercept
