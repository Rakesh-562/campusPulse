"""z / t / F / chi-square tests with effect sizes + Bonferroni."""
import numpy as np, pandas as pd
from scipy import stats


def two_proportion_z(x1, n1, x2, n2):
    p1, p2, p = x1 / n1, x2 / n2, (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return dict(diff=p1 - p2, z=z, p=2 * stats.norm.sf(abs(z)))


def welch_t(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return dict(diff=a.mean() - b.mean(), t=t, p=p, cohens_d=(a.mean() - b.mean()) / sp)


def anova_f(*groups):
    groups = [pd.Series(g).dropna() for g in groups]
    f, p = stats.f_oneway(*groups)
    allv = pd.concat(groups)
    ssb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    return dict(F=f, p=p, eta_sq=ssb / ((allv - allv.mean()) ** 2).sum())


def chi2_test(x, y):
    tab = pd.crosstab(x, y)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    return dict(chi2=chi2, p=p, dof=dof, cramers_v=np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))


def bonferroni(pvalues, alpha=0.05):
    pv = np.asarray(pvalues, float)
    return pd.DataFrame(dict(p=pv, p_adj=np.minimum(pv * len(pv), 1), significant=pv * len(pv) < alpha))


def conclude(p: float, if_significant: str, alpha: float = 0.05, n_tests: int = 1) -> str:
    """Decision sentence computed from the p-value (Bonferroni for n_tests), never hard-coded."""
    a = alpha / n_tests
    if p < a:
        return f"DECISION: reject H0 (p = {p:.2e} < {a:.3g}). {if_significant}"
    return f"DECISION: fail to reject H0 (p = {p:.2e} >= {a:.3g}). No evidence for: {if_significant}"
