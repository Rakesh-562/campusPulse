"""Export every number the dashboard shows into dashboard/data.js.

The dashboard (plain HTML/CSS/JS in dashboard/) never computes or types a result itself: it only reads
what this script writes. Re-run this after changing a pipeline or adding data:

    python scripts/export_dashboard_data.py

Pulses whose data is not on this machine are exported with "available": false and the dashboard says so.
"""
from __future__ import annotations
import json, math, os, sys, traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from campuspulse.common.decisions import (robustness_matrix, status_matrix, fragility_table,
                                          survival_summary, defaults, DecisionLog)

OUT = ROOT / "dashboard" / "data.js"

META = {
    "academic": dict(num="①", title="Academic", question="Who drops out, and how early can we tell?",
                     dataset="UCI 697 · Predict Students' Dropout and Academic Success",
                     source="Polytechnic Institute of Portalegre, Portugal · 4,424 students · 2008–2019 cohorts",
                     licence="CC BY 4.0", challenge="Grades of 0 that are really missing; unresolved 'Enrolled' outcomes"),
    "wellbeing": dict(num="②", title="Wellbeing", question="Which behaviours co-occur with persistent sadness?",
                      dataset="CDC YRBS + Malaysia university mental-health survey",
                      source="US Youth Risk Behavior Survey (high school) · Malaysia survey n=101 (2020)",
                      licence="Public domain / open", challenge="Skipped sensitive answers, survey weights, 49 course spellings"),
    "energy": dict(num="③", title="Energy", question="When does a campus building behave differently, and why?",
                   dataset="I-BLEND · campus building electricity",
                   source="IIIT Delhi, India · 52 months of 1-minute smart-meter data · institute calendar (teaching vs low-activity days)",
                   licence="CC BY 4.0", challenge="Hostel meters silent in ~25% of hours; spikes; vacation falls in the hottest months"),
}


def clean_json(o):
    """Make pandas / numpy objects JSON-safe (NaN -> null)."""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return clean_json(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return clean_json(o.to_dict())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (o is None or math.isnan(o) or math.isinf(o)) else round(float(o), 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, pd.Interval) or hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            return str(o)
    return o


def test_row(name, d, stat_key, effect_key=None, effect_name=None):
    return dict(test=name, statistic=d.get(stat_key), p=d.get("p"),
                effect=d.get(effect_key) if effect_key else None, effect_name=effect_name)


def robustness_block(P):
    vals, _ = robustness_matrix(P.run_pipeline, P.DECISIONS, P.FINDINGS)
    st = status_matrix(vals)
    frag = fragility_table(vals, P.FINDINGS).drop(columns="swing")
    return dict(values={f: {c: vals.loc[f, c] for c in vals.columns} for f in vals.index},
                status={f: {c: st.loc[f, c] for c in st.columns} for f in st.index},
                variants=list(vals.columns), summary=survival_summary(vals), fragility=frag)


# ---------------------------------------------------------------- shared payload builders
def describe(df: pd.DataFrame, cols: dict) -> list:
    """Descriptive statistics (L3): {column: display label}."""
    rows = []
    for c, label in cols.items():
        if c not in df:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append(dict(variable=label, n=int(s.notna().sum()), missing=int(s.isna().sum()), mean=s.mean(),
                         sd=s.std(), min=s.min(), q1=s.quantile(.25), median=s.median(), q3=s.quantile(.75),
                         max=s.max(), skew=s.skew()))
    return rows


def histogram(values, bins, label, note=""):
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    counts, edges = np.histogram(v, bins=bins)
    return dict(label=label, note=note, bins=[dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=int(counts[i]))
                                              for i in range(len(counts))])


def scatter(points2d, categories, n=700, seed=0):
    """Random subsample of 2-D points with a category per point (for PCA / t-SNE / cluster maps)."""
    pts = np.asarray(points2d)[:, :2]
    cats = np.asarray(categories)
    idx = np.random.default_rng(seed).permutation(len(pts))[:n]
    return [dict(x=float(pts[i, 0]), y=float(pts[i, 1]), c=str(cats[i])) for i in idx]


def dr_block(X: pd.DataFrame, labels, label_names: dict, tsne_n=500, n_comp=6, sample=None):
    """PCA scree + PCA scatter + t-SNE scatter (L12-14), all on the standardised matrix X."""
    from campuspulse.common import reduce, clustering
    X = X.astype(float)
    labels = pd.Series(labels, index=X.index).map(lambda v: label_names.get(v, str(v)))
    k = min(n_comp, X.shape[1])
    pts, var, model = reduce.pca(X, n=k)
    loadings = pd.DataFrame(model.components_[:2].T, index=X.columns, columns=["pc1", "pc2"])
    top = loadings.assign(w=loadings.abs().max(axis=1)).sort_values("w", ascending=False).head(6)
    out = dict(variance=[float(v) for v in var], n_features=int(X.shape[1]), n_rows=int(X.shape[0]),
               n_unique=int(X.drop_duplicates().shape[0]),
               pca=scatter(pts, labels.values), categories=list(label_names.values()),
               top_loadings=[dict(feature=i, pc1=r.pc1, pc2=r.pc2) for i, r in top.iterrows()])
    sub = clustering.sample_rows(X, tsne_n)
    out["tsne"] = scatter(reduce.tsne(sub, perplexity=min(30, max(5, len(sub) // 4))), labels.loc[sub.index].values, n=tsne_n)
    return out


def cluster_map(X: pd.DataFrame, k: int, sample_n: int, profile_df: pd.DataFrame | None = None):
    """Same random sample and K-Means as each pipeline's run_clustering, drawn on its first two PCs."""
    from campuspulse.common import clustering, reduce
    Xs = clustering.sample_rows(X.astype(float), sample_n)
    metrics, labels = clustering.run_all(Xs.values, k=k)
    pts, _, _ = reduce.pca(Xs, n=2)
    names = [("Group " + str(int(l) + 1)) for l in labels["kmeans"]]
    noise = labels["dbscan"] == -1
    from sklearn.metrics import adjusted_rand_score
    agree = adjusted_rand_score(labels["kmeans"][~noise], labels["dbscan"][~noise]) if (~noise).sum() > 1 else float("nan")
    return dict(points=scatter(pts, names, n=sample_n), groups=sorted(set(names)),
                noise_share=float(noise.mean()), metrics=metrics, dbscan_kmeans_agreement=float(agree))


def safe(label, fn):
    """Optional dashboard sections: a failure is reported, never fatal for the pulse."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"   (skipped {label}: {type(e).__name__}: {e})")
        return None


# ---------------------------------------------------------------- pulse-specific extras
def extras_academic(P):
    D = defaults(P.DECISIONS)
    raw = P.load_raw()
    zero = (raw[P.GRADE] == 0) & (raw[P.ENROLLED1] > 0)
    outcome = raw["Target"].value_counts()
    df = P.clean(D)
    st = P.run_stats(df)
    a3 = {opt: P.run_pipeline({**D, "zero_grade": opt})["A3"] for opt in P.DECISIONS["zero_grade"]}
    _, strat = P.stratified_debt_effect(df)
    metrics, profiles = P.run_clustering(df)
    rules = P.run_rules(df).head(8)
    dec = P.run_decomposition(df)
    return dict(
        outcomes=[dict(label=k, n=int(v)) for k, v in outcome.items()],
        hidden_zeros=dict(n=int(zero.sum()), n_after_enrolled=int((zero & (raw["Target"] != "Enrolled")).sum()), dropout_rate=float((raw.loc[zero, "Target"] == "Dropout").mean()),
                          dropout_rate_others=float((raw.loc[~zero, "Target"] == "Dropout").mean()),
                          no_units=int((raw[P.ENROLLED1] == 0).sum()),
                          no_units_graduated=int(((raw[P.ENROLLED1] == 0) & (raw["Target"] == "Graduate")).sum())),
        a3_by_zero_grade=[dict(option=k, value=v) for k, v in a3.items()],
        timeline=P.run_timeline(df),
        debt_by_band=strat,
        tests=[test_row("Debtor × dropout (χ²)", st["debtor_chi2"], "chi2", "cramers_v", "Cramér's V"),
               test_row("Scholarship × dropout (χ²)", st["scholarship_chi2"], "chi2", "cramers_v", "Cramér's V"),
               test_row("Semester-1 grade, persisters vs dropouts (Welch t)", st["grade_t"], "t", "cohens_d", "Cohen's d"),
               test_row("Debtor dropout rate (two-proportion z)", st["debtor_z"], "z", "diff", "rate difference"),
               test_row("Grade across 4 largest courses (ANOVA F)", st["course_anova"], "F", "eta_sq", "η²")],
        clusters=dict(metrics=metrics, profiles=profiles.reset_index().rename(columns={"index": "cluster"})),
        rules=rules, pca_variance=list(dec["pca_variance"]),
        describe=describe(df, {"Age at enrollment": "Age at enrolment", "Admission grade": "Admission grade (0–200)",
                               P.GRADE: "Semester-1 grade (0–20, zeros = missing)", "approval_rate_sem1": "Semester-1 approval rate",
                               P.ENROLLED1: "Semester-1 units enrolled", "Curricular units 2nd sem (grade)": "Semester-2 grade (0–20, raw)"}),
        histograms=[histogram(raw[P.GRADE], np.arange(0, 21, 1), "Semester-1 grade in the raw file (all 4,424 students)",
                              "The spike at 0 is the hidden-missing problem: real grades start around 10."),
                    histogram(raw["Age at enrollment"], np.arange(15, 72, 3), "Age at enrolment",
                              "Right-skewed: a long tail of mature students, kept as real (IQR flags them, they are not errors).")],
        dr=dr_block(P.get_feature_matrix(df), df["dropout"].values, {0: "Graduated", 1: "Dropped out"}),
        cluster_map=cluster_map(P.get_feature_matrix(df), 3, 1000),
    )


def extras_wellbeing(P):
    D = defaults(P.DECISIONS)
    yr, mh = P.clean(D)
    st = P.run_stats(yr)
    gradient = yr.groupby("sleep_hours").apply(lambda g: np.average(g["sadness"], weights=g["w"])).rename("rate").reset_index()
    return dict(sleep_gradient=gradient,
                course_association={opt: P.course_depression_association(P.clean({**D, "course_standardization": opt})[1])
                                    for opt in P.DECISIONS["course_standardization"]},
                tests=[test_row("Sleep deprivation × sadness (χ²)", st["sleep_chi2"], "chi2", "cramers_v", "Cramér's V"),
                       test_row("Bullied × sadness (χ²)", st["bully_chi2"], "chi2", "cramers_v", "Cramér's V"),
                       test_row("Inactive × sadness (χ²)", st["inactivity_chi2"], "chi2", "cramers_v", "Cramér's V"),
                       test_row("Heavy social media × sadness (χ²)", st["social_media_chi2"], "chi2", "cramers_v", "Cramér's V")],
                rules=P.mine_sadness_rules(yr).head(8),
                clusters=dict(metrics=P.run_clustering(yr)[0], profiles=P.run_clustering(yr)[1].reset_index()),
                describe=describe(yr, {"sleep_hours": "Hours of sleep on a school night", "active_days": "Days physically active (0–7)",
                                       "sadness": "Persistent sadness (share)", "bullied": "Bullied at school or online (share)",
                                       "heavy_social_media": "Social media several times a day or more (share)"}),
                skipped=[dict(question=k, share=float(v)) for k, v in P.load_raw()[0][["sad", "sleep_code", "active_code",
                         "social_media_code", "bullied_school"]].isna().mean().items()],
                histograms=[histogram(yr["sleep_hours"], np.arange(3.5, 11, 1), "Hours of sleep on a school night",
                                      "4 = four hours or less, 10 = ten or more. CDC counts fewer than 8 as not enough.")],
                dr=safe("dimensionality reduction", lambda: dr_block(P.get_behavior_matrix(yr), yr["sadness"].values,
                                                                     {0: "No persistent sadness", 1: "Persistent sadness"})),
                cluster_map=safe("cluster map", lambda: cluster_map(P.get_behavior_matrix(yr), 4, 1200)))


def extras_energy(P):
    D = defaults(P.DECISIONS)
    df = P.clean(D)
    st = P.run_stats(df)
    w = st["within_month_t"]
    metrics, profiles = P.run_clustering(df)
    raw = P.load_raw()
    gaps = [dict(meter=m, share=float(raw[m].isna().mean())) for m in P.METERS[:9] if m in raw]
    return dict(by_month=st["by_month"], gaps=gaps,
                tests=[test_row("Academic building by day, teaching vs low-activity (Welch t)", st["teaching_t"], "t", "cohens_d", "Cohen's d"),
                       test_row("Hostels at night, term vs vacation (Welch t)", st["term_t"], "t", "cohens_d", "Cohen's d"),
                       dict(test="Term − vacation, same month only (one-sample t)", statistic=w["t"], p=w["p"],
                            effect=w["mean_diff"], effect_name="mean W")],
                rules=P.mine_energy_rules(df).head(8),
                clusters=dict(metrics=metrics, profiles=profiles.reset_index()),
                describe=describe(df, {"Academic": "Academic building (W)", "Library": "Library (W)", "Mess": "Dining hall (W)",
                                       "total_hostel_power": "All hostels (W)", "total_campus_power": "Whole campus (W)"}),
                histograms=[histogram(df["total_hostel_power"] / 1000, 20, "Hourly hostel power (kW)",
                                      "Hours where any hostel meter was silent were dropped (the default meter-gap choice).")],
                dr=safe("dimensionality reduction", lambda: energy_dr(P, df)),
                cluster_map=safe("cluster map", lambda: cluster_map(P.get_daily_profile_matrix(df), 3, 2000)))


def energy_dr(P, df):
    """PCA of daily 24-hour academic-building load shapes, coloured by teaching day vs low-activity day."""
    prof = P.get_daily_profile_matrix(df)
    teaching = df.assign(date=df["datetime"].dt.date).groupby("date")["is_teaching"].max().reindex(prof.index)
    return dr_block(prof, teaching.values, {1: "Teaching day", 0: "Low-activity day"})


# ---------------------------------------------------------------- interpretations (words + computed numbers)

def _pc1_mean(D, category):
    xs = [q["x"] for q in D["pca"] if q["c"] == category]
    return float(np.mean(xs)) if xs else float("nan")


def _metric(M, method, key):
    metrics = M["metrics"].to_dict("records") if isinstance(M["metrics"], pd.DataFrame) else M["metrics"]
    row = next((m for m in metrics if m["method"] == method), {})
    return row.get(key)


def interpret_academic(X):
    D, M, prof = X["dr"], X["cluster_map"], X["clusters"]["profiles"]
    v = D["variance"]
    g = {int(r["cluster"]): r for r in prof.to_dict("records") if r["cluster"] != "DBSCAN noise"}
    noise = next((r for r in prof.to_dict("records") if r["cluster"] == "DBSCAN noise"), None)
    dr = [
        f"PC1 ({v[0]:.0%} of the variance) is course progress: it loads on the number of units enrolled in and passed in both "
        f"semesters. Graduates sit at {_pc1_mean(D, 'Graduated'):+.1f} on PC1 on average, dropouts at {_pc1_mean(D, 'Dropped out'):+.1f}.",
        f"PC2 ({v[1]:.0%}) sets the semester-1 pass rate against age: young students who pass a large share of their units sit high.",
        "t-SNE tells the same story: graduates and dropouts form mostly separate neighbourhoods, with an overlap zone between them.",
        f"Only PC1 and PC2 are worth reading: from PC3 on, each component adds {v[2]:.0%} or less.",
    ]
    cl = [
        f"Silhouette {_metric(M, 'kmeans', 'silhouette'):.2f}: the groups are real but overlap (above 0.5 would mean clearly separated groups).",
        f"Group 1 ({g[0]['size']:.0f} students) looks like transfer students: {g[0]['units_enrolled']:.0f} units enrolled, "
        f"{g[0]['units_credited']:.0f} of them credited from earlier study, mean age {g[0]['mean_age']:.0f}. {g[0]['dropout_rate']:.0%} drop out.",
        f"Group 2 ({g[1]['size']:.0f}) passed almost nothing: {g[1]['units_passed']:.1f} of {g[1]['units_enrolled']:.1f} units on average. "
        f"{g[1]['dropout_rate']:.0%} drop out: this is the at-risk group.",
        f"Group 3 ({g[2]['size']:.0f}) is the typical student: about 21 years old, passes {g[2]['units_passed']:.1f} of {g[2]['units_enrolled']:.1f} "
        f"units, {g[2]['scholarship_rate']:.0%} hold a scholarship. {g[2]['dropout_rate']:.0%} drop out.",
    ]
    if noise:
        cl.append(f"DBSCAN leaves {noise['size']:.0f} students outside every dense group; {noise['dropout_rate']:.0%} of them drop out, "
                  "so unusual profiles are a warning sign too.")
    return dr, cl


def interpret_wellbeing(X):
    D, M, prof = X["dr"], X["cluster_map"], X["clusters"]["profiles"]
    v = D["variance"]
    g = {int(r["group"]): r for r in prof.to_dict("records")}
    dr = [
        f"The variance is spread almost evenly ({', '.join(f'{x:.0%}' for x in v)}): there is no dominant direction, so PCA cannot "
        "compress these five yes/no behaviours. Each one carries its own information.",
        f"PC1 ({v[0]:.0%}) loads positively on every risk (female, bullied, sleep-deprived): a general-risk axis. Students reporting "
        f"persistent sadness sit at {_pc1_mean(D, 'Persistent sadness'):+.2f} on PC1, others at {_pc1_mean(D, 'No persistent sadness'):+.2f}: "
        "a shift, not a separation.",
        f"PC2 ({v[1]:.0%}) sets inactivity against heavy social media.",
        f"t-SNE cannot be interpreted here: only {D['n_unique']} different answer combinations exist, so the map just scatters copies of "
        "the same few points.",
    ]
    cl = [
        "With yes/no answers, K-Means groups students by the answers themselves, so the groups are answer combinations rather "
        "than hidden types. They are still useful for comparing sadness rates:",
        f"Group 1 ({g[0]['size']:.0f}): never active ({g[0]['inactive_rate']:.0%}), mostly short sleep: {g[0]['sadness_rate']:.0%} persistent sadness.",
        f"Group 2 ({g[1]['size']:.0f}): girls, all sleeping under 8 hours, active: {g[1]['sadness_rate']:.0%}, the highest.",
        f"Group 3 ({g[2]['size']:.0f}): enough sleep, active: {g[2]['sadness_rate']:.0%}, the lowest.",
        f"Group 4 ({g[3]['size']:.0f}): boys, all sleeping under 8 hours, active: {g[3]['sadness_rate']:.0%}. Same sleep as group 2, "
        "much less sadness: sex matters as much as sleep.",
        f"DBSCAN and OPTICS cannot be interpreted here: identical answers form perfectly dense 'groups' of single combinations "
        f"(OPTICS silhouette {_metric(M, 'optics', 'silhouette'):.2f} looks excellent only for that reason).",
    ]
    return dr, cl


def interpret_energy(X):
    D, M, prof = X["dr"], X["cluster_map"], X["clusters"]["profiles"]
    v = D["variance"]
    g = {int(r["group"]): r for r in prof.to_dict("records")}
    dr = [
        f"Each day is a 24-hour load curve of the academic building. PC1 ({v[0]:.0%}) is the overall daytime level: every hour loads "
        f"positively, most around midday. Teaching days sit at {_pc1_mean(D, 'Teaching day'):+.1f}, low-activity days at "
        f"{_pc1_mean(D, 'Low-activity day'):+.1f}: the calendar is the main source of variation.",
        f"PC2 ({v[1]:.0%}) is the early-morning load (3:00 to 7:00). Both day types average close to 0 on it, so this base load "
        "is not explained by the teaching calendar.",
        "t-SNE separates teaching days from low-activity days into two regions, matching PC1.",
        f"Together PC1 and PC2 explain {v[0] + v[1]:.0%}: a day's 24 numbers are well summarised by two.",
    ]
    cl = [
        f"Silhouette {_metric(M, 'kmeans', 'silhouette'):.2f}: moderately clear groups of days.",
        f"Group 1 ({g[0]['days']:.0f} days): low-activity days, {g[0]['weekend_share']:.0%} weekends and {g[0]['vacation_share']:.0%} "
        f"vacation; about {g[0]['noon_power'] / 1000:.0f} kW at noon.",
        f"Group 2 ({g[1]['days']:.0f} days): normal teaching days ({g[1]['teaching_share']:.0%} teaching); about {g[1]['noon_power'] / 1000:.0f} kW at noon.",
        f"Group 3 ({g[2]['days']:.0f} days): high-load teaching days ({g[2]['teaching_share']:.0%} teaching, no vacation); "
        f"about {g[2]['noon_power'] / 1000:.0f} kW at noon.",
        f"DBSCAN also finds {_metric(M, 'dbscan', 'n_clusters'):.0f} dense groups, but they only partly match the K-Means groups "
        f"(adjusted Rand index {M['dbscan_kmeans_agreement']:.2f}, where 1 = identical), and it leaves "
        f"{_metric(M, 'dbscan', 'noise_share'):.0%} of days as unusual. We did not check what happened on those days.",
    ]
    return dr, cl


def forensics():
    """Why we dropped the first dataset: three fingerprints of generated data."""
    from sklearn.tree import DecisionTreeClassifier
    f = ROOT / "data" / "raw" / "legacy" / "student_lifestyle_dataset.csv"
    if not f.exists():
        return dict(available=False)
    df = pd.read_csv(f, encoding="utf-8-sig")
    hours = [c for c in df.columns if "Hours" in c]
    s = df[hours].sum(axis=1)
    acc = {d: DecisionTreeClassifier(max_depth=d, random_state=0).fit(df[hours], df["Stress_Level"]).score(df[hours], df["Stress_Level"])
           for d in (1, 2, 3)}
    return dict(available=True, rows=len(df), rows_summing_to_24=int((s.round(6) == 24).sum()),
                tree_accuracy=[dict(depth=k, accuracy=v) for k, v in acc.items()],
                missing=int(df.isna().sum().sum()), duplicates=int(df.drop(columns="Student_ID").duplicated().sum()))


def first(cleaned):
    """clean() returns a table, or (table, extra table) for wellbeing."""
    return cleaned[0] if isinstance(cleaned, tuple) else cleaned


def main():
    import importlib
    out = dict(generated=datetime.now().strftime("%Y-%m-%d %H:%M"), pulses={}, forensics=forensics())
    for name in META:
        P = importlib.import_module(f"campuspulse.{name}.pipeline")
        block = dict(meta=META[name], decisions=P.DECISIONS, findings=P.FINDINGS)
        try:
            log = DecisionLog()
            P.clean(defaults(P.DECISIONS), log)
            extras = globals()[f"extras_{name}"](P)
            extras["dr_notes"], extras["cluster_notes"] = globals()[f"interpret_{name}"](extras)
            block.update(available=True, log=log.to_frame().drop(columns="time"), robustness=robustness_block(P),
                         extras=extras, evidence=P.finding_evidence(),
                         surprising_rules=P.surprising_rules(first(P.clean(defaults(P.DECISIONS)))))
            print(f"[ok]      {name}")
        except FileNotFoundError as e:
            block.update(available=False, reason=str(e))
            print(f"[missing] {name}: {e}")
        except Exception:
            block.update(available=False, reason="export failed: " + traceback.format_exc(limit=1).strip().splitlines()[-1])
            print(f"[error]   {name}"); traceback.print_exc()
        out["pulses"][name] = block
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("// Generated by scripts/export_dashboard_data.py. Do not edit by hand.\n"
                   "window.CAMPUSPULSE = " + json.dumps(clean_json(out), ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
