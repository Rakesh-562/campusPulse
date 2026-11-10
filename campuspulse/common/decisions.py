"""Shared contract: DecisionLog + Finding Robustness Matrix.

Each pulse module exposes:
    DECISIONS    : {decision_name: [option, ...]}   first option = default
    FINDINGS     : {finding_id: "plain-language claim"}
    run_pipeline : (decisions: dict) -> {finding_id: float}
A finding value is a signed effect where POSITIVE means the claim holds.

Each (finding, variant) cell gets one status, compared with the default:
    holds     same sign, size between min_ratio and max_ratio of the default
    weakens   same sign, but shrinks below min_ratio of the default
    inflates  same sign, but grows beyond max_ratio of the default
    reverses  sign flips (the claim no longer holds)
    n/a       the finding cannot be computed under that variant
Only "holds" counts as surviving. An effect that doubles is not "robust": it
means a cleaning choice is manufacturing (or hiding) most of the effect.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

STATUSES = ["holds", "weakens", "inflates", "reverses", "n/a"]
STATUS_COLOURS = {"holds": "#38a169", "weakens": "#ecc94b", "inflates": "#dd6b20",
                  "reverses": "#e53e3e", "n/a": "#cbd5e0"}


@dataclass
class DecisionLog:
    """Record every cleaning step with row counts before/after and a reason."""
    entries: list = field(default_factory=list)

    def step(self, name: str, before: int, after: int, why: str, decision: str | None = None):
        self.entries.append(dict(time=datetime.now().isoformat(timespec="seconds"), step=name,
                                 decision=decision, rows_before=before, rows_after=after,
                                 rows_changed=before - after, why=why))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)


def defaults(decisions: dict) -> dict:
    return {k: v[0] for k, v in decisions.items()}


def classify(value, base, min_ratio: float = 0.5, max_ratio: float = 2.0) -> str:
    if pd.isna(value) or pd.isna(base) or base == 0:
        return "n/a"
    if value * base <= 0:
        return "reverses"
    ratio = abs(value) / abs(base)
    if ratio < min_ratio:
        return "weakens"
    if ratio > max_ratio:
        return "inflates"
    return "holds"


def status_matrix(values: pd.DataFrame, min_ratio: float = 0.5, max_ratio: float = 2.0) -> pd.DataFrame:
    """Status of every (finding, variant) cell relative to the DEFAULT column."""
    out = pd.DataFrame("n/a", index=values.index, columns=values.columns)
    for fid in values.index:
        b = values.loc[fid, "DEFAULT"]
        for c in values.columns:
            out.loc[fid, c] = classify(values.loc[fid, c], b, min_ratio, max_ratio)
    return out


def robustness_matrix(run_pipeline, decisions: dict, findings: dict,
                      min_ratio: float = 0.5, max_ratio: float = 2.0):
    """One-at-a-time sensitivity: change one decision, keep the others at default.
    Returns (values, survives): findings as rows, variants as columns.
    survives[f, v] is True only when the status is "holds" (see status_matrix)."""
    base_cfg = defaults(decisions)
    cols = {"DEFAULT": run_pipeline(base_cfg)}
    for dec, options in decisions.items():
        for opt in options[1:]:
            try:
                cols[f"{dec}={opt}"] = run_pipeline({**base_cfg, dec: opt})
            except Exception as e:  # a variant that breaks is itself worth reporting
                cols[f"{dec}={opt}"] = {k: float("nan") for k in findings}
                print(f"variant {dec}={opt} failed: {e}")
    values = pd.DataFrame(cols).reindex(list(findings))
    status = status_matrix(values, min_ratio, max_ratio)
    return values, status.eq("holds")


def survival_summary(values: pd.DataFrame, min_ratio: float = 0.5, max_ratio: float = 2.0) -> dict:
    """Counts of each status over the non-default variants (n/a excluded from the rate)."""
    st = status_matrix(values, min_ratio, max_ratio).drop(columns="DEFAULT")
    counts = {s: int((st == s).values.sum()) for s in STATUSES}
    tested = sum(v for k, v in counts.items() if k != "n/a")
    counts["tested"] = tested
    counts["survival_rate"] = counts["holds"] / tested if tested else float("nan")
    return counts


def fragility_table(values: pd.DataFrame, findings: dict | None = None,
                    min_ratio: float = 0.5, max_ratio: float = 2.0) -> pd.DataFrame:
    """One row per finding, ranked from most to least fragile.
    swing = largest |log(variant / default)| over variants with the same sign
    (inf if any variant reverses the sign). worst_variant = the decision causing it."""
    st = status_matrix(values, min_ratio, max_ratio)
    rows = []
    for fid in values.index:
        b = values.loc[fid, "DEFAULT"]
        worst, swing = None, 0.0
        for c in values.columns.drop("DEFAULT"):
            v = values.loc[fid, c]
            if pd.isna(v) or pd.isna(b) or b == 0:
                continue
            s = np.inf if v * b <= 0 else abs(np.log(abs(v) / abs(b)))
            if s > swing:
                worst, swing = c, s
        statuses = st.loc[fid].drop("DEFAULT")
        rows.append(dict(finding=fid, claim=(findings or {}).get(fid, ""), default=b,
                         worst_variant=worst,
                         worst_value=values.loc[fid, worst] if worst else np.nan,
                         change=("sign flips" if swing == np.inf else f"x{np.exp(swing):.2f}") if worst else "none",
                         swing=swing,
                         verdict="fragile" if (statuses.isin(["reverses", "inflates", "weakens"])).any() else "robust"))
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def heatmap(values: pd.DataFrame, survives: pd.DataFrame | None = None, title: str = "",
            min_ratio: float = 0.5, max_ratio: float = 2.0):
    """Colour = status (green holds, yellow weakens, orange inflates, red reverses, grey n/a).
    `survives` is accepted for backward compatibility; colours are recomputed from values."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    st = status_matrix(values, min_ratio, max_ratio)
    codes = st.replace({s: i for i, s in enumerate(STATUSES)}).astype(int).values
    cmap = ListedColormap([STATUS_COLOURS[s] for s in STATUSES])
    fig, ax = plt.subplots(figsize=(1.6 * len(values.columns) + 2, 0.7 * len(values) + 2))
    ax.imshow(codes, cmap=cmap, vmin=0, vmax=len(STATUSES) - 1, aspect="auto")
    ax.set_xticks(range(len(values.columns)), values.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(values.index)), values.index)
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values.iat[i, j]
            ax.text(j, i, "n/a" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.legend(handles=[Patch(color=STATUS_COLOURS[s], label=s) for s in STATUSES],
              loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize=7, frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig
