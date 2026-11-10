"""Shared contract: DecisionLog + Finding Robustness Matrix.

Each pulse module exposes:
    DECISIONS    : {decision_name: [option, ...]}   first option = default
    FINDINGS     : {finding_id: "plain-language claim"}
    run_pipeline : (decisions: dict) -> {finding_id: float}
A finding value is a signed effect where POSITIVE means the claim holds.
A finding SURVIVES a variant if its sign equals the default's and its size is >= min_ratio of the default's.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


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


def robustness_matrix(run_pipeline, decisions: dict, findings: dict, min_ratio: float = 0.5):
    """One-at-a-time sensitivity: change one decision, keep the others at default.
    Returns (values, survives): DataFrames, findings as rows, variants as columns."""
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
    surv = pd.DataFrame(False, index=values.index, columns=values.columns)
    for fid in values.index:
        b = values.loc[fid, "DEFAULT"]
        for c in values.columns:
            v = values.loc[fid, c]
            surv.loc[fid, c] = bool(pd.notna(v) and pd.notna(b) and v * b > 0 and abs(v) >= min_ratio * abs(b))
    return values, surv


def heatmap(values: pd.DataFrame, survives: pd.DataFrame, title: str = ""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.6 * len(values.columns) + 2, 0.7 * len(values) + 1.5))
    ax.imshow(survives.astype(int).values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(values.columns)), values.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(values.index)), values.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values.iat[i, j]
            ax.text(j, i, "n/a" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig
