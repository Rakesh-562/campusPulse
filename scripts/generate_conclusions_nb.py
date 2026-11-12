"""Generator script for notebooks/conclusions.ipynb.

RULE: no result is typed by hand. Every number, table and chart in the conclusions notebook is computed from the
pulse pipelines when the notebook is built, so it can never disagree with the code or the dashboard.

Usage:  python scripts/generate_conclusions_nb.py [output_path]
"""
import os, sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".mplcache")

from scripts.notebook_builder import NotebookRunner


def build_conclusions(out_path: Path | None = None):
    print("Building conclusions.ipynb...")
    out_path = Path(out_path) if out_path else BASE_DIR / "notebooks" / "conclusions.ipynb"
    nb = NotebookRunner(out_path)

    nb.md(r"""# CampusPulse: Conclusions
**Umbrella question:** *Can routine campus data warn a university early enough to act, and how much do those
warnings depend on our data-cleaning decisions?*

| Pulse | Question | Dataset |
|---|---|---|
| ① Academic | Who drops out, and how early can we tell? | UCI 697, Portuguese polytechnic, 4,424 students |
| ② Wellbeing | Which behaviours co-occur with persistent sadness? | CDC YRBS 2023 (20,103 students) + Malaysia survey (n=101) |
| ③ Energy | When does a campus building behave differently? | I-BLEND, IIIT Delhi, 52 months of meter data + teaching calendar |

> Every number below is computed by the code cells in this notebook, not typed in.""")

    nb.code(r"""import sys, os
from pathlib import Path
for cand in [Path.cwd(), Path.cwd().parent]:
    if (cand / 'campuspulse').exists():
        sys.path.insert(0, str(cand)); break
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from campuspulse.academic import pipeline as A
from campuspulse.wellbeing import pipeline as W
from campuspulse.energy import pipeline as EN
from campuspulse.common.decisions import robustness_matrix, survival_summary, fragility_table, status_matrix, defaults
from campuspulse.common.plots import theme
theme()
pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 90)
PULSES = {"① Academic": A, "② Wellbeing": W, "③ Energy": EN}""")

    nb.md("## 1. Every finding with its simple hypothesis test")
    nb.code(r"""rows = []
for name, P in PULSES.items():
    vals, _ = robustness_matrix(P.run_pipeline, P.DECISIONS, P.FINDINGS)
    stat = status_matrix(vals).drop(columns="DEFAULT")
    for fid, ev in P.finding_evidence().items():
        t = ev["test"]
        rows.append({"pulse": name, "finding": fid, "claim": P.FINDINGS[fid], "value": round(vals.loc[fid, "DEFAULT"], 3),
                     "test": t["test"], f"statistic": f"{t['statistic_name']} = {t['statistic']:.2f}", "p": f"{t['p']:.2g}",
                     "decision": t["decision"], "holds": f"{(stat.loc[fid] == 'holds').sum()}/{(stat.loc[fid] != 'n/a').sum()}"})
print(pd.DataFrame(rows).to_string(index=False))""")

    nb.md(r"""## 2. How fragile are the insights?
A finding **holds** if it keeps its sign and stays between ½× and 2× of our default under an alternative choice.""")
    nb.code(r"""summary, board = [], []
for name, P in PULSES.items():
    vals, _ = robustness_matrix(P.run_pipeline, P.DECISIONS, P.FINDINGS)
    summary.append({"pulse": name, **survival_summary(vals)})
    board.append(fragility_table(vals, P.FINDINGS).assign(pulse=name))
print(pd.DataFrame(summary)[["pulse", "tested", "holds", "weakens", "inflates", "reverses", "survival_rate"]].to_string(index=False))
board = pd.concat(board).sort_values("swing", ascending=False)
print(board[["pulse", "finding", "worst_variant", "default", "worst_value", "change", "verdict"]].round(3).to_string(index=False))""")

    nb.md(r"""## 3. How early can a university warn? (① Academic)
If the university contacts the 25% highest-risk students, what share of all eventual dropouts does it reach?""")
    nb.code(r"""tl = A.run_timeline(A.clean(defaults(A.DECISIONS)))
print(tl[["stage", "roc_auc", "capture_at_25pct", "max_possible_capture"]].round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(tl["stage"], tl["capture_at_25pct"]); ax.axvline(tl["max_possible_capture"].iloc[0], color="grey")
ax.set_xlim(0, 1); ax.set_xlabel("Share of dropouts reached"); ax.invert_yaxis()
plt.show()""")

    nb.md(r"""## 4. Is the energy term effect just the season? (③ Energy)
The I-BLEND calendar marks teaching days and vacations but not exams. Delhi's long vacation is also its hottest
season, so EN3 compares term and vacation nights only within the same calendar month.""")
    nb.code(r"""st_en = EN.run_stats(EN.clean(defaults(EN.DECISIONS)))
print(st_en["within_month_t"])
print(st_en["by_month"].round(0).to_string(index=False))""")

    nb.md(r"""## 5. Limitations, implications and next steps

**Limitations**
- Three institutions in three countries, none of them ours. We compare *conclusions*, never join rows across datasets.
- Every finding is an association, not a cause.
- YRBS is a high-school survey; the Malaysia survey has n = 101, so W4 is exploratory.
- Hostel power is not a measurement of sleep, and the I-BLEND calendar has no exam dates.
- The robustness matrix changes one decision at a time; combinations are not tested.

**Implications (where the results above support them)**
- A semester-1 checkpoint reaches most future dropouts with a short contact list; fee status and scholarships are cheap signals.
- A cleaning choice can create or hide most of an effect, so dashboards should publish their cleaning decisions.

**Next steps**
- Indian institutional data (AISHE, or our own university with ethics approval) to test which signals transfer.
- Test combinations of decisions, not just one at a time.""")

    nb.save()
    print(f"Saved conclusions notebook to {out_path}")


if __name__ == "__main__":
    build_conclusions(sys.argv[1] if len(sys.argv) > 1 else None)
