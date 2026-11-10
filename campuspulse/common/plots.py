"""Shared colours so all five pulses look like one project."""
import matplotlib.pyplot as plt

PULSE_COLOURS = {"academic": "#2b6cb0", "engagement": "#2f855a", "wellbeing": "#b83280",
                 "food": "#dd6b20", "energy": "#805ad5"}

def theme():
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25})

def badge(ax, dataset: str, real: bool = True):
    """Every chart must carry dataset + real/synthetic badge (Definition of Done)."""
    ax.annotate(f"{dataset} | {'REAL' if real else 'SYNTHETIC'}", xy=(1, 1.02), xycoords="axes fraction",
                ha="right", fontsize=7, color="#555")
