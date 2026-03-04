#!/usr/bin/env python3
"""
Pedagogical ROC curve figure showing three model quality levels:
  - Worthless model (AUC ≈ 0.50) — classifier is no better than random guessing
  - Good model      (AUC ≈ 0.85) — moderate discrimination ability
  - Excellent model (AUC ≈ 0.96) — strong discrimination ability

Curves are generated analytically using the binormal ROC model:
  class-0 scores ~ N(0, 1),  class-1 scores ~ N(d, 1)
  FPR(t) = 1 − Φ(t),   TPR(t) = 1 − Φ(t − d)
  AUC    = Φ(d / √2)

Saves: ROC_curve.png  (in the same directory as this script)

Run from settings/helper_scripts/:
    python plot_roc_curves.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import norm
from scipy.integrate import trapezoid

# ---------- Configuration ----------
STYLE_PATH = Path("../plot_style.mplstyle")
OUTFILE    = Path("ROC_curve.png")
FIGSIZE    = (6, 5)

# d = class-mean separation in the binormal model
# AUC = Φ(d / √2)
MODELS = [
    {"label": "Worthless model", "d": 0.0,  "linestyle": "--",  "alpha": 0.85},
    {"label": "Good model",      "d": 1.5,  "linestyle": "-",   "alpha": 0.85},
    {"label": "Excellent model", "d": 2.7,  "linestyle": "-",   "alpha": 0.85},
]
# -----------------------------------


def binormal_roc(d: float, n: int = 500):
    """Return (fpr, tpr) arrays for the binormal ROC with separation d.

    Threshold decreases from high → low so FPR increases 0 → 1 monotonically,
    which is required for correct trapezoid AUC integration.
    """
    t = np.linspace(5, -5, n)          # decreasing threshold → FPR increases
    fpr = 1 - norm.cdf(t)              # class-0: N(0,1)
    tpr = 1 - norm.cdf(t - d)          # class-1: N(d,1)
    return fpr, tpr


def main():
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    else:
        print(f"[warn] Style not found at {STYLE_PATH.resolve()}. Using defaults.")

    # Pull first three colours from the cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    palette    = [entry["color"] for entry in prop_cycle]
    # Assign: worthless → grey, good → second colour, excellent → first colour
    colours = ["#9E9E9E", "#" + palette[2].lstrip("#"), "#" + palette[0].lstrip("#")]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for model, colour in zip(MODELS, colours):
        fpr, tpr = binormal_roc(model["d"])
        auc = trapezoid(tpr, fpr)
        ax.plot(
            fpr, tpr,
            linestyle=model["linestyle"],
            color=colour,
            alpha=model["alpha"],
            label=f"{model['label']}  (AUC = {auc:.2f})",
        )

    # Shade region between excellent model and diagonal for visual impact
    fpr_ex, tpr_ex = binormal_roc(MODELS[2]["d"])
    ax.fill_between(fpr_ex, fpr_ex, tpr_ex, alpha=0.08, color=colours[2])

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", linewidth=1.2, zorder=0)

    # Annotate corners
    ax.annotate("Perfect\nclassifier", xy=(0, 1), xytext=(0.12, 0.92),
                fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)

    fig.savefig(OUTFILE, bbox_inches="tight", dpi=300)
    print(f"[ok] Saved {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
