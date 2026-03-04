#!/usr/bin/env python3
"""
Pedagogical precision and recall figure.

Single-panel illustration showing how the four outcomes of a binary
classifier (TP, FP, FN, TN) relate to precision and recall.

Layout
------
  Background is split into two vertical bands:
    • Left  — Actual Positive  (all points that truly belong to class 1)
    • Right — Actual Negative  (all points that truly belong to class 0)

  A dashed ellipse spans both bands and represents the set of points the
  classifier predicted as Positive.

  Dots are colour-coded by outcome:
    • Navy  — True  Positive (TP): correctly predicted positive
    • Gold  — False Negative (FN): missed positive (inside actual+, outside ellipse)
    • Red   — False Positive (FP): false alarm (inside ellipse, actual−)
    • Grey  — True  Negative (TN): correctly predicted negative

  Formula callouts at the bottom remind the reader how each metric is
  derived from the four quadrants.

Saves: precision_recall.png  (same directory as this script)

Run from settings/helper_scripts/:
    python plot_precision_recall.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyBboxPatch
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
STYLE_PATH = Path("../plot_style.mplstyle")
OUTFILE    = Path("precision_recall.png")
FIGSIZE    = (10, 6.5)
SEED       = 3

# Diagram coordinate space
X_MIN, X_MAX = 0.0, 10.0
Y_MIN, Y_MAX = 1.2, 7.0   # top portion; bottom reserved for formulas
MID_X        = 5.0        # boundary between actual+/actual−

# Predicted-positive ellipse
ELL_CX, ELL_CY = 4.2, 4.1
ELL_A, ELL_B   = 3.0, 2.2   # semi-axes (x, y)

# Counts in each region
N_TP = 14
N_FN = 7
N_FP = 5
N_TN = 18
# ──────────────────────────────────────────────────────────────────────────────

C_TP   = "#003057"   # navy
C_FN   = "#EAAA00"   # gold
C_FP   = "#C0392B"   # red
C_TN   = "#AAAAAA"   # grey
C_ACT_POS = "#D6E4F0"  # light blue band
C_ACT_NEG = "#F0F0F0"  # light grey band


def in_ellipse(x, y, cx, cy, a, b):
    return ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 < 1.0


def sample_region(rng, n, x_lo, x_hi, y_lo, y_hi, inside_ell, cx, cy, a, b,
                  max_tries=50_000):
    """Sample n points uniformly in a rectangular region, filtered by ellipse."""
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        x = rng.uniform(x_lo, x_hi)
        y = rng.uniform(y_lo, y_hi)
        if in_ellipse(x, y, cx, cy, a, b) == inside_ell:
            pts.append((x, y))
        tries += 1
    return np.array(pts)


def main():
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    else:
        print(f"[warn] Style not found at {STYLE_PATH.resolve()}. Using defaults.")

    rng = np.random.default_rng(SEED)

    # ── Sample dot positions ───────────────────────────────────────────────────
    margin = 0.35   # keep dots away from region edges
    TP_pts = sample_region(rng, N_TP,
                           X_MIN + margin, MID_X - margin,
                           Y_MIN + margin, Y_MAX - margin,
                           inside_ell=True,
                           cx=ELL_CX, cy=ELL_CY, a=ELL_A, b=ELL_B)
    FN_pts = sample_region(rng, N_FN,
                           X_MIN + margin, MID_X - margin,
                           Y_MIN + margin, Y_MAX - margin,
                           inside_ell=False,
                           cx=ELL_CX, cy=ELL_CY, a=ELL_A, b=ELL_B)
    FP_pts = sample_region(rng, N_FP,
                           MID_X + margin, X_MAX - margin,
                           Y_MIN + margin, Y_MAX - margin,
                           inside_ell=True,
                           cx=ELL_CX, cy=ELL_CY, a=ELL_A, b=ELL_B)
    TN_pts = sample_region(rng, N_TN,
                           MID_X + margin, X_MAX - margin,
                           Y_MIN + margin, Y_MAX - margin,
                           inside_ell=False,
                           cx=ELL_CX, cy=ELL_CY, a=ELL_A, b=ELL_B)

    for name, pts, n in [("TP", TP_pts, N_TP), ("FN", FN_pts, N_FN),
                          ("FP", FP_pts, N_FP), ("TN", TN_pts, N_TN)]:
        if len(pts) < n:
            print(f"[warn] Only {len(pts)}/{n} {name} points placed.")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0.0, Y_MAX + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Background bands ───────────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (X_MIN, Y_MIN), MID_X - X_MIN, Y_MAX - Y_MIN,
        boxstyle="square,pad=0", facecolor=C_ACT_POS, edgecolor="none", zorder=0))
    ax.add_patch(mpatches.FancyBboxPatch(
        (MID_X, Y_MIN), X_MAX - MID_X, Y_MAX - Y_MIN,
        boxstyle="square,pad=0", facecolor=C_ACT_NEG, edgecolor="none", zorder=0))

    # Band labels at top
    ax.text(MID_X / 2, Y_MAX + 0.05, "Actual Positive",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333333")
    ax.text((MID_X + X_MAX) / 2, Y_MAX + 0.05, "Actual Negative",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#555555")

    # Dividing line
    ax.axvline(MID_X, ymin=(Y_MIN) / (Y_MAX + 0.2),
               ymax=Y_MAX / (Y_MAX + 0.2),
               color="#999999", lw=1.2, ls="--", zorder=1)

    # ── Predicted-positive ellipse ─────────────────────────────────────────────
    ell = Ellipse(xy=(ELL_CX, ELL_CY), width=2 * ELL_A, height=2 * ELL_B,
                  facecolor="none", edgecolor="#333333",
                  lw=2.2, linestyle="--", zorder=3)
    ax.add_patch(ell)

    # Ellipse label — place on right side outside the ellipse
    ax.text(ELL_CX + ELL_A + 0.15, ELL_CY - 0.6,
            "Predicted\nPositive", ha="left", va="center",
            fontsize=9, color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    # ── Dots ───────────────────────────────────────────────────────────────────
    dot_kw = dict(s=55, edgecolors="none", zorder=5)
    ax.scatter(TP_pts[:, 0], TP_pts[:, 1], c=C_TP, **dot_kw)
    ax.scatter(FN_pts[:, 0], FN_pts[:, 1], c=C_FN, **dot_kw)
    ax.scatter(FP_pts[:, 0], FP_pts[:, 1], c=C_FP, **dot_kw)
    ax.scatter(TN_pts[:, 0], TN_pts[:, 1], c=C_TN, **dot_kw)

    # ── Region labels ──────────────────────────────────────────────────────────
    label_kw = dict(ha="center", va="center", fontsize=12, fontweight="bold", zorder=6)
    ax.text(2.8, 5.3, f"TP\n({N_TP})", color=C_TP,   **label_kw)
    ax.text(0.8, 2.2, f"FN\n({N_FN})", color=C_FN,   **label_kw)
    ax.text(6.5, 5.1, f"FP\n({N_FP})", color=C_FP,   **label_kw)
    ax.text(8.2, 2.6, f"TN\n({N_TN})", color=C_TN,   **label_kw)

    # ── Formula callouts at the bottom ────────────────────────────────────────
    y_form = 0.72
    # Precision box
    ax.text(2.5, y_form,
            "Precision  =",
            ha="right", va="center", fontsize=11, color="#333333")
    ax.text(2.6, y_form + 0.22,
            f"  TP  ({N_TP})",
            ha="left", va="center", fontsize=10, color=C_TP, fontweight="bold")
    ax.plot([2.6, 5.0], [y_form, y_form], color="#333333", lw=1.2)
    ax.text(2.6, y_form - 0.22,
            f"TP + FP  ({N_TP + N_FP})",
            ha="left", va="center", fontsize=10, color="#333333")
    prec = N_TP / (N_TP + N_FP)
    ax.text(5.15, y_form,
            f"= {prec:.2f}",
            ha="left", va="center", fontsize=11, color="#333333")

    # Recall box
    ax.text(2.5, y_form - 1.1,
            "Recall  =",
            ha="right", va="center", fontsize=11, color="#333333")
    ax.text(2.6, y_form - 1.1 + 0.22,
            f"  TP  ({N_TP})",
            ha="left", va="center", fontsize=10, color=C_TP, fontweight="bold")
    ax.plot([2.6, 5.0], [y_form - 1.1, y_form - 1.1], color="#333333", lw=1.2)
    ax.text(2.6, y_form - 1.1 - 0.22,
            f"TP + FN  ({N_TP + N_FN})",
            ha="left", va="center", fontsize=10, color="#333333")
    rec = N_TP / (N_TP + N_FN)
    ax.text(5.15, y_form - 1.1,
            f"= {rec:.2f}",
            ha="left", va="center", fontsize=11, color="#333333")

    # Legend
    handles = [
        mpatches.Patch(color=C_TP, label=f"True Positive  (TP = {N_TP})"),
        mpatches.Patch(color=C_FN, label=f"False Negative (FN = {N_FN})"),
        mpatches.Patch(color=C_FP, label=f"False Positive (FP = {N_FP})"),
        mpatches.Patch(color=C_TN, label=f"True Negative  (TN = {N_TN})"),
    ]
    ax.legend(handles=handles, loc="lower right",
              fontsize=8.5, framealpha=0.9,
              bbox_to_anchor=(1.0, 0.0))

    fig.tight_layout()
    fig.savefig(OUTFILE, bbox_inches="tight", dpi=300)
    print(f"[ok] Saved {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
