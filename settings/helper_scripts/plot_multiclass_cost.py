#!/usr/bin/env python3
"""
Pedagogical native multi-class cost function figure.

Three-panel figure illustrating how a joint multi-class objective
(multinomial logistic regression / softmax) learns all class boundaries
simultaneously — in contrast to the OvR decomposition.

  Panel 1 — Raw 3-class training data.

  Panel 2 — The three pairwise decision boundaries produced by
             multinomial logistic regression, all trained in a single
             optimisation.  Each boundary is drawn in the colour of the
             class it "belongs to" (the boundary that separates class k
             from both other classes).

  Panel 3 — Resulting decision regions.  A winner-takes-all rule assigns
             each point to the class with the highest softmax score.

Key pedagogical message
  Unlike OvR, which trains K independent classifiers, a native multi-class
  objective finds all K weight vectors jointly, so the boundaries are
  mutually consistent by construction.  The decision regions always meet
  at a single point (for a linear classifier).

Saves: multiclass_cost.png  (same directory as this script)

Run from settings/helper_scripts/:
    python plot_multiclass_cost.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
STYLE_PATH = Path("../plot_style.mplstyle")
OUTFILE    = Path("multiclass_cost.png")
FIGSIZE    = (12, 4.0)
SEED       = 42

# Same layout as OvA figure for direct comparability
CENTERS  = np.array([[1.0, 2.2], [3.2, 2.2], [2.1, 0.5]])
N_CLASS  = 30
STD      = 0.42

X_LIM = (-0.2, 4.4)
Y_LIM = (-0.5, 3.2)
# ──────────────────────────────────────────────────────────────────────────────


def pairwise_boundary_x1(clf, i, j, x0_range):
    """
    Return x1 values for the boundary between classes i and j under a
    multinomial logistic regression model.

    Boundary: (w_i - w_j)·x + (b_i - b_j) = 0
             → x1 = -(Δw0·x0 + Δb) / Δw1
    """
    dw = clf.coef_[i] - clf.coef_[j]
    db = clf.intercept_[i] - clf.intercept_[j]
    if abs(dw[1]) < 1e-10:
        return None, None           # horizontal boundary — skip
    x1 = -(dw[0] * x0_range + db) / dw[1]
    return x0_range, x1


def main():
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    else:
        print(f"[warn] Style not found at {STYLE_PATH.resolve()}. Using defaults.")

    palette = [e["color"] for e in plt.rcParams["axes.prop_cycle"]]
    def pal(i):
        h = palette[i]
        return h if h.startswith("#") else "#" + h

    colors = [pal(0), pal(1), pal(2)]   # navy, gold, teal
    labels = ["Class A", "Class B", "Class C"]

    rng = np.random.default_rng(SEED)

    # ── Generate data (identical to OvA figure) ────────────────────────────────
    Xs, ys = [], []
    for k, centre in enumerate(CENTERS):
        Xi = rng.multivariate_normal(centre, STD**2 * np.eye(2), N_CLASS)
        Xs.append(Xi)
        ys.append(np.full(N_CLASS, k))
    X = np.vstack(Xs)
    y = np.concatenate(ys)

    # ── Multinomial logistic regression (native multi-class) ──────────────────
    clf = LogisticRegression(max_iter=1000).fit(X, y)

    x0_range = np.linspace(X_LIM[0], X_LIM[1], 400)

    # Meshgrid for decision regions
    res = 400
    xx, yy = np.meshgrid(
        np.linspace(X_LIM[0], X_LIM[1], res),
        np.linspace(Y_LIM[0], Y_LIM[1], res),
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)

    # ── Panel 1: raw data ──────────────────────────────────────────────────────
    ax = axes[0]
    for k in range(3):
        ax.scatter(X[y == k, 0], X[y == k, 1],
                   c=colors[k], s=40, alpha=0.85, edgecolors="none", zorder=3)
    ax.set_title("Training data", fontsize=12)
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    # ── Panel 2: simultaneous boundaries ──────────────────────────────────────
    ax = axes[1]
    for k in range(3):
        ax.scatter(X[y == k, 0], X[y == k, 1],
                   c=colors[k], s=40, alpha=0.85, edgecolors="none", zorder=3)

    # Draw the three pairwise boundaries (A–B, A–C, B–C)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        # Colour the boundary by the "winning" class midpoint — use darker mix
        bnd_color = "#555555"
        x0_vals, x1_vals = pairwise_boundary_x1(clf, i, j, x0_range)
        if x0_vals is not None:
            # Clip to plot limits
            mask = (x1_vals >= Y_LIM[0]) & (x1_vals <= Y_LIM[1])
            if mask.any():
                ax.plot(x0_vals[mask], x1_vals[mask],
                        color=bnd_color, lw=2.0, zorder=4,
                        label=f"{labels[i]}–{labels[j]}")

    ax.set_title("Joint boundaries (softmax)", fontsize=12)
    ax.set_xlabel("$x_0$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    # ── Panel 3: decision regions ──────────────────────────────────────────────
    ax = axes[2]
    for k in range(3):
        ax.contourf(xx, yy, Z == k,
                    levels=[0.5, 1.5], colors=[colors[k]], alpha=0.20, zorder=0)

    # Boundaries on top
    for i, j in pairs:
        x0_vals, x1_vals = pairwise_boundary_x1(clf, i, j, x0_range)
        if x0_vals is not None:
            mask = (x1_vals >= Y_LIM[0]) & (x1_vals <= Y_LIM[1])
            if mask.any():
                ax.plot(x0_vals[mask], x1_vals[mask],
                        color="#444444", lw=1.8, zorder=4)

    for k in range(3):
        ax.scatter(X[y == k, 0], X[y == k, 1],
                   c=colors[k], s=40, alpha=0.85, edgecolors="none", zorder=5)

    legend_handles = [Patch(facecolor=colors[k], label=labels[k]) for k in range(3)]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.set_title("Decision regions", fontsize=12)
    ax.set_xlabel("$x_0$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    fig.tight_layout()
    fig.savefig(OUTFILE, bbox_inches="tight", dpi=300)
    print(f"[ok] Saved {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
