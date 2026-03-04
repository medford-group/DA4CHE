#!/usr/bin/env python3
"""
Pedagogical One-vs-Rest (OvR) multiclass classification figure.

Four-panel figure illustrating the OvR decomposition strategy:

  Panels 1-3 — Each panel shows one binary sub-problem.
               The target class is drawn in its full colour; all other
               classes are collapsed to grey ("rest").  A logistic-
               regression decision boundary is shown for that sub-problem.

  Panel 4    — The combined OvR result: each point in the feature space
               is assigned to whichever binary classifier returns the
               highest confidence score.  Decision regions are shaded in
               the three class colours.

Key pedagogical message
  OvR decomposes a K-class problem into K independent binary problems.
  The final class is the one whose binary classifier is most confident.
  This is simple and parallelisable, but each sub-classifier sees an
  unbalanced dataset (1 class vs. K−1 classes).

Saves: OvA.png  (same directory as this script)

Run from settings/helper_scripts/:
    python plot_ova.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
STYLE_PATH = Path("../plot_style.mplstyle")
OUTFILE    = Path("OvA.png")
FIGSIZE    = (14, 3.8)
SEED       = 42

# Three class centres in a triangular arrangement
CENTERS  = np.array([[1.0, 2.2], [3.2, 2.2], [2.1, 0.5]])
N_CLASS  = 30
STD      = 0.42

X_LIM = (-0.2, 4.4)
Y_LIM = (-0.5, 3.2)
# ──────────────────────────────────────────────────────────────────────────────

C_REST = "#AAAAAA"   # grey for the "rest" class in binary panels


def plot_decision_regions(ax, clf, x_lim, y_lim, colors, alpha=0.18, res=400):
    """Shade decision regions for a fitted classifier."""
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], res),
        np.linspace(y_lim[0], y_lim[1], res),
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    for label, color in enumerate(colors):
        ax.contourf(xx, yy, Z == label,
                    levels=[0.5, 1.5], colors=[color], alpha=alpha, zorder=0)


def plot_binary_boundary(ax, clf, x_lim, y_lim, color, res=400):
    """Draw the decision boundary of a binary logistic-regression classifier."""
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], res),
        np.linspace(y_lim[0], y_lim[1], res),
    )
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[0], colors=[color], linewidths=2.0, zorder=4)


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

    # ── Generate data ──────────────────────────────────────────────────────────
    Xs, ys = [], []
    for k, centre in enumerate(CENTERS):
        Xi = rng.multivariate_normal(centre, STD**2 * np.eye(2), N_CLASS)
        Xs.append(Xi)
        ys.append(np.full(N_CLASS, k))
    X = np.vstack(Xs)
    y = np.concatenate(ys)

    # ── Train K binary OvR classifiers ────────────────────────────────────────
    binary_clfs = []
    for k in range(3):
        y_k = (y == k).astype(int)
        clf = LogisticRegression(max_iter=1000).fit(X, y_k)
        binary_clfs.append(clf)

    # ── Train combined OvR classifier ─────────────────────────────────────────
    ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(X, y)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=FIGSIZE, sharey=True)

    # ── Panels 1–3: binary sub-problems ───────────────────────────────────────
    for k, (ax, clf) in enumerate(zip(axes[:3], binary_clfs)):
        # Shade decision regions: class k vs. rest
        xx, yy = np.meshgrid(
            np.linspace(X_LIM[0], X_LIM[1], 400),
            np.linspace(Y_LIM[0], Y_LIM[1], 400),
        )
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                    colors=[C_REST, colors[k]], alpha=0.18, zorder=0)

        # Decision boundary
        plot_binary_boundary(ax, clf, X_LIM, Y_LIM, color=colors[k])

        # Scatter: target class coloured, rest grey
        for j in range(3):
            c = colors[j] if j == k else C_REST
            zorder = 4 if j == k else 2
            s = 40 if j == k else 25
            alpha = 1.0 if j == k else 0.55
            ax.scatter(X[y == j, 0], X[y == j, 1],
                       c=c, s=s, alpha=alpha, edgecolors="none", zorder=zorder)

        ax.set_title(f"{labels[k]} vs. Rest", fontsize=11)
        ax.set_xlabel("$x_0$")
        if k == 0:
            ax.set_ylabel("$x_1$")
        ax.set_xlim(X_LIM)
        ax.set_ylim(Y_LIM)

    # ── Panel 4: combined OvR result ──────────────────────────────────────────
    ax = axes[3]
    plot_decision_regions(ax, ovr, X_LIM, Y_LIM, colors, alpha=0.22)

    # Class boundaries (one contour per binary estimator)
    for k, clf_k in enumerate(ovr.estimators_):
        xx, yy = np.meshgrid(
            np.linspace(X_LIM[0], X_LIM[1], 400),
            np.linspace(Y_LIM[0], Y_LIM[1], 400),
        )
        Z = clf_k.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0],
                   colors=[colors[k]], linewidths=1.5, linestyles="--", zorder=4)

    # Scatter all classes
    for j in range(3):
        ax.scatter(X[y == j, 0], X[y == j, 1],
                   c=colors[j], s=40, alpha=0.85, edgecolors="none", zorder=5)

    legend_handles = [Patch(facecolor=colors[j], label=labels[j]) for j in range(3)]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.set_title("Combined OvR result", fontsize=11)
    ax.set_xlabel("$x_0$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    fig.tight_layout()
    fig.savefig(OUTFILE, bbox_inches="tight", dpi=300)
    print(f"[ok] Saved {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
