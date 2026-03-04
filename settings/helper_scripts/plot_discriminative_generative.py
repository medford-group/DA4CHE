#!/usr/bin/env python3
"""
Pedagogical discriminative vs. generative classification figure.

Two-panel figure contrasting the two fundamental philosophies for building
a classification model.

  Left  — Discriminative model.
           Learns a decision boundary directly from the data.  A test
           point (red star) is classified by asking: which side of the
           boundary is it on?  The dashed arrow shows that the model's
           confidence is related to the distance from the boundary.
           The model has no knowledge of the class distributions.

  Right — Generative model.
           Fits a class-conditional probability distribution P(x | class)
           for each class (shown as 1-σ and 2-σ confidence ellipses).
           The same test point is classified by asking: under which
           distribution is this point more likely?  The model can also
           detect out-of-distribution points (low P(x) under all classes).

Key pedagogical message
  Discriminative models (e.g., logistic regression, SVM) directly model
  the boundary P(y | x) and are often more accurate when the boundary is
  the only quantity of interest.  Generative models (e.g., Naive Bayes,
  LDA, GMM) additionally model P(x | y), which enables anomaly detection
  and data generation, but requires a distributional assumption.

Saves: discriminative_vs_generative.png  (same directory as this script)

Run from settings/helper_scripts/:
    python plot_discriminative_generative.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
STYLE_PATH = Path("../plot_style.mplstyle")
OUTFILE    = Path("discriminative_vs_generative.png")
FIGSIZE    = (11, 4.5)
SEED       = 7

MU_0 = np.array([1.3, 2.0])   # class 0 (navy)
MU_1 = np.array([3.5, 1.5])   # class 1 (gold)
COV_0 = np.array([[0.28, 0.10], [0.10, 0.22]])
COV_1 = np.array([[0.30, -0.08], [-0.08, 0.25]])
N     = 22

X_LIM = (-0.5, 5.0)
Y_LIM = (0.0, 3.5)

# Test point: sits outside both clusters, clearly in class-0 territory
TEST_POINT = np.array([0.0, 0.8])
# ──────────────────────────────────────────────────────────────────────────────


def confidence_ellipse(ax, mean, cov, n_std, facecolor, edgecolor, alpha_face, lw=1.5):
    """Draw a covariance confidence ellipse at n_std standard deviations."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  facecolor=facecolor, edgecolor=edgecolor,
                  alpha=alpha_face, lw=lw, zorder=1)
    ax.add_patch(ell)
    return ell


def nearest_boundary_point(clf, test_pt, x_lim, y_lim, res=800):
    """Find the point on the decision boundary closest to test_pt."""
    xx, yy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], res),
                         np.linspace(y_lim[0], y_lim[1], res))
    df = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Contour at 0 → find grid point with smallest |df| near the test point
    idx = np.argmin(np.abs(df))
    return np.array([xx.ravel()[idx], yy.ravel()[idx]])


def main():
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    else:
        print(f"[warn] Style not found at {STYLE_PATH.resolve()}. Using defaults.")

    palette = [e["color"] for e in plt.rcParams["axes.prop_cycle"]]
    def pal(i):
        h = palette[i]
        return h if h.startswith("#") else "#" + h

    c0 = pal(0)   # navy  — class 0
    c1 = pal(1)   # gold  — class 1
    c_test = "#C0392B"   # red — test point

    rng = np.random.default_rng(SEED)

    # ── Generate data ──────────────────────────────────────────────────────────
    X0 = rng.multivariate_normal(MU_0, COV_0, N)
    X1 = rng.multivariate_normal(MU_1, COV_1, N)
    X  = np.vstack([X0, X1])
    y  = np.array([0] * N + [1] * N)

    # ── Fit logistic regression (discriminative boundary) ─────────────────────
    clf = LogisticRegression(max_iter=1000).fit(X, y)

    # Meshgrid for decision regions and boundary
    res = 500
    xx, yy = np.meshgrid(np.linspace(X_LIM[0], X_LIM[1], res),
                         np.linspace(Y_LIM[0], Y_LIM[1], res))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Closest boundary point to test point (for the distance annotation)
    bnd_pt = nearest_boundary_point(clf, TEST_POINT, X_LIM, Y_LIM)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    # ═══ Left panel: Discriminative ═══════════════════════════════════════════
    ax = axes[0]

    # Faint decision regions
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9],
                colors=[c0, c1], alpha=0.10, zorder=0)

    # Decision boundary
    ax.contour(xx, yy, Z, levels=[0],
               colors=["#444444"], linewidths=2.2, zorder=3)

    # Scatter
    ax.scatter(X0[:, 0], X0[:, 1], c=c0, s=35, alpha=0.80,
               edgecolors="none", zorder=4)
    ax.scatter(X1[:, 0], X1[:, 1], c=c1, s=35, alpha=0.80,
               edgecolors="none", zorder=4)

    # Test point
    ax.scatter(*TEST_POINT, c=c_test, s=120, marker="*",
               edgecolors="none", zorder=6, label="Test point")

    # Dashed line: test point → nearest boundary point
    ax.plot([TEST_POINT[0], bnd_pt[0]], [TEST_POINT[1], bnd_pt[1]],
            color=c_test, lw=1.4, ls="--", zorder=5)

    # Mid-point annotation
    mid = 0.5 * (TEST_POINT + bnd_pt)
    ax.annotate("distance", xy=mid, xytext=(mid[0] + 0.55, mid[1] - 0.35),
                fontsize=8, color=c_test,
                arrowprops=dict(arrowstyle="-", color=c_test, lw=0.8))

    ax.annotate("decision\nboundary", fontsize=8, color="#444444",
                xy=(2.45, 2.85), ha="center")

    ax.set_title("Discriminative", fontsize=13, fontweight="bold")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    # ═══ Right panel: Generative ═══════════════════════════════════════════════
    ax = axes[1]

    # Confidence ellipses for each class (1-σ and 2-σ)
    for n_std, alpha in [(2, 0.15), (1, 0.28)]:
        confidence_ellipse(ax, MU_0, COV_0, n_std,
                           facecolor=c0, edgecolor=c0, alpha_face=alpha)
        confidence_ellipse(ax, MU_1, COV_1, n_std,
                           facecolor=c1, edgecolor=c1, alpha_face=alpha)

    # Ellipse outlines (solid edge, no fill) for clarity
    for n_std in [1, 2]:
        confidence_ellipse(ax, MU_0, COV_0, n_std,
                           facecolor="none", edgecolor=c0, alpha_face=0, lw=1.2)
        confidence_ellipse(ax, MU_1, COV_1, n_std,
                           facecolor="none", edgecolor=c1, alpha_face=0, lw=1.2)

    # Scatter
    ax.scatter(X0[:, 0], X0[:, 1], c=c0, s=35, alpha=0.80,
               edgecolors="none", zorder=4)
    ax.scatter(X1[:, 0], X1[:, 1], c=c1, s=35, alpha=0.80,
               edgecolors="none", zorder=4)

    # Test point (same location)
    ax.scatter(*TEST_POINT, c=c_test, s=120, marker="*",
               edgecolors="none", zorder=6)

    # Annotation: P(x|class 0) label near class 0 ellipse centre
    ax.annotate("$P(\\mathbf{x}\\,|\\,y=0)$", xy=MU_0, xytext=(MU_0[0] - 0.2, MU_0[1] - 0.85),
                fontsize=8.5, color=c0, ha="center",
                arrowprops=dict(arrowstyle="->", color=c0, lw=0.9))
    ax.annotate("$P(\\mathbf{x}\\,|\\,y=1)$", xy=MU_1, xytext=(MU_1[0] + 0.35, MU_1[1] + 0.75),
                fontsize=8.5, color=c1, ha="center",
                arrowprops=dict(arrowstyle="->", color=c1, lw=0.9))

    ax.set_title("Generative", fontsize=13, fontweight="bold")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    fig.tight_layout()
    fig.savefig(OUTFILE, bbox_inches="tight", dpi=300)
    print(f"[ok] Saved {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
