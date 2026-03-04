#!/usr/bin/env python3
"""
Pedagogical undersampling figure.

Two-panel figure showing how class imbalance biases the decision boundary
and how undersampling corrects it.

  Left  — Imbalanced dataset (150 majority : 10 minority, 15:1 ratio).
           LDA with empirical priors places the boundary near x ≈ 0.05,
           almost ON the minority cluster centre — ~47% of minority points
           are misclassified.  The dashed "ideal" boundary (trained on
           balanced data) sits at x ≈ 1.0, midway between the classes.
  Right — After random undersampling (10 kept : 140 discarded).  The now-
           balanced LDA boundary is near the ideal.  Discarded majority
           points are shown as faded grey markers.

Key pedagogical message
  Undersampling restores the boundary to a near-ideal position, but it
  throws away 140 real data points (the grey markers in the right panel).

Saves: undersampling.png  (same directory as this script)

Run from settings/helper_scripts/:
    python plot_undersampling.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
STYLE_PATH = Path("../plot_style.mplstyle")
OUTFILE    = Path("undersampling.png")
FIGSIZE    = (11, 4.8)
SEED       = 7

N_MAJORITY = 150
N_MINORITY = 10

# Class centres separated along x₀ only → boundary is nearly vertical
MU_0 = np.array([2.0, 1.0])   # majority (class 0)
MU_1 = np.array([0.0, 1.0])   # minority (class 1)
COV  = np.diag([0.7, 0.5])

X_LIM = (-1.5, 3.5)
Y_LIM = (-0.8, 2.8)

# Theoretical Bayes boundary under equal-covariance Gaussian assumption:
#   x* = midpoint + σ² · ln(π₁/π₀) / (μ₀ − μ₁)
MIDPOINT    = 0.5 * (MU_0[0] + MU_1[0])   # 1.0
SIGMA_X2    = COV[0, 0]                    # 0.7
X_IDEAL     = MIDPOINT                     # 1.00
X_BIASED_TH = MIDPOINT + SIGMA_X2 * np.log(N_MINORITY / N_MAJORITY) / (MU_0[0] - MU_1[0])
# ──────────────────────────────────────────────────────────────────────────────


def boundary_line(clf, y_range):
    """Return x₀ values of the linear decision boundary across a y range."""
    w, b = clf.coef_[0], clf.intercept_[0]
    # w[0]*x0 + w[1]*x1 + b = 0  →  x0 = -(w[1]*x1 + b) / w[0]
    return -(w[1] * y_range + b) / w[0]


def fill_halfplane(ax, x_boundary_fn, y_range, x_lim, color_left, color_right, alpha=0.12):
    """Fill left/right half-planes defined by a tilted boundary line."""
    # Build polygon for left region
    yy = np.linspace(*y_range, 200)
    xx_bnd = x_boundary_fn(yy)
    xx_bnd = np.clip(xx_bnd, x_lim[0], x_lim[1])

    left_poly_x  = np.concatenate([[x_lim[0]], xx_bnd,        [x_lim[0]]])
    left_poly_y  = np.concatenate([[y_range[0]], yy,           [y_range[1]]])
    right_poly_x = np.concatenate([[x_lim[1]], xx_bnd,        [x_lim[1]]])
    right_poly_y = np.concatenate([[y_range[0]], yy,           [y_range[1]]])

    ax.fill(left_poly_x,  left_poly_y,  color=color_left,  alpha=alpha, zorder=0)
    ax.fill(right_poly_x, right_poly_y, color=color_right, alpha=alpha, zorder=0)


def main():
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    else:
        print(f"[warn] Style not found at {STYLE_PATH.resolve()}. Using defaults.")

    palette = [e["color"] for e in plt.rcParams["axes.prop_cycle"]]
    def pal(i):
        h = palette[i]
        return h if h.startswith("#") else "#" + h

    c_maj  = pal(0)   # navy  — majority / class 0
    c_min  = pal(1)   # gold  — minority / class 1
    c_bnd  = "#C0392B"  # red    — biased boundary (stands out)
    c_bnd2 = "#333333"  # dark grey — ideal/corrected boundary

    rng = np.random.default_rng(SEED)

    # ── Generate data ──────────────────────────────────────────────────────────
    X0 = rng.multivariate_normal(MU_0, COV, N_MAJORITY)
    X1 = rng.multivariate_normal(MU_1, COV, N_MINORITY)

    X_imbal = np.vstack([X0, X1])
    y_imbal = np.array([0] * N_MAJORITY + [1] * N_MINORITY)

    # ── Undersampling ──────────────────────────────────────────────────────────
    keep_idx  = rng.choice(N_MAJORITY, N_MINORITY, replace=False)
    drop_mask = np.ones(N_MAJORITY, dtype=bool)
    drop_mask[keep_idx] = False

    X_kept    = X0[keep_idx]
    X_dropped = X0[drop_mask]

    X_under = np.vstack([X_kept, X1])
    y_under = np.array([0] * N_MINORITY + [1] * N_MINORITY)

    # ── Train LDA classifiers ──────────────────────────────────────────────────
    # LDA with empirical priors → boundary shifts toward minority cluster
    lda_imbal = LinearDiscriminantAnalysis().fit(X_imbal, y_imbal)
    # LDA on undersampled (balanced) data → boundary returns to ideal
    lda_under = LinearDiscriminantAnalysis().fit(X_under, y_under)

    # Compute where each boundary crosses y=Y_MID for annotation
    y_range = np.array(Y_LIM)
    y_mid   = 1.0  # class centre height

    def x_at_mid(clf):
        w, b = clf.coef_[0], clf.intercept_[0]
        return -(w[1] * y_mid + b) / w[0]

    x_imbal_mid = x_at_mid(lda_imbal)
    x_under_mid = x_at_mid(lda_under)

    print(f"Theoretical biased boundary: x₀ ≈ {X_BIASED_TH:.3f}")
    print(f"LDA imbalanced boundary:     x₀ ≈ {x_imbal_mid:.3f}  (at y=1)")
    print(f"LDA undersampled boundary:   x₀ ≈ {x_under_mid:.3f}  (at y=1)")
    print(f"Theoretical ideal:           x₀ ≈ {X_IDEAL:.3f}")

    yy_plot = np.linspace(Y_LIM[0], Y_LIM[1], 300)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

    # ── Left panel ─────────────────────────────────────────────────────────────
    ax = axes[0]

    # Decision region fill for imbalanced LDA boundary
    fill_halfplane(ax,
                   lambda yy: boundary_line(lda_imbal, yy),
                   Y_LIM, X_LIM,
                   color_left=c_min, color_right=c_maj, alpha=0.13)

    # Scatter
    sc0 = ax.scatter(X0[:, 0], X0[:, 1],
                     c=c_maj, s=25, alpha=0.65, edgecolors="none", zorder=3,
                     label=f"Majority  (n = {N_MAJORITY})")
    sc1 = ax.scatter(X1[:, 0], X1[:, 1],
                     c=c_min, s=80, alpha=1.0, marker="*", edgecolors="none", zorder=4,
                     label=f"Minority  (n = {N_MINORITY})")

    # Biased boundary (red solid)
    xx_bnd_imbal = boundary_line(lda_imbal, yy_plot)
    (line_bias,) = ax.plot(xx_bnd_imbal, yy_plot, color=c_bnd, lw=2.2, zorder=5,
                           label=f"Biased boundary  (x₀ ≈ {x_imbal_mid:.2f})")

    # Theoretical ideal boundary: vertical line at the class midpoint
    line_ideal = ax.axvline(x=X_IDEAL, color=c_bnd2, lw=1.8, ls="--", zorder=5,
                            label=f"Ideal boundary  (x₀ = {X_IDEAL:.2f})")

    ax.legend(handles=[sc0, sc1, line_bias, line_ideal], fontsize=8, loc="upper right")
    ax.set_title("Imbalanced dataset", fontsize=13)
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    # ── Right panel ─────────────────────────────────────────────────────────────
    ax = axes[1]

    # Decision region fill for undersampled LDA boundary
    fill_halfplane(ax,
                   lambda yy: boundary_line(lda_under, yy),
                   Y_LIM, X_LIM,
                   color_left=c_min, color_right=c_maj, alpha=0.13)

    # Discarded majority points (faded grey)
    sc_drop = ax.scatter(X_dropped[:, 0], X_dropped[:, 1],
                         c="#BBBBBB", s=25, alpha=0.40, edgecolors="none", zorder=2,
                         label=f"Discarded  (n = {N_MAJORITY - N_MINORITY})")
    sc_kept = ax.scatter(X_kept[:, 0], X_kept[:, 1],
                         c=c_maj, s=25, alpha=0.65, edgecolors="none", zorder=3,
                         label=f"Majority kept  (n = {N_MINORITY})")
    sc_min2 = ax.scatter(X1[:, 0], X1[:, 1],
                         c=c_min, s=80, alpha=1.0, marker="*", edgecolors="none", zorder=4,
                         label=f"Minority  (n = {N_MINORITY})")

    # Corrected boundary (dark solid, from undersampled LDA)
    xx_bnd_under = boundary_line(lda_under, yy_plot)
    (line_corr,) = ax.plot(xx_bnd_under, yy_plot, color=c_bnd2, lw=2.2, zorder=5,
                           label=f"Corrected boundary  (x₀ ≈ {x_under_mid:.2f})")
    # Theoretical ideal (same vertical reference as left panel)
    ax.axvline(x=X_IDEAL, color=c_bnd2, lw=1.2, ls=":", alpha=0.55, zorder=5)
    # Show where the biased boundary was (faded red dashed)
    (line_was,) = ax.plot(xx_bnd_imbal, yy_plot, color=c_bnd, lw=1.5, ls="--", alpha=0.45,
                          zorder=5, label=f"Biased (before)  (x₀ ≈ {x_imbal_mid:.2f})")

    ax.legend(handles=[sc_drop, sc_kept, sc_min2, line_corr, line_was],
              fontsize=8, loc="upper right")
    ax.set_title("After undersampling", fontsize=13)
    ax.set_xlabel("$x_0$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    fig.tight_layout()
    fig.savefig(OUTFILE, bbox_inches="tight", dpi=300)
    print(f"[ok] Saved {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
