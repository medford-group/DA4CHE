#!/usr/bin/env python3
"""
Pedagogical oversampling figure.

Two-panel figure showing how class imbalance biases the decision boundary
and how random oversampling "corrects" it — but at the cost of overfitting.

  Left  — Imbalanced dataset (150 majority : 10 minority).
           LDA with empirical priors places the boundary near x ≈ 0.07.
           The dashed theoretical ideal sits at x = 1.00 (midpoint).

  Right — After random oversampling (minority duplicated 15× → 150 points).
           An RBF-kernel SVM fits a complex, non-linear boundary that wraps
           tightly around the duplicated minority cluster positions —
           illustrating overfitting to the repeated samples.
           The theoretical ideal is shown as a dotted reference.

Key pedagogical message
  Oversampling restores class balance, but naive duplication causes a
  flexible classifier to overfit: the boundary memorises the training
  minority positions rather than learning a general decision surface.

Saves: oversampling.png  (same directory as this script)

Run from settings/helper_scripts/:
    python plot_oversampling.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
STYLE_PATH = Path("../plot_style.mplstyle")
OUTFILE    = Path("oversampling.png")
FIGSIZE    = (11, 4.8)
SEED       = 7

N_MAJORITY = 150
N_MINORITY = 10

MU_0 = np.array([2.0, 1.0])   # majority (class 0)
MU_1 = np.array([0.0, 1.0])   # minority (class 1)
COV  = np.diag([0.7, 0.5])

X_LIM = (-1.5, 3.5)
Y_LIM = (-0.8, 2.8)

MIDPOINT = 0.5 * (MU_0[0] + MU_1[0])   # 1.0
X_IDEAL  = MIDPOINT                     # theoretical Bayes boundary

# RBF-SVM parameters — tight kernel to emphasise per-point overfitting
SVM_C     = 10.0
SVM_GAMMA = 1.5
# ──────────────────────────────────────────────────────────────────────────────


def boundary_line(clf, y_range):
    """Return x₀ values of a linear decision boundary across y_range."""
    w, b = clf.coef_[0], clf.intercept_[0]
    return -(w[1] * y_range + b) / w[0]


def fill_halfplane(ax, x_boundary_fn, y_range, x_lim, color_left, color_right, alpha=0.12):
    """Fill left/right half-planes defined by a tilted linear boundary."""
    yy = np.linspace(*y_range, 200)
    xx_bnd = x_boundary_fn(yy)
    xx_bnd = np.clip(xx_bnd, x_lim[0], x_lim[1])

    left_poly_x  = np.concatenate([[x_lim[0]], xx_bnd, [x_lim[0]]])
    left_poly_y  = np.concatenate([[y_range[0]], yy,   [y_range[1]]])
    right_poly_x = np.concatenate([[x_lim[1]], xx_bnd, [x_lim[1]]])
    right_poly_y = np.concatenate([[y_range[0]], yy,   [y_range[1]]])

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

    c_maj  = pal(0)     # navy  — majority
    c_min  = pal(1)     # gold  — minority
    c_bnd  = "#C0392B"  # red   — biased boundary
    c_bnd2 = "#333333"  # dark grey — ideal / corrected

    rng = np.random.default_rng(SEED)

    # ── Generate data ──────────────────────────────────────────────────────────
    X0 = rng.multivariate_normal(MU_0, COV, N_MAJORITY)
    X1 = rng.multivariate_normal(MU_1, COV, N_MINORITY)

    X_imbal = np.vstack([X0, X1])
    y_imbal = np.array([0] * N_MAJORITY + [1] * N_MINORITY)

    # ── Random oversampling: duplicate minority until balanced ─────────────────
    n_repeats = N_MAJORITY // N_MINORITY          # 15
    X1_over   = np.tile(X1, (n_repeats, 1))       # 150 exact copies of the 10 points
    X_over    = np.vstack([X0, X1_over])
    y_over    = np.array([0] * N_MAJORITY + [1] * len(X1_over))

    # ── Train classifiers ──────────────────────────────────────────────────────
    lda_imbal = LinearDiscriminantAnalysis().fit(X_imbal, y_imbal)
    svm_over  = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA).fit(X_over, y_over)

    # LDA boundary position for annotation
    y_mid = 1.0
    w, b = lda_imbal.coef_[0], lda_imbal.intercept_[0]
    x_imbal_mid = -(w[1] * y_mid + b) / w[0]

    print(f"LDA imbalanced boundary:     x₀ ≈ {x_imbal_mid:.3f}  (at y=1)")
    print(f"Theoretical ideal:           x₀ ≈ {X_IDEAL:.3f}")
    print(f"SVM support vectors per class: {svm_over.n_support_}")

    yy_plot      = np.linspace(Y_LIM[0], Y_LIM[1], 300)
    xx_bnd_imbal = boundary_line(lda_imbal, yy_plot)

    # Meshgrid for SVM decision surface
    res = 500
    xx_mesh, yy_mesh = np.meshgrid(
        np.linspace(X_LIM[0], X_LIM[1], res),
        np.linspace(Y_LIM[0], Y_LIM[1], res),
    )
    Z = svm_over.decision_function(
        np.c_[xx_mesh.ravel(), yy_mesh.ravel()]
    ).reshape(xx_mesh.shape)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)

    # ── Left panel: imbalanced dataset ─────────────────────────────────────────
    ax = axes[0]

    fill_halfplane(ax,
                   lambda yy: boundary_line(lda_imbal, yy),
                   Y_LIM, X_LIM,
                   color_left=c_min, color_right=c_maj, alpha=0.13)

    sc0 = ax.scatter(X0[:, 0], X0[:, 1],
                     c=c_maj, s=25, alpha=0.65, edgecolors="none", zorder=3,
                     label=f"Majority  (n = {N_MAJORITY})")
    sc1 = ax.scatter(X1[:, 0], X1[:, 1],
                     c=c_min, s=80, alpha=1.0, marker="*", edgecolors="none", zorder=4,
                     label=f"Minority  (n = {N_MINORITY})")

    (line_bias,) = ax.plot(xx_bnd_imbal, yy_plot, color=c_bnd, lw=2.2, zorder=5,
                           label=f"Biased boundary  (x₀ ≈ {x_imbal_mid:.2f})")
    line_ideal   = ax.axvline(x=X_IDEAL, color=c_bnd2, lw=1.8, ls="--", zorder=5,
                               label=f"Ideal boundary  (x₀ = {X_IDEAL:.2f})")

    ax.legend(handles=[sc0, sc1, line_bias, line_ideal], fontsize=8, loc="upper right")
    ax.set_title("Imbalanced dataset", fontsize=13)
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    # ── Right panel: after oversampling, RBF SVM ───────────────────────────────
    ax = axes[1]

    # Decision regions from SVM (contourf on the meshgrid)
    # decision_function > 0 → class 1 (minority), < 0 → class 0 (majority)
    ax.contourf(xx_mesh, yy_mesh, Z,
                levels=[-1e9, 0, 1e9],
                colors=[c_maj, c_min], alpha=0.13, zorder=0)

    # SVM decision boundary (non-linear)
    ax.contour(xx_mesh, yy_mesh, Z,
               levels=[0], colors=[c_bnd2], linewidths=2.2, zorder=5)

    # Biased boundary for reference (faded red dashed)
    (line_was,) = ax.plot(xx_bnd_imbal, yy_plot, color=c_bnd, lw=1.5, ls="--", alpha=0.45,
                          zorder=5, label=f"Biased (before)  (x₀ ≈ {x_imbal_mid:.2f})")

    # Theoretical ideal (dotted, same reference as left panel)
    ax.axvline(x=X_IDEAL, color=c_bnd2, lw=1.2, ls=":", alpha=0.55, zorder=5)

    # Scatter: majority, then minority (original positions only — duplicates stack)
    sc0b = ax.scatter(X0[:, 0], X0[:, 1],
                      c=c_maj, s=25, alpha=0.65, edgecolors="none", zorder=3,
                      label=f"Majority  (n = {N_MAJORITY})")
    sc1b = ax.scatter(X1[:, 0], X1[:, 1],
                      c=c_min, s=80, alpha=1.0, marker="*", edgecolors="none", zorder=4,
                      label=f"Minority  (n = {N_MINORITY}, oversampled ×{n_repeats})")

    # Proxy artist for the SVM boundary in the legend
    proxy_svm = Line2D([0], [0], color=c_bnd2, lw=2.2,
                       label="Overfit boundary (RBF SVM)")

    ax.legend(handles=[sc0b, sc1b, proxy_svm, line_was],
              fontsize=8, loc="upper right")
    ax.set_title("After oversampling (RBF SVM)", fontsize=13)
    ax.set_xlabel("$x_0$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    fig.tight_layout()
    fig.savefig(OUTFILE, bbox_inches="tight", dpi=300)
    print(f"[ok] Saved {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
