#!/usr/bin/env python3
"""
Pedagogical SMOTE figure.

Two-panel figure contrasting the imbalanced decision boundary with the
corrected boundary after Synthetic Minority Over-sampling Technique (SMOTE).

  Left  — Imbalanced dataset (150 majority : 10 minority).
           LDA with empirical priors places the boundary near x ≈ 0.07.
           The dashed theoretical ideal sits at x = 1.00 (midpoint).

  Right — After SMOTE (10 original + 140 synthetic minority samples).
           Synthetic points (hollow gold circles) are created by interpolating
           between existing minority points — they fill the minority cluster
           smoothly rather than repeating exact coordinates.
           An RBF-kernel SVM trained on the augmented data produces a much
           smoother boundary than naive oversampling, closer to the ideal.

Key pedagogical message
  SMOTE generates plausible synthetic minority samples by interpolation,
  giving the classifier a smoother minority region to learn from.  The
  resulting decision boundary is more generalizable than one learned from
  exact duplicates, and avoids the isolated "island" artefacts of naive
  random oversampling.

Saves: smote.png  (same directory as this script)

Run from settings/helper_scripts/:
    python plot_smote.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
STYLE_PATH = Path("../plot_style.mplstyle")
OUTFILE    = Path("smote.png")
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

# SMOTE parameters
SMOTE_K = 5      # number of nearest neighbours

# RBF-SVM parameters (same as oversampling figure for a fair comparison)
SVM_C     = 10.0
SVM_GAMMA = 1.5
# ──────────────────────────────────────────────────────────────────────────────


def smote(X_min, n_synthetic, k=5, rng=None):
    """
    Generate synthetic minority samples via SMOTE interpolation.

    For each synthetic sample:
      1. Pick a random minority point x_i.
      2. Pick a random neighbour x_nn from its k nearest minority neighbours.
      3. Sample x_new = x_i + λ · (x_nn − x_i),  λ ~ Uniform[0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    # k+1 because the point itself is the closest neighbour
    nn_k = min(k, len(X_min) - 1)
    nbrs = NearestNeighbors(n_neighbors=nn_k + 1).fit(X_min)
    _, indices = nbrs.kneighbors(X_min)   # shape (n_min, nn_k+1)

    synthetic = np.empty((n_synthetic, X_min.shape[1]))
    for idx in range(n_synthetic):
        i      = rng.integers(len(X_min))
        nn_idx = rng.choice(indices[i, 1:])          # exclude self (col 0)
        lam    = rng.uniform()
        synthetic[idx] = X_min[i] + lam * (X_min[nn_idx] - X_min[i])

    return synthetic


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
    c_bnd2 = "#333333"  # dark grey — ideal / SVM boundary

    rng = np.random.default_rng(SEED)

    # ── Generate base data ─────────────────────────────────────────────────────
    X0 = rng.multivariate_normal(MU_0, COV, N_MAJORITY)
    X1 = rng.multivariate_normal(MU_1, COV, N_MINORITY)

    X_imbal = np.vstack([X0, X1])
    y_imbal = np.array([0] * N_MAJORITY + [1] * N_MINORITY)

    # ── SMOTE: generate synthetic minority samples ─────────────────────────────
    n_synthetic = N_MAJORITY - N_MINORITY          # 140 new points → total 150
    X1_synthetic = smote(X1, n_synthetic, k=SMOTE_K, rng=rng)
    X1_augmented = np.vstack([X1, X1_synthetic])  # 10 original + 140 synthetic

    X_smote = np.vstack([X0, X1_augmented])
    y_smote = np.array([0] * N_MAJORITY + [1] * len(X1_augmented))

    # ── Train classifiers ──────────────────────────────────────────────────────
    lda_imbal = LinearDiscriminantAnalysis().fit(X_imbal, y_imbal)
    svm_smote  = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA).fit(X_smote, y_smote)

    # LDA boundary position for annotation
    y_mid = 1.0
    w, b = lda_imbal.coef_[0], lda_imbal.intercept_[0]
    x_imbal_mid = -(w[1] * y_mid + b) / w[0]

    print(f"LDA imbalanced boundary:     x₀ ≈ {x_imbal_mid:.3f}  (at y=1)")
    print(f"Theoretical ideal:           x₀ ≈ {X_IDEAL:.3f}")
    print(f"SVM support vectors per class: {svm_smote.n_support_}")
    print(f"Synthetic points generated:  {n_synthetic}")

    yy_plot      = np.linspace(Y_LIM[0], Y_LIM[1], 300)
    xx_bnd_imbal = boundary_line(lda_imbal, yy_plot)

    # Meshgrid for SVM decision surface
    res = 500
    xx_mesh, yy_mesh = np.meshgrid(
        np.linspace(X_LIM[0], X_LIM[1], res),
        np.linspace(Y_LIM[0], Y_LIM[1], res),
    )
    Z = svm_smote.decision_function(
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

    # ── Right panel: after SMOTE, RBF SVM ─────────────────────────────────────
    ax = axes[1]

    # Decision regions from SVM
    # decision_function > 0 → class 1 (minority), < 0 → class 0 (majority)
    ax.contourf(xx_mesh, yy_mesh, Z,
                levels=[-1e9, 0, 1e9],
                colors=[c_maj, c_min], alpha=0.13, zorder=0)

    # SVM decision boundary
    ax.contour(xx_mesh, yy_mesh, Z,
               levels=[0], colors=[c_bnd2], linewidths=2.2, zorder=5)

    # Biased boundary for reference (faded red dashed)
    (line_was,) = ax.plot(xx_bnd_imbal, yy_plot, color=c_bnd, lw=1.5, ls="--", alpha=0.45,
                          zorder=5, label=f"Biased (before)  (x₀ ≈ {x_imbal_mid:.2f})")

    # Theoretical ideal (dotted reference)
    ax.axvline(x=X_IDEAL, color=c_bnd2, lw=1.2, ls=":", alpha=0.55, zorder=5)

    # Scatter: synthetic first (below originals), then majority, then original minority
    sc_syn = ax.scatter(X1_synthetic[:, 0], X1_synthetic[:, 1],
                        c="none", s=30, alpha=0.6,
                        edgecolors=c_min, linewidths=0.8, zorder=3,
                        label=f"Synthetic minority  (n = {n_synthetic})")
    sc0b = ax.scatter(X0[:, 0], X0[:, 1],
                      c=c_maj, s=25, alpha=0.65, edgecolors="none", zorder=3,
                      label=f"Majority  (n = {N_MAJORITY})")
    sc1b = ax.scatter(X1[:, 0], X1[:, 1],
                      c=c_min, s=80, alpha=1.0, marker="*", edgecolors="none", zorder=4,
                      label=f"Minority original  (n = {N_MINORITY})")

    # Proxy artist for the SVM boundary in the legend
    proxy_svm = Line2D([0], [0], color=c_bnd2, lw=2.2,
                       label="Corrected boundary (RBF SVM + SMOTE)")

    ax.legend(handles=[sc_syn, sc0b, sc1b, proxy_svm, line_was],
              fontsize=8, loc="upper right")
    ax.set_title("After SMOTE (RBF SVM)", fontsize=13)
    ax.set_xlabel("$x_0$")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)

    fig.tight_layout()
    fig.savefig(OUTFILE, bbox_inches="tight", dpi=300)
    print(f"[ok] Saved {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
