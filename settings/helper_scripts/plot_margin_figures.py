#!/usr/bin/env python3
"""
Margin loss function figures for the GLM chapter.

Figure 1 — margin_cost.png
  Two linearly separable classes, the decision boundary, the two margin
  boundaries (dashed), the shaded margin zone, and highlighted support
  vectors.  A point is visible inside the margin on each side to illustrate
  that in-margin points incur a penalty even when correctly classified.

Figure 2 — margin_size.png
  Side-by-side comparison showing the same dataset with two different weight
  vectors.  Left: maximum-margin solution (small ||w̃||, wide margin).
  Right: sub-optimal boundary (large ||w̃||, narrow margin).  Margin widths
  are annotated to make the 2/||w̃|| relationship concrete.

Pedagogical message
  The margin enforces a buffer zone.  Maximizing the margin width is
  equivalent to minimizing the weight-vector norm, which is the key idea
  behind support vector machines.

Saves: margin_cost.png, margin_size.png
       (this directory and ../../3-classification/images/)

Run from settings/helper_scripts/:
    python plot_margin_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE  = Path(__file__).parent
IMGS  = HERE / '../../3-classification/images'
plt.style.use(HERE / '../plot_style.mplstyle')

NAVY = '#003057'
GOLD = '#EAAA00'
TEAL = '#4B8B9B'
LITE = '#E8F0F5'

# ─── Reproducible, well-separated toy data ───────────────────────────────────
np.random.seed(12)
X, y_raw = make_blobs(n_samples=50, centers=[[-2, -1.5], [2, 1.5]],
                      cluster_std=0.65, n_features=2)
y = y_raw * 2 - 1   # {0,1} → {-1,+1}

# x-range for boundary lines
x_lo, x_hi = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
xr = np.linspace(x_lo, x_hi, 300)


def boundary_lines(model):
    """Return (y_db, y_m1, y_p1) boundary line arrays over xr."""
    w = model.coef_[0]
    b = model.intercept_[0]
    y_db = -(w[0] * xr + b)       / w[1]
    y_m1 = -(w[0] * xr + b - 1)   / w[1]
    y_p1 = -(w[0] * xr + b + 1)   / w[1]
    return y_db, y_m1, y_p1


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — margin_cost.png
# ══════════════════════════════════════════════════════════════════════════════
model_hard = SVC(kernel='linear', C=1e6)
model_hard.fit(X, y)

y_db, y_m1, y_p1 = boundary_lines(model_hard)
w_hard = model_hard.coef_[0]

fig, ax = plt.subplots(figsize=(6.5, 5.0))

# Shaded margin region
ax.fill_between(xr, y_m1, y_p1, color=TEAL, alpha=0.12, zorder=1)

# Data
ax.scatter(X[y == -1, 0], X[y == -1, 1], c=NAVY, s=55, zorder=4, label='Class −1')
ax.scatter(X[y == +1, 0], X[y == +1, 1], c=GOLD, s=55, zorder=4, label='Class +1')

# Boundaries
ax.plot(xr, y_db, color='k',    lw=2.2, label='Decision boundary ($\mathbf{x}^\top\mathbf{w}=0$)')
ax.plot(xr, y_m1, color=TEAL, lw=1.8, ls='--', label='Margin boundaries ($\pm 1$)')
ax.plot(xr, y_p1, color=TEAL, lw=1.8, ls='--')

# Support vectors
sv = model_hard.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=220, facecolors='none',
           edgecolors='k', linewidths=1.8, zorder=5, label='Support vectors')

# ── Margin width annotation ────────────────────────────────────────────────
# Pick a point on the decision boundary and draw a perpendicular arrow
# spanning from -1 margin to +1 margin.
w_norm = w_hard / np.linalg.norm(w_hard)
perp   = np.array([w_norm[1], -w_norm[0]])   # perpendicular direction

# Annotation anchor: centre of the plot domain
x_ann  = X[:, 0].mean() + 1.0
y_ann  = -(w_hard[0] * x_ann + model_hard.intercept_[0]) / w_hard[1]
half_w = 1.0 / np.linalg.norm(w_hard)        # half-margin in data units

pt_lo  = np.array([x_ann, y_ann]) - half_w * perp[::-1] * np.array([1, 1])
pt_hi  = np.array([x_ann, y_ann]) + half_w * perp[::-1] * np.array([1, 1])

# Clamp to visible region
ax.annotate('', xy=(pt_hi[0], pt_hi[1]), xytext=(pt_lo[0], pt_lo[1]),
            arrowprops=dict(arrowstyle='<->', color=TEAL, lw=2.0,
                            mutation_scale=14))

margin_width = 2.0 / np.linalg.norm(w_hard)
ax.text(x_ann + 0.25, y_ann,
        f'margin\n= {margin_width:.2f}',
        ha='left', va='center', fontsize=9, color=TEAL, fontweight='bold')

ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.legend(fontsize=8, loc='upper left')
ax.set_title('Margin Loss: buffer zone around the decision boundary')
ax.set_xlim(x_lo, x_hi)

plt.tight_layout()
for out in (IMGS / 'margin_cost.png', HERE / 'margin_cost.png'):
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — margin_size.png
# ══════════════════════════════════════════════════════════════════════════════
# For the narrow-margin panel we deliberately use a sub-optimal weight vector
# (not the max-margin solution) to illustrate that larger ||w̃|| ↔ smaller margin.

# Manufacture a sub-optimal weight vector with a larger norm
w_opt   = model_hard.coef_[0].copy()
b_opt   = model_hard.intercept_[0]

# Scale the weights up by 4× (same boundary, but 4× larger norm → 4× narrower margin)
scale = 4.0
w_narrow = w_opt * scale
b_narrow = b_opt * scale

def lines_from_w(w, b):
    y_db = -(w[0] * xr + b)       / w[1]
    y_m1 = -(w[0] * xr + b - 1)   / w[1]
    y_p1 = -(w[0] * xr + b + 1)   / w[1]
    return y_db, y_m1, y_p1

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

configs = [
    (w_opt,    b_opt,    'Wide margin\n(small $\\|\\tilde{\\mathbf{w}}\\|$)'),
    (w_narrow, b_narrow, 'Narrow margin\n(large $\\|\\tilde{\\mathbf{w}}\\|$)'),
]

for ax, (w_i, b_i, title) in zip(axes, configs):
    y_db_i, y_m1_i, y_p1_i = lines_from_w(w_i, b_i)
    margin_i = 2.0 / np.linalg.norm(w_i)

    # Shaded margin
    ax.fill_between(xr, y_m1_i, y_p1_i, color=TEAL, alpha=0.14, zorder=1)

    # Data
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c=NAVY, s=50, zorder=4)
    ax.scatter(X[y == +1, 0], X[y == +1, 1], c=GOLD, s=50, zorder=4)

    # Boundaries
    ax.plot(xr, y_db_i, 'k-',      lw=2.2)
    ax.plot(xr, y_m1_i, color=TEAL, lw=1.8, ls='--')
    ax.plot(xr, y_p1_i, color=TEAL, lw=1.8, ls='--')

    # Support vectors (only available for the max-margin model)
    if np.linalg.norm(w_i - w_opt) < 1e-6:
        ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
                   edgecolors='k', linewidths=1.8, zorder=5)

    # Margin annotation
    norm_i = w_i / np.linalg.norm(w_i)
    perp_i = np.array([norm_i[1], -norm_i[0]])
    x_ann  = X[:, 0].mean() + 0.8
    y_ann  = -(w_i[0] * x_ann + b_i) / w_i[1]
    hw     = 1.0 / np.linalg.norm(w_i)
    pt_lo  = np.array([x_ann, y_ann]) - hw * perp_i[::-1]
    pt_hi  = np.array([x_ann, y_ann]) + hw * perp_i[::-1]
    ax.annotate('', xy=(pt_hi[0], pt_hi[1]), xytext=(pt_lo[0], pt_lo[1]),
                arrowprops=dict(arrowstyle='<->', color=TEAL, lw=2.0,
                                mutation_scale=12))
    ax.text(x_ann + 0.2, y_ann,
            f'$2/\\|\\tilde{{\\mathbf{{w}}}}\\|$ = {margin_i:.2f}',
            ha='left', va='center', fontsize=9, color=TEAL, fontweight='bold')

    ax.set_xlabel('$x_0$')
    ax.set_title(title, fontsize=11)
    ax.set_xlim(x_lo, x_hi)

axes[0].set_ylabel('$x_1$')
fig.suptitle(
    'Margin width $= 2/\\|\\tilde{\\mathbf{w}}\\|$ — minimizing $\\|\\tilde{\\mathbf{w}}\\|$ maximizes the margin',
    fontsize=11, y=1.02)

plt.tight_layout()
for out in (IMGS / 'margin_size.png', HERE / 'margin_size.png'):
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
plt.close()
