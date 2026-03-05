#!/usr/bin/env python3
"""
Margin loss function figures for the GLM chapter.

Figure 1 — margin_cost.png
  Two linearly separable classes, the decision boundary, the two margin
  boundaries (dashed), the shaded margin zone, and highlighted support
  vectors.

Figure 2 — margin_size.png
  Side-by-side comparison: maximum-margin (wide) vs. sub-optimal (narrow).

Annotation strategy
  After tight_layout() + canvas.draw() we work in display (pixel) space:
    1. Take a 1-pixel step in the visual-perpendicular direction.
    2. Compute how much w·x changes per pixel (w_dot_step).
    3. px_to_margin = 1 / |w_dot_step| gives the exact pixel count to span
       from the decision boundary to each margin boundary.
  This guarantees the arrow is visually perpendicular AND correctly sized.
  Label placed below the lower margin line, right-aligned to the arrow's
  bottom (leftward) endpoint.

Saves: margin_cost.png, margin_size.png
       (this directory and ../../3-classification/images/)

Run from settings/helper_scripts/:
    python plot_margin_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# ─── Reproducible, well-separated toy data ───────────────────────────────────
np.random.seed(12)
X, y_raw = make_blobs(n_samples=50, centers=[[-2, -1.5], [2, 1.5]],
                      cluster_std=0.65, n_features=2)
y = y_raw * 2 - 1   # {0,1} → {-1,+1}

x_lo, x_hi = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
xr = np.linspace(x_lo, x_hi, 300)


def boundary_lines(w, b):
    y_db = -(w[0] * xr + b)       / w[1]
    y_m1 = -(w[0] * xr + b - 1)   / w[1]
    y_p1 = -(w[0] * xr + b + 1)   / w[1]
    return y_db, y_m1, y_p1


def add_margin_annotation(ax, w, b, x_ann, label_str, arrow_kw, fontsize=9):
    """
    Draw a double-headed arrow that is visually perpendicular to the SVM
    decision boundary and exactly spans from one margin boundary to the other.
    Place the label below the lower margin line, right-aligned to the arrow's
    bottom endpoint.

    Must be called after tight_layout() + fig.canvas.draw().
    """
    trans = ax.transData
    inv   = trans.inverted()

    y_ann       = -(w[0] * x_ann + b) / w[1]   # point on decision boundary
    center      = np.array([x_ann, y_ann])
    center_disp = trans.transform(center)

    # ── Boundary tangent in display space ────────────────────────────────────
    tang_data     = np.array([-w[1], w[0]])
    tang_data_hat = tang_data / np.linalg.norm(tang_data)
    tang_disp     = trans.transform(center + 0.01 * tang_data_hat) - center_disp
    tang_disp_hat = tang_disp / np.linalg.norm(tang_disp)

    # ── Visual perpendicular: 90° CCW rotation of tangent in display space ───
    perp_disp_hat = np.array([-tang_disp_hat[1], tang_disp_hat[0]])

    # ── Arrow length: pixels needed to reach each margin boundary ────────────
    # One pixel in perp direction → this data displacement:
    step_data = inv.transform(center_disp + perp_disp_hat) - center
    w_dot_step = np.dot(w, step_data)          # change in w·x per pixel

    # Ensure perp_disp_hat points toward the +1 margin (w·x+b > 0 side)
    if w_dot_step < 0:
        perp_disp_hat = -perp_disp_hat
        w_dot_step    = -w_dot_step

    px_to_margin = 1.0 / w_dot_step           # pixels from boundary to margin

    pt_lo = inv.transform(center_disp - perp_disp_hat * px_to_margin)
    pt_hi = inv.transform(center_disp + perp_disp_hat * px_to_margin)

    ax.annotate('', xy=(pt_hi[0], pt_hi[1]), xytext=(pt_lo[0], pt_lo[1]),
                arrowprops=dict(**arrow_kw))

    # ── Label: below the lower margin line, right-aligned to pt_lo ───────────
    y_lo_margin = min(-(w[0] * x_ann + b + 1) / w[1],
                      -(w[0] * x_ann + b - 1) / w[1])
    # pt_lo is the lower/leftward endpoint (w·x+b = -1 side)
    ax.text(pt_lo[0], y_lo_margin - 0.2, label_str,
            ha='right', va='top', fontsize=fontsize,
            color=TEAL, fontweight='bold')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — margin_cost.png
# ══════════════════════════════════════════════════════════════════════════════
model_hard = SVC(kernel='linear', C=1e6)
model_hard.fit(X, y)

w_hard = model_hard.coef_[0]
b_hard = model_hard.intercept_[0]
y_db, y_m1, y_p1 = boundary_lines(w_hard, b_hard)
sv = model_hard.support_vectors_

fig, ax = plt.subplots(figsize=(6.5, 5.0))

ax.fill_between(xr, y_m1, y_p1, color=TEAL, alpha=0.12, zorder=1)
ax.scatter(X[y == -1, 0], X[y == -1, 1], c=NAVY, s=55, zorder=4, label='Class −1')
ax.scatter(X[y == +1, 0], X[y == +1, 1], c=GOLD, s=55, zorder=4, label='Class +1')
ax.plot(xr, y_db, color='k',  lw=2.2,
        label='Decision boundary ($\\mathbf{x}^\\top\\mathbf{w}=0$)')
ax.plot(xr, y_m1, color=TEAL, lw=1.8, ls='--', label='Margin boundaries ($\\pm 1$)')
ax.plot(xr, y_p1, color=TEAL, lw=1.8, ls='--')
ax.scatter(sv[:, 0], sv[:, 1], s=220, facecolors='none',
           edgecolors='k', linewidths=1.8, zorder=5, label='Support vectors')

ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.legend(fontsize=8, loc='upper left')
ax.set_title('Margin Loss: buffer zone around the decision boundary')
ax.set_xlim(x_lo, x_hi)

plt.tight_layout()
fig.canvas.draw()

margin_width = 2.0 / np.linalg.norm(w_hard)
add_margin_annotation(
    ax, w_hard, b_hard,
    x_ann      = X[:, 0].mean() + 1.0,
    label_str  = f'margin = {margin_width:.2f}',
    arrow_kw   = dict(arrowstyle='<->', color=TEAL, lw=2.0, mutation_scale=14),
)

for out in (IMGS / 'margin_cost.png', HERE / 'margin_cost.png'):
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — margin_size.png
# ══════════════════════════════════════════════════════════════════════════════
w_opt    = model_hard.coef_[0].copy()
b_opt    = model_hard.intercept_[0]
w_narrow = w_opt * 4.0
b_narrow = b_opt * 4.0

configs = [
    (w_opt,    b_opt,    'Wide margin\n(small $\\|\\tilde{\\mathbf{w}}\\|$)'),
    (w_narrow, b_narrow, 'Narrow margin\n(large $\\|\\tilde{\\mathbf{w}}\\|$)'),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

ann_params = []
for ax, (w_i, b_i, title) in zip(axes, configs):
    y_db_i, y_m1_i, y_p1_i = boundary_lines(w_i, b_i)

    ax.fill_between(xr, y_m1_i, y_p1_i, color=TEAL, alpha=0.14, zorder=1)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c=NAVY, s=50, zorder=4)
    ax.scatter(X[y == +1, 0], X[y == +1, 1], c=GOLD, s=50, zorder=4)
    ax.plot(xr, y_db_i, 'k-',       lw=2.2)
    ax.plot(xr, y_m1_i, color=TEAL, lw=1.8, ls='--')
    ax.plot(xr, y_p1_i, color=TEAL, lw=1.8, ls='--')

    if np.linalg.norm(w_i - w_opt) < 1e-6:
        ax.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none',
                   edgecolors='k', linewidths=1.8, zorder=5)

    ax.set_xlabel('$x_0$')
    ax.set_title(title, fontsize=11)
    ax.set_xlim(x_lo, x_hi)

    ann_params.append((ax, w_i, b_i))

axes[0].set_ylabel('$x_1$')
fig.suptitle(
    'Margin width $= 2/\\|\\tilde{\\mathbf{w}}\\|$ — '
    'minimizing $\\|\\tilde{\\mathbf{w}}\\|$ maximizes the margin',
    fontsize=11, y=1.02)

plt.tight_layout()
fig.canvas.draw()

for ax, w_i, b_i in ann_params:
    margin_i = 2.0 / np.linalg.norm(w_i)
    add_margin_annotation(
        ax, w_i, b_i,
        x_ann     = X[:, 0].mean() + 0.8,
        label_str = f'$2/\\|\\tilde{{\\mathbf{{w}}}}\\|$ = {margin_i:.2f}',
        arrow_kw  = dict(arrowstyle='<->', color=TEAL, lw=2.0, mutation_scale=12),
    )

for out in (IMGS / 'margin_size.png', HERE / 'margin_size.png'):
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
plt.close()
