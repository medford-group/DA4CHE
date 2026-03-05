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

Annotation strategy
  The double-headed arrow is drawn so that it looks visually perpendicular
  to the decision boundary on screen, regardless of the axis aspect ratio.
  After tight_layout() + canvas.draw(), we work in display (pixel) space:
    1. Compute the margin width in pixels along the true normal direction.
    2. Rotate the boundary tangent 90° in pixel space to get the visual
       perpendicular direction.
    3. Place the arrow along that direction with the correct pixel length.
    4. Convert the two endpoints back to data coordinates for annotate().

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

# x-range for boundary lines
x_lo, x_hi = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
xr = np.linspace(x_lo, x_hi, 300)


def boundary_lines(w, b):
    """Return (y_db, y_m1, y_p1) boundary line arrays over xr."""
    y_db = -(w[0] * xr + b)       / w[1]
    y_m1 = -(w[0] * xr + b - 1)   / w[1]
    y_p1 = -(w[0] * xr + b + 1)   / w[1]
    return y_db, y_m1, y_p1


def perp_annotation(ax, w, x_ann, y_ann):
    """
    Return (pt_lo, pt_hi) in data coordinates for a double-headed arrow that
    is visually perpendicular to the SVM boundary w·x + b = 0 and spans the
    full margin (2/||w||).

    Works by rotating the boundary tangent 90° in display (pixel) space, then
    converting the endpoints back to data coordinates.
    """
    trans = ax.transData
    inv   = trans.inverted()

    center      = np.array([x_ann, y_ann])
    center_disp = trans.transform(center)

    # Margin width in display pixels (measured along true normal w_hat)
    w_hat  = w / np.linalg.norm(w)
    hw     = 1.0 / np.linalg.norm(w)
    hi_disp = trans.transform(center + hw * w_hat)
    lo_disp = trans.transform(center - hw * w_hat)
    margin_disp_len = np.linalg.norm(hi_disp - lo_disp)

    # Boundary tangent direction in display space (small step along tangent)
    tang_data = np.array([-w[1], w[0]])
    tang_data = tang_data / np.linalg.norm(tang_data)
    tang_disp = trans.transform(center + 0.01 * tang_data) - center_disp
    tang_disp_hat = tang_disp / np.linalg.norm(tang_disp)

    # 90° rotation of tangent in display space → visual perpendicular
    perp_disp_hat = np.array([-tang_disp_hat[1], tang_disp_hat[0]])

    pt_lo = inv.transform(center_disp - perp_disp_hat * margin_disp_len / 2)
    pt_hi = inv.transform(center_disp + perp_disp_hat * margin_disp_len / 2)
    return pt_lo, pt_hi


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

# ── Finalise layout, then add the perpendicular annotation ────────────────────
plt.tight_layout()
fig.canvas.draw()

x_ann = X[:, 0].mean() + 1.0
y_ann = -(w_hard[0] * x_ann + b_hard) / w_hard[1]
pt_lo, pt_hi = perp_annotation(ax, w_hard, x_ann, y_ann)

ax.annotate('', xy=(pt_hi[0], pt_hi[1]), xytext=(pt_lo[0], pt_lo[1]),
            arrowprops=dict(arrowstyle='<->', color=TEAL, lw=2.0, mutation_scale=14))

margin_width = 2.0 / np.linalg.norm(w_hard)
tang_hat = np.array([-w_hard[1], w_hard[0]]) / np.linalg.norm(w_hard)
label_off = tang_hat * 0.35
ax.text(x_ann + label_off[0], y_ann + label_off[1],
        f'margin = {margin_width:.2f}',
        ha='center', va='center', fontsize=9, color=TEAL, fontweight='bold')

for out in (IMGS / 'margin_cost.png', HERE / 'margin_cost.png'):
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — margin_size.png
# ══════════════════════════════════════════════════════════════════════════════
w_opt    = model_hard.coef_[0].copy()
b_opt    = model_hard.intercept_[0]
scale    = 4.0
w_narrow = w_opt * scale
b_narrow = b_opt * scale

configs = [
    (w_opt,    b_opt,    'Wide margin\n(small $\\|\\tilde{\\mathbf{w}}\\|$)'),
    (w_narrow, b_narrow, 'Narrow margin\n(large $\\|\\tilde{\\mathbf{w}}\\|$)'),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

ann_params = []   # store annotation info, added after layout is finalised
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

    x_ann = X[:, 0].mean() + 0.8
    y_ann = -(w_i[0] * x_ann + b_i) / w_i[1]
    ann_params.append((ax, w_i, b_i, x_ann, y_ann))

axes[0].set_ylabel('$x_1$')
fig.suptitle(
    'Margin width $= 2/\\|\\tilde{\\mathbf{w}}\\|$ — '
    'minimizing $\\|\\tilde{\\mathbf{w}}\\|$ maximizes the margin',
    fontsize=11, y=1.02)

# ── Finalise layout, then add perpendicular annotations ───────────────────────
plt.tight_layout()
fig.canvas.draw()

for ax, w_i, b_i, x_ann, y_ann in ann_params:
    margin_i = 2.0 / np.linalg.norm(w_i)
    pt_lo, pt_hi = perp_annotation(ax, w_i, x_ann, y_ann)

    ax.annotate('', xy=(pt_hi[0], pt_hi[1]), xytext=(pt_lo[0], pt_lo[1]),
                arrowprops=dict(arrowstyle='<->', color=TEAL, lw=2.0, mutation_scale=12))

    tang_hat = np.array([-w_i[1], w_i[0]]) / np.linalg.norm(w_i)
    label_off = tang_hat * 0.3
    ax.text(x_ann + label_off[0], y_ann + label_off[1],
            f'$2/\\|\\tilde{{\\mathbf{{w}}}}\\|$ = {margin_i:.2f}',
            ha='center', va='center', fontsize=9, color=TEAL, fontweight='bold')

for out in (IMGS / 'margin_size.png', HERE / 'margin_size.png'):
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
plt.close()
