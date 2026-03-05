#!/usr/bin/env python3
"""
Kernel trick schematic for the GLM chapter.

Two-panel figure illustrating how a feature map φ(x) can transform a
non-linearly separable dataset into a linearly separable one:

  Left panel  — Original 2-D feature space.  Inner-ring points (class −1)
                and outer-ring points (class +1) are arranged concentrically;
                no straight line can separate them.

  Right panel — Transformed space using the Gaussian feature map
                x₂ = exp(−(x₀² + x₁²)).  In the (x₀, x₂) projection
                the two classes are linearly separable, and a horizontal
                decision boundary is shown.

The φ(x) arrow between panels reinforces that the transformation is
implicit — the kernel trick computes it without materialising the
high-dimensional representation.

Pedagogical message
  Non-linearly separable data can often be mapped into a higher-dimensional
  space where a linear classifier succeeds.  The RBF kernel approximates
  the inner product in an infinite-dimensional version of this space.

Saves: kernel_schematic.png
       (this directory and ../../3-classification/images/)

Run from settings/helper_scripts/:
    python plot_kernel_schematic.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.datasets import make_circles
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
IMGS = HERE / '../../3-classification/images'
plt.style.use(HERE / '../plot_style.mplstyle')

NAVY = '#003057'
GOLD = '#EAAA00'
TEAL = '#4B8B9B'

# ─── Data ─────────────────────────────────────────────────────────────────────
np.random.seed(7)
X, y = make_circles(n_samples=120, factor=0.42, noise=0.07, random_state=7)
clrs = np.where(y == 0, NAVY, GOLD)

# Gaussian transform  x₂ = exp(−(x₀² + x₁²))
x2 = np.exp(-(X[:, 0]**2 + X[:, 1]**2))

# Decision threshold in transformed space (fitted manually)
thresh = 0.52   # separates the two rings cleanly

# ─── Layout: 3 columns with a narrow centre column for the arrow label ────────
fig = plt.figure(figsize=(11, 4.5))
gs  = fig.add_gridspec(1, 3, width_ratios=[5, 1, 5], wspace=0.05)

ax_left  = fig.add_subplot(gs[0])
ax_mid   = fig.add_subplot(gs[1])
ax_right = fig.add_subplot(gs[2])
ax_mid.axis('off')

# ── Left panel: original 2-D space ───────────────────────────────────────────
ax_left.scatter(X[:, 0], X[:, 1], c=clrs, s=55, zorder=3)

# Show the ideal (circular) boundary that would work — but can't be linear
theta = np.linspace(0, 2 * np.pi, 300)
r_sep = 0.72
ax_left.plot(r_sep * np.cos(theta), r_sep * np.sin(theta),
             'k--', lw=1.6, label='Ideal boundary\n(circle, non-linear)')

ax_left.set_xlabel('$x_0$', fontsize=12)
ax_left.set_ylabel('$x_1$', fontsize=12)
ax_left.set_title('Original space\n(not linearly separable)', fontsize=11)
ax_left.legend(fontsize=8, loc='upper right')
ax_left.set_aspect('equal')

# ── Middle column: φ(x) arrow + label ────────────────────────────────────────
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 1)
ax_mid.annotate('',
    xy=(0.92, 0.5), xytext=(0.08, 0.5),
    xycoords='axes fraction', textcoords='axes fraction',
    arrowprops=dict(arrowstyle='->', color=TEAL, lw=2.5,
                    mutation_scale=20))
ax_mid.text(0.5, 0.62, r'$\phi(\mathbf{x})$',
            ha='center', va='bottom', fontsize=14,
            color=TEAL, fontweight='bold',
            transform=ax_mid.transAxes)

# ── Right panel: transformed space (x₀, x₂) ──────────────────────────────────
ax_right.scatter(X[:, 0], x2, c=clrs, s=55, zorder=3)

# Horizontal decision boundary at threshold
x_lo = X[:, 0].min() - 0.1
x_hi = X[:, 0].max() + 0.1
ax_right.axhline(thresh, color='k', lw=2.0, label='Decision boundary\n(linear)', zorder=4)

ax_right.set_xlabel('$x_0$', fontsize=12)
ax_right.set_ylabel(r'$x_2 = e^{-(x_0^2 + x_1^2)}$', fontsize=11)
ax_right.set_title('Transformed space\n(linearly separable)', fontsize=11)
ax_right.legend(fontsize=8, loc='upper right')
ax_right.set_xlim(x_lo, x_hi)

# ── Shared legend patch for class colours ─────────────────────────────────────
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=NAVY, label='Class 0'),
              Patch(facecolor=GOLD, label='Class 1')]
fig.legend(handles=legend_els, loc='lower center', ncol=2, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
for out in (IMGS / 'kernel_schematic.png', HERE / 'kernel_schematic.png'):
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
plt.close()
