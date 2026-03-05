#!/usr/bin/env python3
"""
Perceptron as a single-layer neural network diagram.

Pedagogical message
  Shows the structural equivalence between a perceptron and a biological
  neuron: weighted inputs are summed and passed through a step activation
  function to produce a binary class label.

Saves: perceptron_NN.png  (this directory and ../../3-classification/images/)

Run from settings/helper_scripts/:
    python plot_perceptron_nn.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).parent
IMG    = HERE / '../../3-classification/images/perceptron_NN.png'
LOCAL  = HERE / 'perceptron_NN.png'

plt.style.use(HERE / '../plot_style.mplstyle')

# ─── GT palette ───────────────────────────────────────────────────────────────
NAVY = '#003057'
GOLD = '#EAAA00'
GRAY = '#8E8B76'
LITE = '#F5F4EE'

# ─── Canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 12)
ax.set_ylim(-0.5, 6.5)
ax.axis('off')
fig.patch.set_facecolor('white')

r_in  = 0.40   # input node radius
r_sum = 0.60   # summation node radius (larger)

# ─── Input node positions ──────────────────────────────────────────────────────
input_x  = 1.2
input_ys = [5.0, 3.5, 2.0]            # x0, x1, x2
inp_labs = ['$x_0$', '$x_1$', '$x_2$']
wt_labs  = ['$w_0$', '$w_1$', '$w_2$']

bias_y = 0.6   # bias node below the inputs

# ─── Summation neuron ──────────────────────────────────────────────────────────
sum_x, sum_y = 5.5, 3.2

# ─── Step activation box ──────────────────────────────────────────────────────
act_x, act_y = 8.5, 3.2
box_w, box_h = 1.6, 1.2

# ═══════════════════════════════════════════════════════════════════════════════
# Draw input nodes
# ═══════════════════════════════════════════════════════════════════════════════
for y, lab in zip(input_ys, inp_labs):
    c = Circle((input_x, y), r_in,
                facecolor=NAVY, edgecolor='white', linewidth=1.5, zorder=4)
    ax.add_patch(c)
    ax.text(input_x, y, lab,
            ha='center', va='center', color='white',
            fontsize=12, fontweight='bold', zorder=5)

# Ellipsis for "more inputs"
ax.text(input_x, 0.9, r'$\vdots$', ha='center', va='center',
        fontsize=17, color=NAVY, zorder=5)

# Bias node
c_bias = Circle((input_x, bias_y), r_in,
                 facecolor=GRAY, edgecolor='white', linewidth=1.5, zorder=4)
ax.add_patch(c_bias)
ax.text(input_x, bias_y, '+1',
        ha='center', va='center', color='white',
        fontsize=10, fontweight='bold', zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# Draw summation neuron
# ═══════════════════════════════════════════════════════════════════════════════
c_sum = Circle((sum_x, sum_y), r_sum,
                facecolor=GOLD, edgecolor=NAVY, linewidth=2.5, zorder=4)
ax.add_patch(c_sum)
ax.text(sum_x, sum_y, r'$\Sigma$',
        ha='center', va='center', color=NAVY,
        fontsize=20, fontweight='bold', zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# Arrows: inputs → summation neuron
# ═══════════════════════════════════════════════════════════════════════════════
for y, wlab in zip(input_ys, wt_labs):
    # Arrow
    ax.annotate('', xy=(sum_x - r_sum, sum_y), xytext=(input_x + r_in, y),
                arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.8), zorder=3)
    # Weight label at midpoint, offset slightly above the line
    mid_x = (input_x + r_in + sum_x - r_sum) / 2
    mid_y = (y + sum_y) / 2
    angle  = np.degrees(np.arctan2(sum_y - y, sum_x - r_sum - input_x - r_in))
    perp_x = -np.sin(np.radians(angle)) * 0.25
    perp_y =  np.cos(np.radians(angle)) * 0.25
    ax.text(mid_x + perp_x, mid_y + perp_y, wlab,
            ha='center', va='center', fontsize=11, color=NAVY,
            rotation=angle, rotation_mode='anchor')

# Bias → neuron (dashed)
ax.annotate('', xy=(sum_x - r_sum * 0.65, sum_y - r_sum * 0.65),
            xytext=(input_x + r_in, bias_y),
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5,
                            linestyle='dashed'), zorder=3)
ax.text(3.0, 1.7, '$b$', ha='center', va='center', fontsize=12, color=GRAY)

# ═══════════════════════════════════════════════════════════════════════════════
# Draw step activation box
# ═══════════════════════════════════════════════════════════════════════════════
box = FancyBboxPatch((act_x - box_w/2, act_y - box_h/2), box_w, box_h,
                      boxstyle='round,pad=0.08', linewidth=2.0,
                      edgecolor=NAVY, facecolor=LITE, zorder=4)
ax.add_patch(box)

# Step function curve inside the box
sx = np.array([act_x - 0.58, act_x - 0.58, act_x + 0.08, act_x + 0.08, act_x + 0.58])
sy = np.array([act_y - 0.32, act_y + 0.05, act_y + 0.05, act_y + 0.32, act_y + 0.32])
# Vertical step
sx2 = np.array([act_x + 0.08, act_x + 0.08])
sy2 = np.array([act_y + 0.05, act_y + 0.32])
ax.plot(sx, sy, color=NAVY, lw=2.2, zorder=5)
ax.text(act_x, act_y - box_h/2 - 0.28, 'step activation',
        ha='center', va='top', fontsize=9.5, color=NAVY)

# ═══════════════════════════════════════════════════════════════════════════════
# Arrow: neuron → step box
# ═══════════════════════════════════════════════════════════════════════════════
ax.annotate('', xy=(act_x - box_w/2, act_y), xytext=(sum_x + r_sum, sum_y),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.8), zorder=3)

# Intermediate label (linear combination)
mid2_x = (sum_x + r_sum + act_x - box_w/2) / 2
ax.text(mid2_x, act_y + 0.35,
        r'$\mathbf{x}^\top\!\mathbf{w} + b$',
        ha='center', va='bottom', fontsize=10, color=NAVY)

# ═══════════════════════════════════════════════════════════════════════════════
# Arrow: step box → output
# ═══════════════════════════════════════════════════════════════════════════════
ax.annotate('', xy=(act_x + box_w/2 + 0.9, act_y),
            xytext=(act_x + box_w/2, act_y),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.8), zorder=3)
ax.text(act_x + box_w/2 + 1.0, act_y,
        r'$\hat{y} \in \{-1,\;+1\}$',
        ha='left', va='center', fontsize=12, color=NAVY, fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════════════
# Column labels
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(input_x, -0.3, 'Inputs', ha='center', va='top',
        fontsize=11, color=NAVY, fontweight='bold')
ax.text(sum_x,   -0.3, 'Neuron', ha='center', va='top',
        fontsize=11, color=NAVY, fontweight='bold')
ax.text(act_x,   -0.3, 'Activation', ha='center', va='top',
        fontsize=11, color=NAVY, fontweight='bold')

# ─── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout()
for out in (IMG, LOCAL):
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out}")
plt.close()
