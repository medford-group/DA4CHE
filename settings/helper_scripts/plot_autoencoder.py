#!/usr/bin/env python3
"""
Autoencoder architecture schematic.

Pedagogical message:
  An autoencoder is a neural network trained to reconstruct its own input.
  The encoder f_θ compresses the d-dimensional input x to a k-dimensional
  latent vector z (the bottleneck); the decoder g_φ reconstructs x̂ from z.
  Training minimizes the reconstruction error ||x - x̂||².
  The bottleneck forces the network to learn a compact representation of the
  data without any labels.

Architecture shown (layer sizes):  6 → 4 → 3 → 2 → 3 → 4 → 6

Saves:
    autoencoder.png  (this directory and ../../5-exploratory_data_analysis/images/)

Run from settings/helper_scripts/:
    python plot_autoencoder.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE  = Path(__file__).parent
IMG   = HERE / '../../5-exploratory_data_analysis/images/autoencoder.png'
LOCAL = HERE / 'autoencoder.png'

plt.style.use(HERE / '../plot_style.mplstyle')

# ─── GT colour palette ────────────────────────────────────────────────────────
NAVY  = '#003057'
GOLD  = '#EAAA00'
TEAL  = '#4B8B9B'
WHITE = 'white'
RED   = '#B03A2E'
GOLD_DARK = '#7A5C00'   # for text on gold background

# ─── Architecture ─────────────────────────────────────────────────────────────
LAYER_X     = [0.9, 2.4, 3.9, 5.4, 6.9, 8.4, 9.9]
LAYER_SIZES = [6,   4,   3,   2,   3,   4,   6  ]
NODE_R      = 0.22
CENTER_Y    = 3.75
SPACING     = 0.72

def layer_ys(n):
    """y-positions for n nodes, centred at CENTER_Y."""
    start = CENTER_Y - (n - 1) * SPACING / 2
    return [start + i * SPACING for i in range(n)]

all_ys = [layer_ys(n) for n in LAYER_SIZES]

# ─── Canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7.4))
ax.set_xlim(-0.1, 11.2)
ax.set_ylim(-1.0, 7.2)
ax.axis('off')
fig.patch.set_facecolor('white')

# ─── Background bands ─────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (0.4, 0.8), 4.0, 5.9, boxstyle='round,pad=0.15',
    facecolor=NAVY, edgecolor='none', alpha=0.06, zorder=0))

ax.add_patch(FancyBboxPatch(
    (4.8, 2.0), 1.35, 3.5, boxstyle='round,pad=0.15',
    facecolor=GOLD, edgecolor=GOLD, linewidth=1.5, alpha=0.14, zorder=0))

ax.add_patch(FancyBboxPatch(
    (6.3, 0.8), 4.0, 5.9, boxstyle='round,pad=0.15',
    facecolor=TEAL, edgecolor='none', alpha=0.06, zorder=0))

# ─── Section title labels ──────────────────────────────────────────────────────
ax.text(2.4,  6.8, r'Encoder  $f_\theta$',
        ha='center', va='center', fontsize=13, fontweight='bold', color=NAVY)
ax.text(5.45, 6.8, 'Latent Space',
        ha='center', va='center', fontsize=12, fontweight='bold', color=GOLD_DARK)
ax.text(8.4,  6.8, r'Decoder  $g_\phi$',
        ha='center', va='center', fontsize=13, fontweight='bold', color=TEAL)

# ─── "compress" / "reconstruct" flow arrows ───────────────────────────────────
arr_y = 6.35
ax.annotate('', xy=(LAYER_X[3] - 0.05, arr_y), xytext=(LAYER_X[0] + 0.05, arr_y),
            arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.6), zorder=3)
ax.text((LAYER_X[0] + LAYER_X[3]) / 2, arr_y + 0.18,
        'compress', ha='center', va='bottom', fontsize=10, color=NAVY)

ax.annotate('', xy=(LAYER_X[6] - 0.05, arr_y), xytext=(LAYER_X[3] + 0.05, arr_y),
            arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.6), zorder=3)
ax.text((LAYER_X[3] + LAYER_X[6]) / 2, arr_y + 0.18,
        'reconstruct', ha='center', va='bottom', fontsize=10, color=TEAL)

# ─── Inter-layer connections (all-to-all, thin, semi-transparent) ─────────────
for li in range(len(LAYER_X) - 1):
    x0   = LAYER_X[li]     + NODE_R
    x1   = LAYER_X[li + 1] - NODE_R
    ys0  = all_ys[li]
    ys1  = all_ys[li + 1]
    col  = GOLD if li in (2, 3) else (NAVY if li < 3 else TEAL)
    for y0 in ys0:
        for y1 in ys1:
            ax.plot([x0, x1], [y0, y1], color=col, lw=0.55, alpha=0.20, zorder=1)

# ─── Directional arrowheads at the midpoint of each inter-layer gap ───────────
for li in range(len(LAYER_X) - 1):
    x_mid = (LAYER_X[li] + LAYER_X[li + 1]) / 2
    col   = GOLD if li in (2, 3) else (NAVY if li < 3 else TEAL)
    ax.annotate('', xy=(x_mid + 0.12, CENTER_Y), xytext=(x_mid - 0.12, CENTER_Y),
                arrowprops=dict(arrowstyle='->', color=col, lw=2.0), zorder=3)

# ─── Draw nodes ───────────────────────────────────────────────────────────────
for li, (lx, ln, ys) in enumerate(zip(LAYER_X, LAYER_SIZES, all_ys)):
    if li == 3:                        # bottleneck
        fcolor, ecolor, lw = GOLD, GOLD_DARK, 2.2
    elif li <= 2:                      # input + encoder hidden
        fcolor, ecolor, lw = NAVY, WHITE, 1.5
    else:                              # decoder hidden + output
        fcolor, ecolor, lw = TEAL, WHITE, 1.5

    for y in ys:
        ax.add_patch(Circle((lx, y), NODE_R,
                            facecolor=fcolor, edgecolor=ecolor,
                            linewidth=lw, zorder=4))

    # Bottleneck: label z₁, z₂ inside nodes
    if li == 3:
        for yi, y in enumerate(ys):
            ax.text(lx, y, f'$z_{yi + 1}$',
                    ha='center', va='center', fontsize=9,
                    color=NAVY, fontweight='bold', zorder=5)

    # Input / output: show x₁ in top node and ellipsis
    if li == 0:
        ax.text(lx, ys[-1], '$x_1$',
                ha='center', va='center', fontsize=8,
                color=WHITE, fontweight='bold', zorder=5)
        ax.text(lx, ys[-2], '$x_2$',
                ha='center', va='center', fontsize=8,
                color=WHITE, fontweight='bold', zorder=5)
        ax.text(lx, (ys[1] + ys[2]) / 2, r'$\vdots$',
                ha='center', va='center', fontsize=14, color=WHITE, zorder=5)
        ax.text(lx, ys[0], '$x_d$',
                ha='center', va='center', fontsize=8,
                color=WHITE, fontweight='bold', zorder=5)

    if li == 6:
        ax.text(lx, ys[-1], r'$\hat{x}_1$',
                ha='center', va='center', fontsize=8,
                color=WHITE, fontweight='bold', zorder=5)
        ax.text(lx, ys[-2], r'$\hat{x}_2$',
                ha='center', va='center', fontsize=8,
                color=WHITE, fontweight='bold', zorder=5)
        ax.text(lx, (ys[1] + ys[2]) / 2, r'$\vdots$',
                ha='center', va='center', fontsize=14, color=WHITE, zorder=5)
        ax.text(lx, ys[0], r'$\hat{x}_d$',
                ha='center', va='center', fontsize=8,
                color=WHITE, fontweight='bold', zorder=5)

# ─── Column labels below each boundary layer ──────────────────────────────────
ax.text(LAYER_X[0], 0.55, r'$\mathbf{x}$' + '\n($d$ features)',
        ha='center', va='top', fontsize=11, color=NAVY, fontweight='bold')
ax.text(LAYER_X[3], 0.55, r'$\mathbf{z}$' + '\n($k \ll d$)',
        ha='center', va='top', fontsize=11, color=GOLD_DARK, fontweight='bold')
ax.text(LAYER_X[6], 0.55, r'$\hat{\mathbf{x}}$' + '\n($d$ features)',
        ha='center', va='top', fontsize=11, color=TEAL, fontweight='bold')

# ─── Reconstruction error bracket ─────────────────────────────────────────────
err_y = -0.55
ax.annotate('', xy=(LAYER_X[6], err_y), xytext=(LAYER_X[0], err_y),
            arrowprops=dict(arrowstyle='<->', color=RED, lw=1.8))
ax.text((LAYER_X[0] + LAYER_X[6]) / 2, err_y + 0.18,
        r'minimize $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$',
        ha='center', va='bottom', fontsize=11, color=RED, style='italic')

# ─── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout()
for out in (IMG, LOCAL):
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {out}')
plt.close()
