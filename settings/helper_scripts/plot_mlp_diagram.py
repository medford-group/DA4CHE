#!/usr/bin/env python3
"""
Standard neural-network diagram for the MLP "Stacking Layers" section (Topic 6.3).

Draws the chapter's two-hidden-layer example — 3 inputs, hidden layers of 4 and 3
neurons, 1 output — as the conventional circles-and-arrows diagram, and connects it
to the math: each column is labeled with the corresponding equation, and each bundle
of edges is labeled with the weight matrix it represents, so the equations read as
the diagram traversed left to right.

The script checks all rendered text bounding boxes for pairwise overlap and fails
loudly if any two collide.

Saves: mlp_diagram.png
       (this directory and ../../6-advanced_topics/images/)

Run from settings/helper_scripts/:
    python plot_mlp_diagram.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
IMGDIR = HERE / '../../6-advanced_topics/images'
IMGDIR.mkdir(exist_ok=True)
plt.style.use(HERE / '../plot_style.mplstyle')

# ─── Layout ───────────────────────────────────────────────────────────────────
LAYERS = [3, 4, 3, 1]                       # input, hidden 1, hidden 2, output
XS = [0.0, 2.2, 4.4, 6.6]                   # column x-positions
COLORS = ['#003057', '#EAAA00', '#4B8B9B', '#377117']
NODE_R = 0.28
YSPACE = 1.0

LAYER_LABELS = [
    'inputs\n$\\mathbf{x}$',
    'hidden layer 1\n$\\mathbf{h}^{(1)} = \\sigma(W^{(1)}\\mathbf{x} + \\mathbf{b}^{(1)})$',
    'hidden layer 2\n$\\mathbf{h}^{(2)} = \\sigma(W^{(2)}\\mathbf{h}^{(1)} + \\mathbf{b}^{(2)})$',
    'output\n$\\hat{y} = W^{(3)}\\mathbf{h}^{(2)} + b^{(3)}$',
]
EDGE_LABELS = ['$W^{(1)}$', '$W^{(2)}$', '$W^{(3)}$']

def node_ys(n):
    return (np.arange(n) - (n - 1) / 2) * YSPACE

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(-1.0, 7.6)
ax.set_ylim(-2.9, 2.4)
ax.axis('off')
ax.set_aspect('equal')

# Edges first (behind nodes), one bundle label per layer pair
for li in range(len(LAYERS) - 1):
    ys_a, ys_b = node_ys(LAYERS[li]), node_ys(LAYERS[li + 1])
    for ya in ys_a:
        for yb in ys_b:
            ax.plot([XS[li], XS[li + 1]], [ya, yb], color='0.8',
                    linewidth=0.9, zorder=1)
    top = max(ys_a.max(), ys_b.max())

texts = []
for li in range(len(LAYERS) - 1):
    texts.append(ax.text((XS[li] + XS[li + 1]) / 2, 2.15, EDGE_LABELS[li],
                         ha='center', va='center', fontsize=13, color='#333333'))

# Nodes
input_labels = ['$x_1$', '$x_2$', '$x_3$']
for li, (n, x, color) in enumerate(zip(LAYERS, XS, COLORS)):
    for ni, ypos in enumerate(node_ys(n)):
        ax.add_patch(Circle((x, ypos), NODE_R, facecolor=color,
                            edgecolor='black', linewidth=1.0, zorder=3))
        if li == 0:
            texts.append(ax.text(x, ypos, input_labels[n - 1 - ni], ha='center',
                                 va='center', fontsize=11, color='white', zorder=4))
        if li == len(LAYERS) - 1:
            texts.append(ax.text(x, ypos, '$\\hat{y}$', ha='center',
                                 va='center', fontsize=12, color='white', zorder=4))

# Column labels with their equations
for x, label in zip(XS, LAYER_LABELS):
    texts.append(ax.text(x, -2.35, label, ha='center', va='center', fontsize=11))

# One annotated neuron: what every circle computes
ys_h1 = node_ys(LAYERS[1])
ann = ax.annotate('each neuron:\nweighted sum, then $\\sigma$',
                  xy=(XS[1] - NODE_R * 0.8, ys_h1[-1] + NODE_R * 0.6),
                  xytext=(-1.0, 1.45),
                  fontsize=10, color='#555555', ha='left',
                  arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0))
texts.append(ann)

fig.tight_layout()

# ─── Overlap check ────────────────────────────────────────────────────────────
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
bboxes = [t.get_window_extent(renderer=renderer) for t in texts]
overlaps = [(texts[i].get_text(), texts[j].get_text())
            for i in range(len(bboxes)) for j in range(i + 1, len(bboxes))
            if bboxes[i].overlaps(bboxes[j])]
if overlaps:
    raise SystemExit(f'FAIL — overlapping text elements: {overlaps}')
print(f'Overlap check passed: {len(texts)} text elements, no collisions.')

for out in (HERE / 'mlp_diagram.png', IMGDIR / 'mlp_diagram.png'):
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')
