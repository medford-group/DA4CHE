#!/usr/bin/env python3
"""
RESTful URL anatomy figure for the Online Data Access chapter (Topic 4.2).

Renders the PubChem PUG REST example URL used in the chapter,

    https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/ethanol/cids/TXT

as four colored path segments — prolog (base endpoint), input (resource
specifier), operation, and output format — with leader lines down to a
label for each component. The segment widths are proportional to their
character counts so the URL reads as one continuous string.

The script checks all rendered text bounding boxes for pairwise overlap
and fails loudly if any two collide.

Saves: restful_url_anatomy.png
       (this directory and ../../4-data_management/images/)

Run from settings/helper_scripts/:
    python plot_restful_url.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
IMGS = HERE / '../../4-data_management/images'
plt.style.use(HERE / '../plot_style.mplstyle')

# ─── Content ──────────────────────────────────────────────────────────────────
SEGMENTS = [
    # text, component, descriptor, example meaning, color, text color
    ('https://pubchem.ncbi.nlm.nih.gov/rest/pug', 'Prolog', 'base endpoint',
     'where the API lives', '#003057', 'white'),
    ('/compound/name/ethanol', 'Input', 'resource specifier',
     'compound named "ethanol"', '#EAAA00', '#003057'),
    ('/cids', 'Operation', 'what to return',
     'compound IDs', '#377117', 'white'),
    ('/TXT', 'Output', 'response format',
     'plain text', '#4B8B9B', 'white'),
]

# ─── Layout (axis coordinates) ────────────────────────────────────────────────
X0, X1 = 0.02, 0.98
n_chars = sum(len(s[0]) for s in SEGMENTS)
char_w = (X1 - X0) / n_chars

URL_Y = 0.80          # vertical center of the URL boxes
BOX_H = 0.17          # box height
LEAD_TOP = URL_Y - BOX_H / 2 - 0.02
LEAD_BOT = 0.47       # where leader lines end
LABEL_Y = 0.42        # top line of each label block
LABEL_XS = [0.125, 0.375, 0.625, 0.875]   # evenly spaced label anchors

fig, ax = plt.subplots(figsize=(13, 3.4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

texts = []
x = X0
for (seg, name, desc, meaning, color, tcolor), lx in zip(SEGMENTS, LABEL_XS):
    w = len(seg) * char_w
    # Segment box
    ax.add_patch(FancyBboxPatch(
        (x + 0.0015, URL_Y - BOX_H / 2), w - 0.003, BOX_H,
        boxstyle='round,pad=0.004,rounding_size=0.008',
        linewidth=0, facecolor=color, mutation_aspect=13 / 3.4))
    # Segment text, centered in its box
    texts.append(ax.text(x + w / 2, URL_Y, seg, ha='center', va='center',
                         family='monospace', fontsize=13.5, color=tcolor))
    # Leader line from segment midpoint to label anchor
    ax.plot([x + w / 2, lx], [LEAD_TOP, LEAD_BOT], color=color,
            linewidth=1.4, clip_on=False)
    # Label block: bold name, descriptor, example meaning
    texts.append(ax.text(lx, LABEL_Y, name, ha='center', va='top',
                         fontsize=13, fontweight='bold', color=color))
    texts.append(ax.text(lx, LABEL_Y - 0.13, f'({desc})', ha='center', va='top',
                         fontsize=11.5, color='#333333'))
    texts.append(ax.text(lx, LABEL_Y - 0.26, meaning, ha='center', va='top',
                         fontsize=10.5, style='italic', color='#777777'))
    x += w

fig.tight_layout()

# ─── Overlap check: no two text bounding boxes may intersect ──────────────────
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
bboxes = [t.get_window_extent(renderer=renderer) for t in texts]
overlaps = [(texts[i].get_text(), texts[j].get_text())
            for i in range(len(bboxes)) for j in range(i + 1, len(bboxes))
            if bboxes[i].overlaps(bboxes[j])]
if overlaps:
    raise SystemExit(f'FAIL — overlapping text elements: {overlaps}')
print(f'Overlap check passed: {len(texts)} text elements, no collisions.')

for out in (HERE / 'restful_url_anatomy.png', IMGS / 'restful_url_anatomy.png'):
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')
