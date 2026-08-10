#!/usr/bin/env python3
"""
LASSO vs. ridge constraint-region figure for Topic 2.3 (Complexity Optimization).

Replaces a web-sourced version of the classic ESL-style diagram with a
script-generated one. Two panels share the same least-squares problem
(elliptical RSS contours centered on the unconstrained optimum). The left
panel constrains the coefficients to the L1 ball (a diamond), the right panel
to the L2 ball (a circle). The smallest RSS contour that touches each region
is computed numerically, so the contact points are exact: the L1 contact lands
on the diamond's corner (driving beta_1 to exactly zero — sparsity), while the
L2 contact is tangency at a generic point (both coefficients shrink, neither
is zero).

The script checks all rendered text bounding boxes for pairwise overlap and
fails loudly if any two collide.

Saves: lasso_ridge_constraints.png
       (this directory and ../../2-regression/images/)

Run from settings/helper_scripts/:
    python plot_lasso_ridge_constraints.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
IMGDIR = HERE / '../../2-regression/images'
plt.style.use(HERE / '../plot_style.mplstyle')

NAVY, GOLD, TEAL = '#003057', '#EAAA00', '#4B8B9B'

# ─── RSS quadratic form (parameters chosen so the L1 contact is the corner) ──
theta = np.radians(20)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
M = R @ np.diag([1.0, 2.5]) @ R.T
center = np.array([1.0, 1.6])   # unconstrained least-squares optimum

def q(pts):
    d = pts - center
    return np.einsum('ij,jk,ik->i', d, M, d)

def ellipse_pts(level, n=400):
    """Points on the contour q(x) = level."""
    evals, evecs = np.linalg.eigh(M)
    a = np.linspace(0, 2*np.pi, n)
    unit = np.column_stack([np.cos(a), np.sin(a)])
    scaled = unit / np.sqrt(evals) * np.sqrt(level)
    return scaled @ evecs.T + center

# Contact points: minimum of q over each constraint boundary
t = np.linspace(0, 1, 2000)
diamond = np.vstack([np.column_stack([sx*t, sy*(1-t)])
                     for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]])
ang = np.linspace(0, 2*np.pi, 4000)
circle = np.column_stack([np.cos(ang), np.sin(ang)])

contact = {}
for name, boundary in [('lasso', diamond), ('ridge', circle)]:
    idx = np.argmin(q(boundary))
    contact[name] = (boundary[idx], q(boundary)[np.newaxis][0][idx]
                     if False else q(boundary)[idx])

assert np.allclose(contact['lasso'][0], [0, 1], atol=0.01), 'L1 contact not at corner'
assert min(abs(contact['ridge'][0])) > 0.2, 'L2 contact unexpectedly near an axis'

# ─── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
texts = []

for ax, name, title in [(axes[0], 'lasso', 'LASSO ($L_1$ penalty)'),
                        (axes[1], 'ridge', 'Ridge ($L_2$ penalty)')]:
    # Constraint region
    if name == 'lasso':
        ax.add_patch(Polygon([(1, 0), (0, 1), (-1, 0), (0, -1)], closed=True,
                             facecolor=GOLD, alpha=0.5, edgecolor=GOLD, lw=2))
    else:
        ax.add_patch(Circle((0, 0), 1.0, facecolor=GOLD, alpha=0.5,
                            edgecolor=GOLD, lw=2))
    # RSS contours: the touching contour plus two larger ones
    pt, level = contact[name]
    for lv, lw in [(level, 2.0), (level*2.2, 1.0), (level*3.8, 1.0)]:
        e = ellipse_pts(lv)
        ax.plot(e[:, 0], e[:, 1], color=NAVY, lw=lw, alpha=0.85)
    # Unconstrained optimum and constrained solution
    ax.plot(*center, 'o', color=NAVY, ms=7)
    texts.append(ax.text(center[0]+0.1, center[1]+0.1,
                         r'$\hat{w}$ (least squares)', fontsize=11, color=NAVY))
    ax.plot(*pt, 'o', color=TEAL, ms=9, zorder=5)
    # Axes through origin
    ax.axhline(0, color='0.6', lw=0.8, zorder=0)
    ax.axvline(0, color='0.6', lw=0.8, zorder=0)
    ax.set_title(title)
    ax.set_xlabel('$w_1$')
    ax.set_xlim(-1.7, 3.1)
    ax.set_ylim(-1.6, 3.0)
    ax.set_aspect('equal')

texts.append(axes[0].text(-0.08, 1.28, 'constrained solution:\n$w_1 = 0$ exactly',
                          fontsize=10, color=TEAL, ha='right'))
texts.append(axes[1].text(1.62, 0.75, 'constrained solution:\nboth shrink,\nneither is zero',
                          fontsize=10, color=TEAL, ha='left'))
texts.append(axes[0].text(0.0, -0.55, 'allowed\ncoefficients', fontsize=10,
                          ha='center', color='#7a6a1e'))
texts.append(axes[1].text(0.0, -0.55, 'allowed\ncoefficients', fontsize=10,
                          ha='center', color='#7a6a1e'))
axes[0].set_ylabel('$w_2$')

fig.tight_layout()

# ─── Overlap check ────────────────────────────────────────────────────────────
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
bboxes = [t_.get_window_extent(renderer=renderer) for t_ in texts]
overlaps = [(texts[i].get_text()[:30], texts[j].get_text()[:30])
            for i in range(len(bboxes)) for j in range(i + 1, len(bboxes))
            if bboxes[i].overlaps(bboxes[j])]
if overlaps:
    raise SystemExit(f'FAIL — overlapping text elements: {overlaps}')
print(f'Overlap check passed: {len(texts)} text elements, no collisions.')
print(f"L1 contact: {np.round(contact['lasso'][0], 3)}  "
      f"L2 contact: {np.round(contact['ridge'][0], 3)}")

for out in (HERE / 'lasso_ridge_constraints.png', IMGDIR / 'lasso_ridge_constraints.png'):
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')
