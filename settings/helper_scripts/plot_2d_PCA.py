#!/usr/bin/env python3
"""
PCA two-panel figure:
- Left: original data with principal axes drawn from the mean
- Right: rotated data (PC scores), with explained variance in the title
Saves: pca_two_panel.png

Created with signicant assistance from ChatGPT 5
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Configuration ----------------
STYLE_PATH = Path("../plot_style.mplstyle")   
FIGSIZE = (8, 3.5)
OUTFILE = Path("pca_illustration.png")
SEED = 7
NPTS = 150
# ------------------------------------------------

def main():
    # Use the uploaded/attached mplstyle if available
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    else:
        print(f"[warn] Style file not found at {STYLE_PATH.resolve()}. Using Matplotlib defaults.")

    # --- Data (seeded for reproducibility) ---
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 3, NPTS)
    y = x - 15 + 2 - 3 * rng.random(NPTS)
    X = np.column_stack([x, y])

    # --- PCA via covariance eigendecomposition ---
    mu = X.mean(axis=0)
    Xc = X - mu
    C = np.cov(Xc, rowvar=False)           # covariance matrix
    w, V = np.linalg.eigh(C)               # eigenvalues (asc), eigenvectors
    idx = np.argsort(w)[::-1]              # sort by descending variance
    w, V = w[idx], V[:, idx]
    Z = Xc @ V                              # rotated data (PC scores)
    exp_var = 100 * w / w.sum()

    # --- Figure with two axes (horizontal) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)

    # Left: original data + principal axes
    ax1.plot(X[:, 0], X[:, 1], '.', alpha=0.7)
    ax1.plot(mu[0], mu[1], '+', ms=10, mew=2)  # mean marker

    # Draw arrows along PCs, length ~ a few std dev for visibility
    for j in range(2):
        vec = V[:, j]
        length = 2.5 * np.sqrt(w[j])
        ax1.quiver(mu[0], mu[1],
                   vec[0]*length, vec[1]*length,
                   angles='xy', scale_units='xy', scale=1,
                   width=0.004)
        if j == 0:
            ax1.text(mu[0] + 1.05*vec[0]*length,
                     mu[1] + 1.05*vec[1]*length,
                     f'PC{j+1}',fontsize=10)
        if j > 0:
            ax1.text(mu[0] + 1.75*vec[0]*length,
                     mu[1] + 1.75*vec[1]*length,
                     f'PC{j+1}',fontsize=10)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Original data + principal axes')
    ax1.axis('equal')

    # Right: rotated data (PC scores)
    ax2.plot(Z[:, 0], Z[:, 1], '.', alpha=0.7)
    ax2.axhline(0, lw=1)
    ax2.axvline(0, lw=1)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title(f'Rotated data (PC1 {exp_var[0]:.1f}%, PC2 {exp_var[1]:.1f}%)')
    ax2.axis('equal')

    # Save
    fig.savefig(OUTFILE, bbox_inches='tight')
    print(f"[ok] Saved {OUTFILE.resolve()}")

if __name__ == "__main__":
    main()

