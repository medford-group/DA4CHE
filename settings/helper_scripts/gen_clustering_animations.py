"""Generate copyright-safe, size-optimized clustering animations + a toy dendrogram
for Topic5.3 (Clustering). All data is synthetic/illustrative.

Usage: python gen_clustering_animations.py <output_dir>
"""
import io
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
CLRS = np.array(['#003057', '#EAAA00', '#4B8B9B', '#B3A369', '#377117', '#1879DB'])


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=70, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames, path, duration=160):
    # quantize all frames to a single 64-color palette for a small, flicker-free GIF
    pal = frames[0].convert("P", palette=Image.ADAPTIVE, colors=64)
    q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:],
              duration=duration, loop=0, optimize=True)
    import os
    print(f"  {os.path.basename(path)}: {os.path.getsize(path)/1024:.0f} KB, {len(frames)} frames")


def cov_ellipse(ax, mean, cov, color):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * 2 * np.sqrt(np.maximum(vals, 1e-9))   # 2-sigma
    e = Ellipse(mean, w, h, angle=angle, fc=color, ec=color, alpha=0.25, lw=2)
    ax.add_patch(e)


# ── 1. k-means convergence ──────────────────────────────────────────────────
def kmeans_gif():
    X, ylab = make_blobs(n_samples=300, centers=3, cluster_std=0.9, random_state=7)
    rng = np.random.default_rng(0)
    pool = X[ylab == 0]                                   # seed all 3 centroids inside one blob
    centers = pool[rng.choice(len(pool), 3, replace=False)].astype(float)  # they must spread out; no orphans
    frames = []
    for it in range(11):
        d = np.linalg.norm(X[:, None] - centers[None], axis=2)
        lab = d.argmin(1)
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.scatter(X[:, 0], X[:, 1], c=CLRS[lab], s=14, alpha=0.6)
        ax.scatter(centers[:, 0], centers[:, 1], c='k', marker='*', s=200,
                   edgecolors='w', zorder=5)
        ax.set_title(f'k-means — iteration {it}'); ax.set_xticks([]); ax.set_yticks([])
        frames.append(fig_to_img(fig))
        new = np.array([X[lab == k].mean(0) if (lab == k).any() else centers[k]
                        for k in range(3)])
        if np.allclose(new, centers):
            frames.append(frames[-1]); break
        centers = new
    save_gif(frames, f"{OUT}/kmeans.gif")


# ── 2. GMM expectation-maximization ─────────────────────────────────────────
def gmm_gif():
    from sklearn.mixture import GaussianMixture
    centers = np.array([[-4.0, -4.0], [4.5, -1.0], [0.0, 5.0]])   # well-separated triangle
    X, _, true_c = make_blobs(n_samples=300, centers=centers, cluster_std=1.0,
                              random_state=11, return_centers=True)
    rng = np.random.default_rng(0)
    means_init = true_c + rng.normal(0, 2.0, true_c.shape)        # start near but off
    precisions_init = np.array([np.eye(2) * 0.04] * 3)           # broad initial ellipses (var~25)
    gm = GaussianMixture(3, covariance_type='full', warm_start=True, max_iter=1,
                         means_init=means_init, precisions_init=precisions_init,
                         random_state=0, tol=1e-9)
    frames = []
    for it in range(14):
        gm.fit(X)
        lab = gm.predict(X)
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.scatter(X[:, 0], X[:, 1], c=CLRS[lab], s=14, alpha=0.6)
        for k in range(3):
            cov_ellipse(ax, gm.means_[k], gm.covariances_[k], CLRS[k])
            ax.plot(*gm.means_[k], 'k*', ms=12, mec='w')
        ax.set_title(f'GMM (EM) — iteration {it + 1}'); ax.set_xticks([]); ax.set_yticks([])
        frames.append(fig_to_img(fig))
    save_gif(frames, f"{OUT}/GMM.gif")


# ── 3. mean-shift convergence ───────────────────────────────────────────────
def meanshift_gif():
    centers = np.array([[-4.0, -4.0], [4.5, -1.0], [0.0, 5.0]])   # match GMM: 3 separated blobs
    X, _ = make_blobs(n_samples=250, centers=centers, cluster_std=0.8, random_state=5)
    r = 2.5
    seeds = X[::8].copy().astype(float)   # a subset of points as moving centroids
    frames = []
    for it in range(12):
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.scatter(X[:, 0], X[:, 1], c='0.8', s=12)
        ax.scatter(seeds[:, 0], seeds[:, 1], c=CLRS[1], s=40, edgecolors='k', zorder=5)
        ax.set_title(f'mean shift — iteration {it}'); ax.set_xticks([]); ax.set_yticks([])
        frames.append(fig_to_img(fig))
        new = []
        for c in seeds:
            near = X[np.linalg.norm(X - c, axis=1) <= r]
            new.append(near.mean(0) if len(near) else c)
        new = np.array(new)
        if np.allclose(new, seeds, atol=1e-3):
            frames.append(frames[-1]); break
        seeds = new
    save_gif(frames, f"{OUT}/meanshift.gif")


# ── 4. DBSCAN (moons: arbitrary shapes + noise) ─────────────────────────────
def dbscan_gif():
    X, _ = make_moons(n_samples=260, noise=0.07, random_state=0)
    rng = np.random.default_rng(1)
    X = np.vstack([X, rng.uniform([-1.5, -1], [2.5, 1.5], size=(15, 2))])  # add noise pts
    lab = DBSCAN(eps=0.25, min_samples=5).fit_predict(X)
    n_cl = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"    (DBSCAN: {n_cl} clusters, {(lab == -1).sum()} noise)")
    # reveal points in label order to mimic cluster growth
    order = np.argsort(lab + (lab == -1) * 1000)   # clusters first, noise last
    frames = []
    steps = np.linspace(20, len(X), 18).astype(int)
    for s in steps:
        shown = order[:s]
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.scatter(X[:, 0], X[:, 1], c='0.9', s=10)
        c = np.array(['#cccccc' if lab[i] == -1 else CLRS[lab[i] % len(CLRS)] for i in shown])
        ax.scatter(X[shown, 0], X[shown, 1], c=c, s=16)
        ax.set_title('DBSCAN — clusters grow, noise stays gray')
        ax.set_xticks([]); ax.set_yticks([])
        frames.append(fig_to_img(fig))
    frames += [frames[-1]] * 3
    save_gif(frames, f"{OUT}/DBSCAN.gif")


# ── 5. agglomerative merging ────────────────────────────────────────────────
def agglomerative_gif():
    X, _ = make_blobs(n_samples=14, centers=3, cluster_std=0.7, random_state=2)
    Z = linkage(X, method='single')
    n = len(X)
    members = {i: [i] for i in range(n)}
    clusters = {i: i for i in range(n)}   # point -> current cluster id
    frames = []
    def draw(active_pair=None):
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ids = sorted(set(clusters.values()))
        cmap = {cid: CLRS[j % len(CLRS)] for j, cid in enumerate(ids)}
        ax.scatter(X[:, 0], X[:, 1], c=[cmap[clusters[i]] for i in range(n)],
                   s=60, edgecolors='k', zorder=3)
        if active_pair:
            for a in members[active_pair[0]]:
                for b in members[active_pair[1]]:
                    ax.plot(*zip(X[a], X[b]), color='0.5', lw=0.6, zorder=1)
        ax.set_title(f'agglomerative — {len(ids)} clusters')
        ax.set_xticks([]); ax.set_yticks([])
        frames.append(fig_to_img(fig))
    draw()
    nid = n
    for a, b, _, _ in Z:
        a, b = int(a), int(b)
        draw(active_pair=(a, b))
        members[nid] = members[a] + members[b]
        for p in members[nid]:
            clusters[p] = nid
        nid += 1
        draw()
    frames += [frames[-1]] * 3
    save_gif(frames, f"{OUT}/agglomerative.gif", duration=400)


# ── 6. toy species dendrogram (static) ──────────────────────────────────────
def bio_dendrogram():
    species = ['Human', 'Chimp', 'Gorilla', 'Orangutan', 'Mouse', 'Rat', 'Chicken', 'Frog']
    # illustrative feature vectors (not real genetics) producing a sensible tree
    feats = np.array([
        [0.0, 0.0, 0.0], [0.3, 0.1, 0.0], [0.6, 0.3, 0.1], [1.2, 0.6, 0.2],
        [3.0, 3.0, 0.5], [3.1, 3.2, 0.6], [6.0, 5.0, 4.0], [7.0, 6.5, 6.0]])
    Z = linkage(feats, method='average')
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    dendrogram(Z, labels=species, ax=ax, color_threshold=2.5,
               leaf_rotation=30, leaf_font_size=9)
    ax.set_ylabel('Distance')
    ax.set_title('Illustrative species dendrogram')
    fig.tight_layout()
    fig.savefig(f"{OUT}/bio_dendrogram.png", dpi=110)
    plt.close(fig)
    import os
    print(f"  bio_dendrogram.png: {os.path.getsize(f'{OUT}/bio_dendrogram.png')/1024:.0f} KB")


if __name__ == "__main__":
    print("Generating clustering animations ->", OUT)
    kmeans_gif()
    gmm_gif()
    meanshift_gif()
    dbscan_gif()
    agglomerative_gif()
    bio_dendrogram()
    print("done.")
