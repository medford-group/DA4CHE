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


def save_gif(frames, path, duration=550, end_hold=2000):
    # Build ONE palette from ALL frames stacked together, so colors that only appear
    # in later frames (e.g. a cluster revealed mid-animation) are not lost. Quantizing
    # against a palette derived from frame 0 alone would map such colors to the nearest
    # available entry (often gray) — a subtle but real bug.
    w, h = frames[0].size
    montage = Image.new("RGB", (w, h * len(frames)))
    for i, f in enumerate(frames):
        montage.paste(f, (0, i * h))
    pal = montage.convert("P", palette=Image.ADAPTIVE, colors=128)
    q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    # slow base frame rate so students can follow; hold the final frame longer
    durations = [duration] * (len(q) - 1) + [end_hold]
    q[0].save(path, save_all=True, append_images=q[1:],
              duration=durations, loop=0, optimize=True)
    import os
    print(f"  {os.path.basename(path)}: {os.path.getsize(path)/1024:.0f} KB, {len(frames)} frames")


def cov_ellipse(ax, mean, cov, color, nsigs=(1, 2)):
    """Draw 1- and 2-standard-deviation ellipses for a Gaussian component."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    for ns in nsigs:
        w, h = 2 * ns * np.sqrt(np.maximum(vals, 1e-9))   # ns-sigma
        ax.add_patch(Ellipse(mean, w, h, angle=angle,
                             fc=color, ec=color, alpha=0.16, lw=1.5))


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
    save_gif(frames, f"{OUT}/kmeans_animation.gif")


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
    # fixed axis limits so the data stays put and only the ellipses move
    xlim = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    ylim = (X[:, 1].min() - 1, X[:, 1].max() + 1)
    frames = []
    for it in range(10):
        gm.fit(X)
        lab = gm.predict(X)
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.scatter(X[:, 0], X[:, 1], c=CLRS[lab], s=14, alpha=0.6)
        for k in range(3):
            cov_ellipse(ax, gm.means_[k], gm.covariances_[k], CLRS[k])   # 1- and 2-sigma
            ax.plot(*gm.means_[k], 'k*', ms=12, mec='w')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(f'GMM (EM) — iteration {it + 1}'); ax.set_xticks([]); ax.set_yticks([])
        frames.append(fig_to_img(fig))
    save_gif(frames, f"{OUT}/gmm_em_animation.gif")


# ── 3. mean-shift convergence ───────────────────────────────────────────────
def meanshift_gif():
    centers = np.array([[-3.0, -2.0], [3.0, -2.0], [0.0, 3.2]])   # blobs with overlapping edges
    X, ylab = make_blobs(n_samples=260, centers=centers, cluster_std=1.1, random_state=5)
    r = 2.6
    seeds = X[::8].copy().astype(float)   # a subset of points as moving centroids
    frames = []
    for it in range(15):
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        # faint true groups so the overlap between clusters is visible
        ax.scatter(X[:, 0], X[:, 1], c=CLRS[ylab], s=12, alpha=0.25)
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
    save_gif(frames, f"{OUT}/meanshift_animation.gif")


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
        # bright, clearly non-gray colors for the clusters; noise stays gray
        cluster_palette = ['#EAAA00', '#1879DB', '#377117']
        c = np.array(['#bdbdbd' if lab[i] == -1 else cluster_palette[lab[i] % len(cluster_palette)]
                      for i in shown])
        ax.scatter(X[shown, 0], X[shown, 1], c=c, s=16)
        ax.set_title('DBSCAN — clusters grow, noise stays gray')
        ax.set_xticks([]); ax.set_yticks([])
        frames.append(fig_to_img(fig))
    frames += [frames[-1]] * 3
    save_gif(frames, f"{OUT}/dbscan_animation.gif", duration=320)


# ── 5. agglomerative merging ────────────────────────────────────────────────
def agglomerative_gif():
    X, _ = make_blobs(n_samples=14, centers=3, cluster_std=0.7, random_state=2)
    Z = linkage(X, method='single')
    n = len(X)
    # dendrogram link coordinates, ordered by merge height (== Z order)
    dn = dendrogram(Z, no_plot=True)
    ic, dc = dn['icoord'], dn['dcoord']
    order = sorted(range(len(ic)), key=lambda i: max(dc[i]))
    heights = [max(dc[order[j]]) for j in range(len(order))]
    xmax = max(max(x) for x in ic) + 5
    hmax = max(heights) * 1.08
    xlim = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    ylim = (X[:, 1].min() - 1, X[:, 1].max() + 1)

    members = {i: [i] for i in range(n)}
    clusters = {i: i for i in range(n)}
    frames = []

    def frame(step, active=None):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.7))
        # left: scatter colored by current clusters; highlight the newest merge
        ids = sorted(set(clusters.values()))
        cmap = {cid: CLRS[j % len(CLRS)] for j, cid in enumerate(ids)}
        axL.scatter(X[:, 0], X[:, 1], c=[cmap[clusters[i]] for i in range(n)],
                    s=55, edgecolors='k', zorder=3)
        if active is not None:
            ca = X[members[active[0]]].mean(0)
            cb = X[members[active[1]]].mean(0)
            axL.plot([ca[0], cb[0]], [ca[1], cb[1]], color=CLRS[1], lw=1.8, zorder=1)
        axL.set_xlim(*xlim); axL.set_ylim(*ylim)
        axL.set_title(f'{len(ids)} cluster(s)'); axL.set_xticks([]); axL.set_yticks([])
        # right: dendrogram links revealed so far; newest highlighted; bar at current height
        for j in range(step):
            li = order[j]
            axR.plot(ic[li], dc[li], color='0.35', lw=1.2)
        if step > 0:
            li = order[step - 1]
            axR.plot(ic[li], dc[li], color=CLRS[1], lw=2.6)
            axR.axhline(heights[step - 1], color=CLRS[1], ls='--', lw=1.3)
        axR.set_xlim(0, xmax); axR.set_ylim(0, hmax)
        axR.set_xticks([]); axR.set_ylabel('Merge distance'); axR.set_title('Dendrogram')
        plt.tight_layout()
        frames.append(fig_to_img(fig))

    frame(0)
    nid = n
    for i in range(len(Z)):
        a, b = int(Z[i, 0]), int(Z[i, 1])
        members[nid] = members[a] + members[b]
        for p in members[nid]:
            clusters[p] = nid
        frame(i + 1, active=(a, b))
        nid += 1
    frames += [frames[-1]] * 3
    save_gif(frames, f"{OUT}/agglomerative_animation.gif", duration=420)


# ── 6. toy species dendrogram (static) ──────────────────────────────────────
def bio_dendrogram():
    species = ['Human', 'Chimp', 'Gorilla', 'Orangutan', 'Mouse', 'Rat', 'Chicken', 'Frog']
    # illustrative feature vectors (not real genetics) producing a sensible tree
    feats = np.array([
        [0.0, 0.0, 0.0], [0.3, 0.1, 0.0], [0.6, 0.3, 0.1], [1.2, 0.6, 0.2],
        [3.0, 3.0, 0.5], [3.1, 3.2, 0.6], [6.0, 5.0, 4.0], [7.0, 6.5, 6.0]])
    Z = linkage(feats, method='average')
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    dendrogram(Z, labels=species, ax=ax, color_threshold=0,
               above_threshold_color='#003057',
               leaf_rotation=30, leaf_font_size=9)
    ax.set_ylabel('Distance')
    ax.set_title('Illustrative taxonomy: cutting higher up yields coarser groups')

    # taxonomic-rank cut lines — a higher horizontal cut merges finer groups into coarser ranks
    x0, x1 = ax.get_xlim()
    ranks = [(0.8, 'Family'), (1.8, 'Order'), (3.2, 'Class'), (6.0, 'Phylum')]
    for h, name in ranks:
        ax.axhline(h, color='0.55', ls='--', lw=0.9)
        ax.text(x1, h, f'  {name}', va='center', ha='left', fontsize=8, color='0.3')
    ax.text(x1, 8.7, '  Kingdom', va='center', ha='left', fontsize=8, color='0.3')
    ax.set_xlim(x0, x1 + (x1 - x0) * 0.16)
    ax.set_ylim(0, 9.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/species_dendrogram.png", dpi=110)
    plt.close(fig)
    import os
    print(f"  species_dendrogram.png: {os.path.getsize(f'{OUT}/species_dendrogram.png')/1024:.0f} KB")


if __name__ == "__main__":
    print("Generating clustering animations ->", OUT)
    kmeans_gif()
    gmm_gif()
    meanshift_gif()
    dbscan_gif()
    agglomerative_gif()
    bio_dendrogram()
    print("done.")
