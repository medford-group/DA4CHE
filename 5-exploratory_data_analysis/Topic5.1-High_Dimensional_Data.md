---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# High-Dimensional Data

:::{admonition} Learning Objectives
:class: tip

By the end of this chapter, you will be able to:
- Describe the curse and blessing of dimensionality and their practical implications
- Compute and interpret summary statistics (mean, standard deviation, max) across a high-dimensional feature matrix
- Construct histogram grids and scatter plot matrices to explore feature distributions and pairwise relationships
- Build and interpret correlation heatmaps for both tabular process data and image-derived features
- Identify zero-variance and highly correlated features as candidates for removal
:::

Working with data that has many features — more than two or three — presents challenges that do not arise in low-dimensional settings. We live in a three-dimensional world and have strong geometric intuition there, but in data science the number of dimensions equals the number of features, and that number can reach into the thousands or more. This chapter introduces the conceptual and practical tools needed to make sense of high-dimensional datasets before applying any model.

Throughout this chapter we use two contrasting datasets. The **Dow dataset** has 40 continuous process variables from a chemical distillation column — a typical scale for industrial sensor data. The **MNIST dataset** encodes each hand-written digit image as a 64-dimensional pixel vector — a simple example of structured high-dimensional data where every feature has an identical type and scale. Comparing strategies across both datasets builds transferable intuition.

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('../settings/plot_style.mplstyle')
clrs = np.array([p['color'] for p in plt.rcParams['axes.prop_cycle']])
```

## High-Dimensional Dataset Examples

### The Dow Dataset

We load and clean the Dow impurity dataset using the same procedure as Module 4. The function `is_real_and_finite` filters out rows that contain non-numeric or infinite values, which can appear in raw industrial time-series data.

```{code-cell} ipython3
df = pd.read_excel('data/impurity_dataset-training.xlsx')

def is_real_and_finite(x):
    return np.isreal(x) and np.isfinite(x)

# Flag rows where every non-Date column is a finite real number
numeric_map = df[df.columns[1:]].apply(lambda col: col.map(is_real_and_finite))
real_rows = numeric_map.all(axis=1).values

all_data = df[df.columns[1:]].values
X_dow = np.array(all_data[real_rows, :-5], dtype='float')   # 40 input features
y_dow = np.array(all_data[real_rows, -3],  dtype='float').reshape(-1, 1)
df_dow_clean = df[real_rows]

print(f'Dow dataset:  {X_dow.shape[0]} samples × {X_dow.shape[1]} features')
```

The Dow dataset has roughly 10,000 time-stamped observations and 40 sensor variables. The last five columns of the raw spreadsheet are derived outputs rather than inputs, so they are excluded from `X_dow`.

### The MNIST Dataset

The MNIST dataset is one of the most widely used benchmark datasets in machine learning. The variant built into scikit-learn contains 1,797 images of hand-written digits 0–9, each represented as an 8×8 grid of pixel intensities in the range [0, 16]. Flattening each image produces a 64-dimensional feature vector.

```{code-cell} ipython3
from sklearn.datasets import load_digits

digits = load_digits()
X_mnist = np.array(digits.data)
y_mnist = np.array(digits.target)

print(f'MNIST dataset: {X_mnist.shape[0]} samples × {X_mnist.shape[1]} features')
```

```{code-cell} ipython3
def show_image(data, n, ax=None, title=None):
    """Display the n-th row of data as an 8×8 grayscale image."""
    if ax is None:
        fig, ax = plt.subplots()
    img = data[n].reshape(8, 8)
    ax.imshow(img, cmap='binary', vmin=0, vmax=16)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    show_image(X_mnist, i, ax=ax, title=f'Digit: {y_mnist[i]}')
plt.suptitle('Sample MNIST digits', y=1.02)
plt.tight_layout()
```

Each row of `X_mnist` is a 64-element vector. The `show_image` helper reshapes it back to 8×8 for display. Although the data is stored as a flat vector for modeling purposes, the spatial arrangement of pixels carries meaning — a point we will return to in the Feature Engineering module.

:::{exercise}
:label: ex-eda-dataset-shapes

Compute the **samples-to-features ratio** (n/d) for both the Dow and MNIST datasets. If a linear model has roughly one free parameter per feature (plus an intercept), what does a low n/d ratio imply about the risk of overfitting? Which dataset is more at risk?
:::

## The Curse and the Blessing of Dimensionality

Dimensionality in data science is not the same as the three physical dimensions we navigate daily. In data science, dimensionality equals the number of features, and it can reach thousands or more. Two complementary phenomena govern what happens as dimensionality grows.

**The curse of dimensionality** arises because the volume of a high-dimensional space grows exponentially with the number of dimensions. Consider a $d$-dimensional hypercube of side length $L$: its volume is $V_d = L^d$. To sample this space uniformly at resolution $\Delta L = L/N$ requires $N^d$ grid points. For a moderate $N = 10$ and $d = 100$ (common in process data), that is $10^{100}$ points — far more than the number of atoms in the observable universe. In practice this means that any finite dataset becomes exponentially sparse as $d$ grows, making distance-based methods and density estimation unreliable.

**The blessing of dimensionality** is a lesser-known counterpart. Because data becomes sparse in high dimensions, points that might overlap in low dimensions tend to become well-separated as dimensions increase. This can make classification easier: high-dimensional data is often more linearly separable than the same data projected to a lower-dimensional space. Additionally, individual features tend to look more Gaussian in high dimensions due to averaging effects.

The curse always applies; the blessing is not guaranteed. In general, adding uninformative features makes problems harder.

The following plots illustrate both effects using two-class data in 2D and 3D:

```{code-cell} ipython3
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs

rng = np.random.default_rng(3)
X2, y2 = make_blobs(100, 2, centers=2, cluster_std=3.0, random_state=3)
z = rng.normal(y2 * 2, 0.3)

X3 = np.column_stack([X2, z])

fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                          subplot_kw={})
# 2-D view
axes[0].scatter(X3[:, 0], X3[:, 1], c=clrs[y2])
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('2-D projection (overlapping classes)')

# 3-D view via a second axes
fig.delaxes(axes[1])
ax3d = fig.add_subplot(1, 2, 2, projection='3d')
ax3d.scatter(X3[:, 2], X3[:, 0], X3[:, 1], c=clrs[y2])
ax3d.set_xlabel('Feature 3')
ax3d.set_ylabel('Feature 1')
ax3d.set_zlabel('Feature 2')
ax3d.set_title('3-D view (classes separate)')
plt.tight_layout()
```

Adding the third feature separates the two classes that overlapped in 2D — a concrete illustration of the blessing. However, if Feature 3 were pure noise, the added dimension would only degrade model performance.

:::{exercise}
:label: ex-eda-volume-growth

Write a function `grid_points(d, N)` that returns the number of uniform grid points needed to cover a $d$-dimensional unit hypercube at resolution $1/N$ (i.e. $N^d$). Plot $\log_{10}(\text{grid points})$ vs $d$ for $N = 10$ and $d \in \{1, 2, \ldots, 20\}$. Add a horizontal reference line at $\log_{10}(10^{83})$ (approximate number of atoms in the observable universe). At what dimension does the required sampling exceed this limit?
:::

## Inspecting High-Dimensional Features

Unlike a 1-D or 2-D dataset, we cannot simply scatter-plot a high-dimensional feature matrix. However, targeted visualizations of individual features and pairs of features remain essential for catching data quality issues, understanding scale and distribution, and identifying informative structure before modeling.

### Summary Statistics

The most direct starting point is computing per-feature summary statistics: mean, standard deviation, minimum, and maximum. For image data like MNIST, these statistics can be visualized directly as images.

```{code-cell} ipython3
means  = X_mnist.mean(axis=0).reshape(1, -1)
stdevs = X_mnist.std(axis=0).reshape(1, -1)
maxima = X_mnist.max(axis=0).reshape(1, -1)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, stat, title in zip(axes,
                            [means, stdevs, maxima],
                            ['Mean', 'Std Dev', 'Max']):
    show_image(stat, 0, ax=ax, title=title)
plt.tight_layout()
```

The mean image shows that pixel intensities are highest near the center of the 8×8 grid — all digits are roughly centered. The standard deviation image highlights pixels that vary a lot across digits; these are the most *informative* pixels. The maximum image reveals that several corner pixels are essentially always zero: their maximum value is near zero, meaning they carry no information regardless of which digit is shown. These zero-variance features are candidates for removal before modeling.

**Demonstration: Summary statistics for the Dow dataset**

For tabular data with named features, a DataFrame is more readable than an image:

```{code-cell} ipython3
feature_names = df.columns[1:-5]
summary = pd.DataFrame({
    'mean': X_dow.mean(axis=0),
    'std':  X_dow.std(axis=0),
    'min':  X_dow.min(axis=0),
    'max':  X_dow.max(axis=0),
}, index=feature_names)
summary.round(3)
```

Summary statistics can quickly reveal:
- Features with near-zero standard deviation (effectively constant — no predictive value)
- Features on very different scales (e.g. flows in the hundreds vs. concentrations near zero), which may require standardization before distance-based modeling
- Skewed ranges suggesting outliers or bounded physical quantities

:::{exercise}
:label: ex-eda-zero-pixels

Identify all MNIST pixels (features) with standard deviation less than 0.5. Visualize their positions as an 8×8 binary mask using `plt.imshow` (1 = near-zero variance, 0 = informative). How many such pixels are there, and where are they located in the image grid?
:::

### Histogram Plots

Summary statistics reduce each feature to a few numbers and can miss non-Gaussian structure — multimodality, heavy tails, or hard boundaries. Plotting a histogram for every feature gives a richer view of each marginal distribution.

```{code-cell} ipython3
n_side = 8   # 8×8 = 64 features
fig, axes = plt.subplots(n_side, n_side, figsize=(16, 16))
for i, ax in enumerate(axes.ravel()):
    ax.hist(X_mnist[:, i], bins=10, color=clrs[0])
    ax.set_xlabel(f'px {i}', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle('MNIST: per-pixel histograms', y=1.01)
plt.tight_layout()
```

:::{note}
**Do these features follow a normal distribution?** No. Most MNIST pixel histograms are strongly non-Gaussian: many corner pixels are almost always zero (spike at 0), while central pixels show bimodal or multimodal distributions reflecting the different shapes of the ten digit classes. This matters for methods that assume normality (e.g. linear discriminant analysis, Gaussian naive Bayes), but is not a problem for tree-based models or SVMs.
:::

## Scatter Plots

Histograms reveal single-feature distributions but say nothing about how features relate to each other. Scatter plots of pairs of features expose correlations, clusters, and outliers that univariate summaries miss.

For a dataset with $d$ features, a full scatter plot matrix has $d^2$ panels — 4,096 panels for MNIST. In practice, we restrict to a small subset of features at a time:

```{code-cell} ipython3
features = [0, 1, 2, 3, 4]   # first 5 MNIST pixels

# Manual matrix: diagonal = histogram, off-diagonal = scatter
n = len(features)
fig, axes = plt.subplots(n, n, figsize=(10, 10))
for i, fi in enumerate(features):
    for j, fj in enumerate(features):
        ax = axes[i, j]
        if i == j:
            ax.hist(X_mnist[:, fi], bins=15, color=clrs[0])
        else:
            ax.scatter(X_mnist[:, fj], X_mnist[:, fi],
                       s=1, alpha=0.3, color=clrs[0])
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(f'px {fi}', fontsize=8)
        if i == n - 1:
            ax.set_xlabel(f'px {fj}', fontsize=8)
plt.tight_layout()
```

The `seaborn` library provides `pairplot` as a convenient one-liner that produces the same layout with sensible defaults:

```{code-cell} ipython3
mnist_df = pd.DataFrame(X_mnist, columns=[f'px{i}' for i in range(X_mnist.shape[1])])
sns.pairplot(mnist_df[[f'px{i}' for i in features]], plot_kws={'s': 2, 'alpha': 0.3});
```

:::{note}
**What does a vertical or horizontal line in a scatter plot mean?** It means one of the two features plotted on that panel takes only a single value (or a very narrow range of values) across the dataset. The feature contributes no information to differentiate observations. This is exactly the near-zero-variance pattern identified in the histogram and summary statistics sections, and is a strong signal to exclude that feature before modeling.
:::

For the Dow dataset with continuous process variables, the pairplot shows smoother scatter patterns:

```{code-cell} ipython3
include_cols = list(df_dow_clean.columns[1:5])
sns.pairplot(df_dow_clean[include_cols].apply(pd.to_numeric, errors='coerce').dropna(),
             plot_kws={'s': 2, 'alpha': 0.2});
```

### Joint Plots

A joint plot zooms in on a single pair of features, showing the scatter and marginal distributions together. This is useful when a pairplot has identified a relationship worth examining in detail:

```{code-cell} ipython3
x_col = df_dow_clean.columns[3]   # x3: Input to Primary Column Bed 3 Flow
y_col = df_dow_clean.columns[4]   # x4: Input to Primary Column Bed 2 Flow

sns.jointplot(
    x=x_col, y=y_col,
    data=df_dow_clean.apply(pd.to_numeric, errors='coerce').dropna(),
    kind='reg',
    scatter_kws={'s': 2, 'alpha': 0.2},
);
```

The `kind='reg'` option overlays a linear regression fit with a confidence band, and prints the Pearson correlation coefficient and p-value. For x3 and x4 (two feed flows to the same column), a strong positive correlation is expected physically — both flows tend to increase or decrease together with production rate.

### Correlation Matrix

While a scatter plot matrix gives rich pairwise information, it is impractical for datasets with more than ~10 features. The correlation matrix compresses pairwise relationships into a single number per pair: the Pearson correlation coefficient. Values near ±1 indicate strong linear relationships; values near 0 indicate weak or no linear relationship.

```{code-cell} ipython3
# MNIST correlation matrix — select features 2–10 (avoid near-zero-variance corner pixels)
features_corr = list(range(2, 11))
corr_mnist = mnist_df[[f'px{i}' for i in features_corr]].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_mnist, annot=True, fmt='.2f',
            annot_kws={'fontsize': 8}, ax=ax)
ax.set_title('MNIST pixel correlations (features 2–10)')
plt.tight_layout()
```

For the Dow dataset, feature labels make the heatmap directly interpretable:

```{code-cell} ipython3
corr_dow = df_dow_clean[include_cols].apply(pd.to_numeric, errors='coerce').dropna().corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_dow, annot=True, fmt='.2f', ax=ax)
ax.set_title('Dow dataset: first 4 feature correlations')
plt.tight_layout()
```

**Demonstration: Correlation as a regression slope**

There is a precise algebraic relationship between the Pearson correlation coefficient and regression: if two features $x_i$ and $x_j$ are each standardized to zero mean and unit variance, the slope of a simple ordinary least squares regression of $x_j$ on $x_i$ equals their Pearson correlation coefficient. This connection helps interpret correlation matrices in terms of predictive relationships.

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Standardize x3 and x4 from the Dow dataset
X_pair = df_dow_clean[include_cols[:2]].apply(pd.to_numeric, errors='coerce').dropna().values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pair)

x_scaled = X_scaled[:, 0].reshape(-1, 1)
y_scaled = X_scaled[:, 1]

reg = LinearRegression().fit(x_scaled, y_scaled)
r = np.corrcoef(X_scaled[:, 0], X_scaled[:, 1])[0, 1]

print(f'OLS slope on standardized features: {reg.coef_[0]:.6f}')
print(f'Pearson correlation coefficient:     {r:.6f}')
print(f'Difference: {abs(reg.coef_[0] - r):.2e}')
```

The two values agree to numerical precision. Intuitively, the correlation matrix is a compact representation of all pairwise linear regressions on standardized data — a useful framing when deciding which features to include or exclude from a model.

:::{exercise}
:label: ex-eda-corr-slope

Extend the demonstration above to verify that the **full correlation matrix** of the first four Dow features equals the matrix of pairwise OLS regression slopes on standardized data. For each off-diagonal pair $(i, j)$: standardize both features, fit `LinearRegression`, and compare `.coef_[0]` to the corresponding entry in `corr_dow`. Display the maximum absolute difference across all pairs.
:::

## Summary

- High-dimensional data presents challenges (exponentially sparse sampling from the curse of dimensionality) and opportunities (better class separability from the blessing of dimensionality).
- Summary statistics (mean, std, max) computed per feature can expose constant features, scale mismatches, and structured spatial patterns in image data.
- Histogram grids reveal marginal distributions; most real-world features are non-Gaussian, which matters for methods that assume normality.
- Scatter plot matrices and seaborn `pairplot` expose pairwise relationships and flag zero-variance features (appear as vertical or horizontal lines).
- Joint plots provide detailed views of individual feature pairs, including regression fits and marginal distributions.
- Correlation heatmaps compress all pairwise linear relationships into a single matrix; the Pearson correlation between two standardized features equals the OLS regression slope.
- Near-zero-variance features and highly correlated feature groups are candidates for removal before fitting a model — both waste model capacity without adding information.

## Additional Reading

- [Seaborn documentation: pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html)
- [Seaborn documentation: heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- [Seaborn documentation: jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html)
- [Scikit-learn: the digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Wikipedia: Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
