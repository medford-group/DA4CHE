# Exploratory Data Analysis

Before applying any machine learning model, it is essential to understand the
structure and properties of the data you are working with. Exploratory Data
Analysis (EDA) is the practice of summarizing, visualizing, and probing datasets
to uncover patterns, anomalies, and relationships that inform both feature
engineering and model selection.

This module focuses on the unique challenges that arise when the number of
features is large — a regime commonly encountered in chemical engineering applications
such as process data, spectroscopy, and materials science. We use two contrasting
datasets throughout: the **Dow Chemical distillation column dataset** (40 continuous
process variables) and the **MNIST handwritten digits dataset** (64 pixel-intensity
features), which together illustrate EDA strategies across a range of feature types
and scales.

## Topics

- **Topic 5.1 — High-Dimensional Data**: The curse and blessing of dimensionality,
  summary statistics, histogram grids, scatter plot matrices, and correlation heatmaps.

- **Topic 5.2 — Dimensionality Reduction**: Principal component analysis, t-SNE,
  and other techniques for projecting high-dimensional data to interpretable
  low-dimensional representations.

- **Topic 5.3 — Clustering**: k-means, hierarchical clustering, DBSCAN, and
  evaluation metrics for unsupervised grouping of data.

- **Topic 5.4 — Generative Models**: Gaussian mixture models and autoencoders
  for density estimation and latent-space representation.

## Datasets

The **Dow impurity dataset** (`impurity_dataset-training.xlsx`) contains time-stamped
sensor readings from a primary distillation column with 40 process variable inputs and
a product impurity target. The same dataset was introduced in Module 4; here the focus
shifts from data cleaning to exploratory analysis of the feature structure.

The **MNIST digits dataset** is a classic benchmark in machine learning, consisting
of 8×8 pixel images of hand-written digits 0–9. It is built into scikit-learn and
requires no download.
