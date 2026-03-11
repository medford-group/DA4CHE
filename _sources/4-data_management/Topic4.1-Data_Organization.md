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

# Data Organization

:::{admonition} Learning Objectives
:class: tip

By the end of this chapter, you will be able to:
- Explain the tidy data principles and recognize tidy vs. untidy data structures
- Index and filter a pandas DataFrame by column name, row label, and datetime range
- Detect and handle missing values using dropping, imputation, and correlation analysis
- Identify outliers with the z-score method and remove them systematically
- Store and retrieve large datasets efficiently using HDF5 and Parquet formats
- Explain the trade-offs between different missing-value strategies and storage formats
:::

Data organization and management is a critical part of any data science project.
Yet it is often underestimated: in practice, data scientists spend the majority
of their time cleaning and preparing data rather than fitting models. Poor data
management leads to silent errors that are hard to trace — wrong predictions built
on corrupted inputs. Good data management, by contrast, creates a reproducible
pipeline where every transformation is explicit and documented.

This chapter uses the **Dow Chemical distillation column dataset**: a real
industrial time-series with 12 sensor measurements (reflux flow, feed flow,
temperature, etc.) recorded at irregular intervals, along with a target variable
`y:Impurity`. The dataset contains the kinds of problems that appear routinely in
practice: inconsistent null representations, missing entries, and outliers.

---

## Tidy Data Principles

Before writing any code, it is worth establishing a mental framework for what
well-organized data looks like. Wickham (2014) formalized the concept of
**tidy data**, which defines a standard structure for analytical datasets:

1. **Each variable forms a column.** A variable is any quantity you measure or
   record — temperature, flow rate, class label, timestamp.
2. **Each observation forms a row.** One row = one measurement event (e.g., one
   timestamp in a time-series, one material in a materials dataset).
3. **Each type of observational unit forms a separate table.** Don't mix
   patient demographics with lab measurements in the same table; keep them
   separate and join on a key when needed.

Data that violates these rules is called **untidy** (or "messy"). Common untidy
patterns include:

- Column headers that are values, not variable names (e.g., months as column
  headers instead of a `month` column)
- Multiple variables stored in one column (e.g., `"glucose/insulin"`)
- Multiple observational units in one table (sensor metadata mixed with readings)
- Variables stored in both rows and columns (wide vs. long confusion)

The Dow dataset is already tidy: each row is one timestamped reading, each
column is one sensor or the target variable. Most pandas operations — filtering,
groupby, merge, pivot — assume tidy data implicitly, so arriving in tidy form
avoids many downstream headaches.

When data arrives in untidy form, `pd.melt()` (wide → long) and `pd.pivot()`
(long → wide) are the primary reshape tools. For example, if temperature readings
for three sensors arrived as three separate columns (`T_sensor1`, `T_sensor2`,
`T_sensor3`), melting would convert them to a long format with a `sensor_id`
column and a single `temperature` column — enabling a single groupby operation
instead of three separate code paths.

:::{exercise}
:label: ex-dm-tidy-check

After loading `df`, print `df.dtypes` and `df.isnull().sum()` to perform a basic
data validation check. Then answer: does the Dow DataFrame satisfy all three tidy
data rules? Identify one potential violation (hint: think about what each column
represents and whether the `Date` column is a variable or an index).
:::

:::{note}
**Data validation** is a closely related concern. After loading data, it is
good practice to programmatically assert that the data meets your expectations:
column types are correct, values fall within physically plausible ranges, and
no unexpected null values exist. Checking `df.dtypes`, `df.describe()`, and
`df.isnull().sum()` immediately after loading is a minimal validation step.
For production pipelines, libraries like **pandera** (schema-based validation
for DataFrames) and **great_expectations** (expectation suites for large data)
provide systematic validation frameworks, though they are beyond the scope of
this course.
:::

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('../settings/plot_style.mplstyle')

clrs = np.array(['#003057', '#EAAA00', '#4B8B9B', '#B3A369', '#377117',
                 '#1879DB', '#8E8B76', '#F5D580', '#002233'])
```

```{code-cell} ipython3
df = pd.read_excel('data/impurity_dataset-training.xlsx')
df.head(3)
```

```{code-cell} ipython3
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df["Date"].min()} → {df["Date"].max()}')
```

---

## Pandas Indexing and Filtering

Pandas provides flexible tools for selecting subsets of data. Understanding the
distinction between label-based and position-based indexing is essential for
avoiding subtle bugs.

**Column access** — use the column name directly:

```{code-cell} ipython3
df['x5:Primary Column Feed Flow from Feed Column'].head()
```

**Label-based indexing with `.loc`** — selects by the row *label* (which is the
index value, not necessarily the position) and the column name:

```{code-cell} ipython3
df.loc[0, 'x1:Primary Column Reflux Flow']
```

**Position-based indexing with `.iloc`** — zero-based integer positions regardless
of the index:

```{code-cell} ipython3
# .iloc always uses integer positions, even after filtering
df.iloc[0, 1]
```

### Filtering with Boolean Masks

Logical conditions return a boolean Series that can be used to select rows:

```{code-cell} ipython3
mask = df['x1:Primary Column Reflux Flow'] > 350
df_high_reflux = df[mask]
print(f'Rows with reflux > 350: {len(df_high_reflux)} of {len(df)}')
df_high_reflux.head(3)
```

:::{note}
After filtering, the row index retains the original labels (e.g., rows 5, 8, 11…).
Use `.iloc` rather than `.loc` for position-based access on filtered DataFrames to
avoid `KeyError`. Alternatively, call `.reset_index(drop=True)` to re-number
from zero.
:::

### Renaming Columns

Long column names improve readability in the raw data but are cumbersome to type
in code. A renaming dictionary maps old names to short aliases:

```{code-cell} ipython3
columns = df.columns
# Keep the short prefix before ':' (e.g. 'x1:Primary Column Reflux Flow' → 'x1')
new_columns = {ci: ci.split(':', 1)[0] if ':' in ci else ci for ci in columns}
df_shortnames = df.rename(columns=new_columns)
df_shortnames.head(3)
```

### Datetime Indexing

When a column contains timestamps, setting it as the index enables natural
slice-based time selection:

```{code-cell} ipython3
df_dt = df_shortnames.set_index('Date')
df_dt.head(3)
```

```{code-cell} ipython3
# Slice a single day by string — pandas parses the date automatically
df_oneday = df_dt['2015-12-02':'2015-12-03']
print(f'Rows for Dec 2–3: {len(df_oneday)}')
df_oneday.head()
```

Datetime string parsing in pandas is flexible — you can pass full timestamps
(`'2015-12-02 08:30:00'`), just dates (`'2015-12-02'`), or even just years
(`'2015'`), and pandas will interpret the slice boundaries appropriately.

The underlying NumPy array is always accessible via `.values`:

```{code-cell} ipython3
X_oneday = df_oneday.values
print(f'Array shape: {X_oneday.shape}, dtype: {X_oneday.dtype}')
```

### Demonstration: Extracting a Labeled Subset

The code below extracts a clean numeric subset for a specific date range,
keeping only the 12 sensor features (x1–x12):

```{code-cell} ipython3
df_subset = (df_shortnames
             .set_index('Date')['2015-12-05':'2015-12-12']
             .loc[:, 'x1':'x12'])
print(f'Subset shape: {df_subset.shape}')
df_subset.head()
```

:::{exercise}
:label: ex-dm-dow-filter

Using `df_shortnames`, create a filtered DataFrame that satisfies **all three**
conditions simultaneously:
1. Date between `2015-12-10` and `2015-12-20`
2. `x1` (Reflux Flow) greater than 300
3. Only keep columns `x1` through `x6` and `y`

Print the shape of the result and display the first five rows.
:::

---

## Data Quality

### Detecting Missing Values

Missing values in real industrial datasets rarely appear as clean `NaN`. Operators
may enter a placeholder like `0`, `-999`, or — as in this dataset — an exclamation
mark `!` to flag a sensor alarm or out-of-range reading. These must be identified
before any numerical computation, because a string mixed into a numeric column will
silently convert the whole column to `object` dtype.

```{code-cell} ipython3
nondate_cols = df.columns[1:]  # everything except 'Date'
print(f'Column dtypes:\n{df[nondate_cols].dtypes.value_counts()}')
```

The `object` dtype signals that at least some values are non-numeric. We can find
the offending rows using `pd.isnull`:

```{code-cell} ipython3
df[pd.isnull(df).any(axis=1)].head(3)
```

To also catch non-null but non-numeric entries like `!`, we write a custom checker:

```{code-cell} ipython3
def is_real_and_finite(x):
    try:
        val = float(x)
        return np.isfinite(val)
    except (TypeError, ValueError):
        return False

# Apply element-wise — compatible with all pandas versions
numeric_map = df[nondate_cols].apply(lambda col: col.map(is_real_and_finite))
print(f'Fully numeric rows: {numeric_map.all(axis=1).sum()} of {len(df)}')
```

### Dropping Observations

When the proportion of problematic rows is small, dropping them is the simplest
strategy. The `numeric_map` boolean DataFrame lets us select only fully numeric rows:

```{code-cell} ipython3
real_rows = numeric_map.all(axis=1).values
df_dropped = df[real_rows].copy()
X = df_dropped.loc[:, nondate_cols].values.astype('float')
print(f'After dropping bad rows: {df_dropped.shape}  (lost {len(df)-len(df_dropped)} rows)')
```

### Dropping Features by Correlation

Not all columns are equally valuable. A feature that is almost perfectly correlated
with another is redundant — it adds noise to some models (e.g., ordinary least squares)
without adding information.

```{code-cell} ipython3
# Columns still have object dtype after filtering; cast explicitly for corr
numeric_only = df_dropped.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
corr = numeric_only.corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0,
            linewidths=0.3, annot=False)
ax.set_title('Feature correlation matrix')
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
target_col = 'Avg_Delta_Composition Primary Column'
high_corr = corr[target_col][corr[target_col] > 0.95]
print(f'Features highly correlated with "{target_col}":\n{high_corr}')
```

`Primary Column Make Flow` is essentially a linear proxy for
`Avg_Delta_Composition Primary Column`, so we can safely drop the latter without
losing predictive information. In general, keeping both highly correlated features
can cause numerical instability in models that invert the feature matrix (e.g., OLS)
and can make regularization hyperparameter tuning less interpretable.

```{code-cell} ipython3
df_no_avg = df_dropped.drop(columns=[target_col])
print(f'Shape after dropping redundant column: {df_no_avg.shape}')
```

### Imputation

Sometimes dropping rows or columns discards too much data. **Imputation** fills in
missing values using information from the rest of the dataset.

A simple approach uses the linear relationship between two correlated features to
predict the missing values in one from the other:

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

# Predict Avg_Delta_Composition from x6 (Make Flow) — high corr seen above
x_train = pd.to_numeric(df_dropped['x6:Primary Column Make Flow']).values.reshape(-1, 1)
y_train = pd.to_numeric(df_dropped['Avg_Delta_Composition Primary Column']).values

reg = LinearRegression().fit(x_train, y_train)
print(f'R² for imputation model: {reg.score(x_train, y_train):.3f}')
```

:::{note}
The choice of missing-value strategy has significant downstream effects on model
performance. Dropping observations is safe when missingness is rare and random.
Dropping features by correlation is justified when two columns are near-perfectly
correlated. Imputation via regression is powerful but introduces model assumptions;
always validate that the imputation model is accurate before using it.
A common mistake is to fit an imputation model on the **full** dataset (including
test data), which leaks future information into training. Fit imputation models
only on training data.
:::

:::{exercise}
:label: ex-dm-mean-impute

Using `df_dropped`, create a copy of the DataFrame where any `NaN` or non-numeric
value (after converting to float with `pd.to_numeric(..., errors='coerce')`) is
replaced with the **column mean**. Use `DataFrame.fillna()`. Print the count of
remaining null values to confirm the result is complete.
:::

:::{note}
**Connecting cleaning to the modeling pipeline.** Every cleaning step applied to
training data must be identically applied to any new data at inference time —
otherwise the model sees data in a different form than it was trained on. A common
mistake is to compute, say, the column mean for imputation on the full dataset
(including test rows) and then use that mean to fill training rows. This leaks
future information. `sklearn.pipeline.Pipeline` solves this cleanly: wrap each
cleaning step as a `Transformer` (with `fit` on training data only and `transform`
applied to both), and the pipeline ensures correct train/test separation
automatically. This pattern is not explored further here but is worth adopting in
any real project.
:::

---

## Outlier Detection

The general definition of an outlier is a datapoint that was not created by the
same underlying process. However, in practice this definition is not always
helpful, since it requires knowledge of the mechanism that generated the data.
We will not delve into advanced outlier detection methods here, but show a few
simple examples that are commonly used in practice.

### The Z-score Method

An outlier is an observation that deviates unusually far from the bulk of the data.
The **z-score** measures how many standard deviations a point is from the mean:

$$z_i = \frac{x_i - \mu}{\sigma}$$

Points with $|z_i| > z_\text{cutoff}$ (commonly 3) are flagged as outliers under
the assumption that the data follows a Gaussian distribution.

```{code-cell} ipython3
# Demonstrate on a single column
col = 'x3:Input to Primary Column Bed 3 Flow'
xi = pd.to_numeric(df_dropped[col], errors='coerce').dropna()
mu, sigma = xi.mean(), xi.std()
z_cutoff = 3

z_scores = (xi - mu) / sigma
outlier_mask = z_scores.abs() > z_cutoff
xi_clean = xi[~outlier_mask]

print(f'Column: {col}')
print(f'  Total rows:   {len(xi)}')
print(f'  Outliers:     {outlier_mask.sum()}')
print(f'  Retained:     {len(xi_clean)}')
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(xi.values, bins=30)
axes[0].set_title('Before outlier removal')
axes[0].set_xlabel(col.split(':')[1])

axes[1].hist(xi_clean.values, bins=30)
axes[1].set_title('After z-score outlier removal (|z| > 3)')
axes[1].set_xlabel(col.split(':')[1])

plt.tight_layout()
plt.show()
```

### Demonstration: Applying Z-score to All Columns

```{code-cell} ipython3
df_no_outliers = df_dropped.copy()
z_cutoff = 3

for col in df_dropped.columns[1:]:   # skip 'Date'
    try:
        xi = pd.to_numeric(df_no_outliers[col], errors='coerce')
        mu, sigma = xi.mean(), xi.std()
        if sigma == 0:
            continue
        z_scores = (xi - mu) / sigma
        df_no_outliers = df_no_outliers[z_scores.abs() <= z_cutoff]
    except Exception:
        pass

print(f'Before: {df_dropped.shape[0]} rows')
print(f'After:  {df_no_outliers.shape[0]} rows  '
      f'({df_dropped.shape[0]-df_no_outliers.shape[0]} removed)')
```

The z-score method assumes that the data are approximately Gaussian and evaluates
each feature independently. For variables with discrete or heavily skewed
distributions, IQR-based filtering is more robust.

:::{exercise}
:label: ex-dm-iqr-outlier

Implement an **IQR-based** outlier filter as an alternative to z-score.
For a single column `x1:Primary Column Reflux Flow`:
1. Compute Q1 (25th percentile), Q3 (75th percentile), and IQR = Q3 − Q1.
2. Flag points outside $[\text{Q1} - 1.5 \cdot \text{IQR},\ \text{Q3} + 1.5 \cdot \text{IQR}]$.
3. Plot histograms before and after removal.
4. Compare the number of outliers flagged by IQR vs. z-score (cutoff = 3) for this column.
:::

### Multivariate Outlier Detection

Z-score and IQR methods examine each feature in isolation. A point can look
perfectly normal on every individual feature while being a genuine **multivariate
outlier** — sitting in a region of joint feature space that never occurs in real
data. For example, a very high reflux flow combined with a very low feed flow
might be individually plausible but operationally impossible together.

Two scikit-learn tools detect multivariate outliers:

- **`EllipticEnvelope`** (`sklearn.covariance.EllipticEnvelope`) fits a Gaussian
  model to the joint distribution. It uses the **Mahalanobis distance**, which
  accounts for feature correlations — so two features that are individually
  normal but jointly impossible are correctly flagged. It works best when the
  data are approximately elliptically distributed.
- **`IsolationForest`** (`sklearn.ensemble.IsolationForest`) is a non-parametric
  alternative that isolates anomalies by recursively partitioning the feature
  space with random splits. Points that are easy to isolate (requiring few splits)
  are anomalous. It scales well to high dimensions without distributional
  assumptions.

### Demonstration: Multivariate Outlier Detection

:::{note}
Both `EllipticEnvelope` and `IsolationForest` require a `contamination` parameter, which specifies the **expected fraction of outliers in the dataset** (a number between 0 and 0.5). This value is set by the analyst based on domain knowledge or prior inspection of the data. A value of `0.05` tells the model to treat the 5% of points with the most anomalous scores as outliers and the remaining 95% as inliers. Choosing `contamination` too high flags clean data as anomalous; too low misses real outliers. In exploratory analysis, it is common to try a few values (e.g. 0.01, 0.05, 0.10) and inspect the flagged points for reasonableness.
:::

```{code-cell} ipython3
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import numpy as np

# Use the two reflux/feed columns as a 2D example
cols_2d = ['x1:Primary Column Reflux Flow', 'x3:Input to Primary Column Bed 3 Flow']
X2 = df_no_outliers[cols_2d].apply(pd.to_numeric, errors='coerce').dropna()

# EllipticEnvelope: fits a robust Gaussian and flags points outside the
# contamination fraction (5%) as outliers
ee = EllipticEnvelope(contamination=0.05, random_state=0)
ee_labels = ee.fit_predict(X2)   # +1 = inlier, -1 = outlier

# IsolationForest: tree-based anomaly score, no distributional assumption
iso = IsolationForest(contamination=0.05, random_state=0)
iso_labels = iso.fit_predict(X2)  # +1 = inlier, -1 = outlier

print(f'EllipticEnvelope: {(ee_labels == -1).sum()} outliers flagged')
print(f'IsolationForest:  {(iso_labels == -1).sum()} outliers flagged')
print(f'Agreement (both flag as outlier): {((ee_labels == -1) & (iso_labels == -1)).sum()}')
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, labels, title in zip(
    axes,
    [ee_labels, iso_labels],
    ['EllipticEnvelope', 'IsolationForest'],
):
    inliers  = X2[labels ==  1]
    outliers = X2[labels == -1]
    ax.scatter(inliers.iloc[:, 0],  inliers.iloc[:, 1],
               c=clrs[0], s=10, alpha=0.4, label='Inlier')
    ax.scatter(outliers.iloc[:, 0], outliers.iloc[:, 1],
               c=clrs[1], s=40, marker='x', label='Outlier')
    ax.set_xlabel(cols_2d[0].split(':')[1])
    ax.set_ylabel(cols_2d[1].split(':')[1])
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()
```

:::{exercise}
:label: ex-dm-multivar-outlier

Apply `IsolationForest` to the **full numeric feature matrix** of `df_no_outliers`
(all columns except `Date`) with `contamination=0.05`.

1. How many rows are flagged as outliers across all features?
2. Inspect the flagged rows — think about whether these rows look unusual compared
   to the bulk of the data.
3. How does the count compare to applying z-score (cutoff = 3) column-by-column?
:::

---

## Efficient Data Storage

### Why File Format Matters

The Dow dataset is small enough to load from Excel in a few seconds. For real
process data — which may span months of high-frequency sensor readings at one-second
intervals, easily reaching millions of rows — loading an Excel or CSV file at the
start of every analysis session becomes impractical. A 10 GB CSV that takes 3 minutes
to load and parse effectively prevents interactive exploration. Choosing an appropriate
file format is an engineering decision with real consequences for workflow efficiency.

The three main properties to consider are **read speed**, **write speed**, and
**partial-read support** (the ability to load a subset of rows or columns without
reading the entire file). Text formats like CSV have none of these; binary formats
like HDF5 and Parquet have all three.

```{code-cell} ipython3
import time
start = time.time()
_ = pd.read_excel('data/impurity_dataset-training.xlsx')
print(f'Excel load time: {time.time()-start:.2f} s')
```

### HDF5 Files

**HDF5** (Hierarchical Data Format 5) is a binary format designed for large
numerical datasets. Its key advantages are:

- **Fast partial reads** — you can load a single column from a 10 GB file without
  reading the whole thing into memory.
- **Hierarchical structure** — datasets can be organized into groups, like a
  filesystem within a file.
- **Metadata attributes** — descriptive information can be stored alongside the
  data, keeping provenance in one place.

```{code-cell} ipython3
import h5py
from pathlib import Path

hdf_path = Path('data/impurity_data.hdf5')
if hdf_path.exists():
    hdf_path.unlink()   # remove stale file to avoid append errors
```

```{code-cell} ipython3
# Prepare numeric array (drop Date column, cast object cols to float)
df_numeric = df_no_outliers.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').dropna()
X_dow = df_numeric.values.astype('float')
col_names_dow = list(df_numeric.columns)
print(f'Array shape: {X_dow.shape}')

with h5py.File(hdf_path, 'w') as f:
    dset = f.create_dataset('training', data=X_dow)
    # Store metadata as attributes
    dset.attrs['name']    = 'Dow impurity training set'
    dset.attrs['company'] = 'Dow Chemical'
    dset.attrs['course']  = 'ChBE 4745/6745'
    dset.attrs['columns'] = str(col_names_dow)
    print(f'Dataset stored: shape={dset.shape}, dtype={dset.dtype}')
    print(f'Attributes: {dict(dset.attrs)}')
```

HDF5 files support hierarchical organization with groups (analogous to folders):

```{code-cell} ipython3
with h5py.File(hdf_path, 'a') as f:
    grp = f.create_group('data_by_feature')
    for i, col in enumerate(col_names_dow):
        ds = grp.create_dataset(col, data=X_dow[:, i])
    print('Groups and datasets:')
    for key in f.keys():
        print(f'  /{key}: {f[key]}')
```

### Timing Comparison

```{code-cell} ipython3
# Full array load from HDF5
start = time.time()
with h5py.File(hdf_path, 'r') as f:
    X_loaded = f['training'][:, :]
print(f'HDF5 full load:   {time.time()-start:.4f} s')

# Single-column load — the real HDF5 advantage
start = time.time()
with h5py.File(hdf_path, 'r') as f:
    x1_col = f['training'][:, 0]
print(f'HDF5 one column:  {time.time()-start:.4f} s')

# In-memory access for comparison
start = time.time()
_ = X_dow[:, 0]
print(f'In-memory slice:  {time.time()-start:.6f} s')
```

:::{note}
HDF5's selective-read advantage becomes significant only for datasets that are too
large to fit in RAM. For the datasets in this course (thousands to tens of thousands
of rows), the speed difference is small. However, building good HDF5 habits now
pays dividends when you encounter GB- or TB-scale industrial process data in practice.
HDF5 is also deeply embedded in scientific computing workflows — NetCDF (climate/ocean
data), PyTables, and many HPC file systems build on the HDF5 specification.
:::

### Modern Alternatives: Parquet

**Parquet** is a columnar binary format developed by Apache that has become the
de facto standard for analytical data pipelines outside of scientific computing.
Unlike HDF5, Parquet stores schema information (column names and types) natively,
handles mixed dtypes (including strings and timestamps) without manual conversion,
and integrates directly with cloud data warehouses (BigQuery, Snowflake, Databricks).
It is natively supported by `pandas`, `polars`, `Spark`, and `DuckDB`.

```{code-cell} ipython3
# Write a clean DataFrame to Parquet
df_clean = df_no_outliers.copy()
# Ensure numeric cols are float (Parquet does not accept object dtype numerics)
for col in df_clean.columns[1:]:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

start = time.time()
df_clean.to_parquet('data/impurity_clean.parquet', index=False)
print(f'Parquet write: {time.time()-start:.4f} s')

start = time.time()
df_reloaded = pd.read_parquet('data/impurity_clean.parquet')
print(f'Parquet read:  {time.time()-start:.4f} s')

print(f'\nShape preserved: {df_reloaded.shape}')
print(f'Dtypes after round-trip:\n{df_reloaded.dtypes.value_counts()}')
```

Other modern formats worth knowing:

| Format | Best for | Key advantage |
|---|---|---|
| **Parquet** | Tabular DataFrames | Schema metadata, columnar reads, cloud-native |
| **Feather / Arrow** | In-process exchange | Near-zero serialization overhead |
| **Zarr** | Multi-dimensional arrays | Cloud-native, concurrent writes, chunked storage |
| **LMDB** | Key-value lookups | Memory-mapped, extremely fast random reads; widely used in machine learning data loaders (PyTorch, TensorFlow) |
| **HDF5** | Scientific numerical arrays | Hierarchical, mature ecosystem, partial reads |

:::{exercise}
:label: ex-dm-parquet

Compare the file sizes of `impurity_data.hdf5` and `impurity_clean.parquet`
using `Path('data/...').stat().st_size`. Then use `pd.read_parquet()` to load
only two numeric feature columns and the target `y` (Parquet supports column
selection natively via the `columns=` argument). Print the shape and compare
the read time to the full-file HDF5 read above.
:::

---

## Summary

- **Tidy data** (Wickham, 2014): each variable is a column, each observation is
  a row, each observational unit is a table. Most pandas operations assume this
  structure; arriving in tidy form avoids many downstream headaches.
  `pd.melt()` and `pd.pivot()` are the primary reshape tools when data is untidy.

- **Data validation** — checking dtypes, value ranges, and null counts immediately
  after loading — catches silent data quality issues before they corrupt downstream
  analysis. Libraries like pandera and great_expectations automate this in production.

- **Pandas** provides flexible tools for indexing (`.loc`, `.iloc`, boolean masks),
  renaming, and datetime-based slicing. Setting a datetime column as the index
  enables natural time-range queries.

- **Missing values** in industrial data rarely appear as simple `NaN` — custom
  detection functions are often needed. Strategies include dropping problematic
  rows or columns, and regression imputation. All cleaning must be fit on training
  data only; `sklearn.pipeline.Pipeline` enforces this automatically.

- **Correlation analysis** identifies redundant features that can be removed without
  information loss, reducing model complexity and preventing numerical issues.

- **Outlier detection** with z-scores is simple but assumes Gaussian, independent
  variables. For high-dimensional or non-Gaussian data, consider IQR bounds,
  Mahalanobis distance (`EllipticEnvelope`), or Isolation Forest.

- **HDF5** excels for large numerical arrays: hierarchical structure, metadata
  attributes, and fast partial reads. **Parquet** is the modern default for tabular
  DataFrames: schema metadata, native mixed-dtype support, and cloud integration.
  **LMDB** is a strong choice for key-value data in machine learning pipelines.

## Additional Reading

- McKinney, W., *Python for Data Analysis* (3rd ed.) — the definitive pandas
  reference, covering indexing, missing data, and I/O in depth
- Wickham, H. (2014), "Tidy Data," *Journal of Statistical Software* 59(10) —
  foundational principles for organizing tabular data; [free PDF](https://doi.org/10.18637/jss.v059.i10)
- The HDF Group, [HDF5 User's Guide](https://docs.hdfgroup.org/hdf5/develop/index.html)
- Apache Parquet, [Parquet Format Specification](https://parquet.apache.org/docs/)
- pandas documentation:
  [Indexing](https://pandas.pydata.org/docs/user_guide/indexing.html),
  [Missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html),
  [IO tools](https://pandas.pydata.org/docs/user_guide/io.html)
