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

# Nonlinear Feature Engineering

```{contents}
:local:
:depth: 2
```

:::{admonition} Learning Objectives
:class: tip

By the end of this chapter, you will be able to:
- Explain why nonlinear feature transformations are necessary when linear combinations cannot improve a linear model's performance.
- Construct polynomial feature matrices with `sklearn.preprocessing.PolynomialFeatures` and estimate the combinatorial growth of feature count with polynomial order.
- Apply LASSO regularization to a polynomial feature matrix to select a sparse, interpretable set of nonlinear features.
- Use `autofeat` to generate and select physically meaningful nonlinear features from engineering data, with and without dimensional constraints.
- Articulate the practical trade-offs of symbolic regression: interpretability and efficiency at test time vs. high training cost and sensitivity to train/test splits.
:::

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.style.use('../settings/plot_style.mplstyle')

clrs = np.array(['#003057', '#EAAA00', '#4B8B9B', '#B3A369', '#377117',
                 '#1879DB', '#8E8B76', '#F5D580', '#002233'])
```

```{code-cell} ipython3
# Load Dow dataset — same preprocessing as Topic 2.4/2.5
df = pd.read_excel('data/impurity_dataset-training.xlsx')

def is_real_and_finite(x):
    try:
        val = float(x)
        return np.isfinite(val)
    except (TypeError, ValueError):
        return False

nondate_cols = df.columns[1:]
numeric_map = df[nondate_cols].apply(lambda col: col.map(is_real_and_finite))
real_rows = numeric_map.all(axis=1).values

all_data = df[nondate_cols].values
dow_feature_names = list(nondate_cols)

X_dow = np.array(all_data[real_rows, :-5], dtype='float')
y_dow = np.array(all_data[real_rows, -3],  dtype='float')

# Standardize features
X_dow_scaled = (X_dow - X_dow.mean(axis=0)) / X_dow.std(axis=0)
print(f'X shape: {X_dow_scaled.shape},  y shape: {y_dow.shape}')
```

## Motivation: When Linear Combinations Are Not Enough

In Topic 2.5 we showed that linear combinations — including PCA and PLS — can reduce
dimensionality and sometimes improve model performance. But they share a fundamental
limitation: **a linear combination of features fed into a linear model is still a linear
model**. No matter how we rotate or re-weight the input columns, we cannot capture
interactions like $x_i \cdot x_j$ or nonlinear effects like $x_i^2$.

This matters for engineering systems, where the physics is rarely linear. A distillation
column impurity may depend on the product of two flow rates (a multiplicative interaction),
or on the square of a temperature difference. To capture such effects with a linear
regression model, we must **create the nonlinear features explicitly** — add columns like
$x_i^2$ or $x_i x_j$ to the feature matrix before fitting.

The challenge is that the number of such features grows rapidly, so we need both a
systematic way to generate them and a regularization strategy to select the useful ones.

:::{exercise}
:label: ex-reg-nonlin-motive

Verify the linear-model ceiling on the Dow dataset.

1. Using the scaled features `X_dow_scaled` and target `y_dow` from above, fit a
   `LinearRegression()` and record the train $r^2$.
2. Fit a `Ridge(alpha=1.0)` and compare the train and test $r^2$ (use a 70/30 split
   with `random_state=0`). Is there evidence of overfitting?
3. Compute the Pearson correlation between each individual feature and the target.
   Report the top-5 most correlated features. Are any correlations very strong
   ($|r| > 0.5$)?
4. Based on your results, argue briefly whether a linear model is likely to be
   the best achievable model for this dataset.
:::

## Polynomial Features

### Generating Polynomial Features with sklearn

`sklearn.preprocessing.PolynomialFeatures` creates all monomials up to a specified
degree from an input matrix. For degree 2 with two input features $x_0, x_1$, it produces
$[1,\, x_0,\, x_1,\, x_0^2,\, x_0 x_1,\, x_1^2]$.

```{code-cell} ipython3
from sklearn.preprocessing import PolynomialFeatures

# Subsample to keep memory manageable (see note below)
X_sub = X_dow_scaled[::2]
y_sub = y_dow[::2]

X_train, X_test, y_train, y_test = train_test_split(
    X_sub, y_sub, test_size=0.5, random_state=0)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test  = poly.transform(X_test)

print(f'Original features:  {X_train.shape[1]}')
print(f'Degree-2 features:  {X_poly_train.shape[1]}')
```

:::{note}
**Why subsample?** With 40 features and degree-2 polynomial expansion, the feature count
is $\binom{40 + 2}{2} - 1 = 859$. At degree 3 it exceeds 11,000 — larger than many training
sets. We subsample to avoid memory issues during the fitting step below. In practice you
would use regularization (LASSO, ridge) rather than subsampling to handle the
large-feature regime.
:::

### Combinatorial Explosion

The exact count of degree-$d$ polynomial features from $p$ original features (without the
intercept) is the multiset coefficient $\binom{p + d}{d} - 1$:

```{code-cell} ipython3
from math import comb

p = 40  # original features
for d in range(1, 5):
    n_poly = comb(p + d, d) - 1
    print(f'Degree {d}: {n_poly:,} features')
```

Beyond degree 2 or 3, the number of features quickly exceeds the number of data points,
making unregularized least squares completely ill-conditioned. This is one manifestation
of the **curse of dimensionality**.

### Regularized Polynomial Regression with LASSO

With more features than samples, standard linear regression overfits badly. LASSO
regularization shrinks many coefficients to exactly zero, effectively selecting a sparse
subset of the polynomial features:

```{code-cell} ipython3
# Baseline: plain linear regression on original features
linreg = LinearRegression()
linreg.fit(X_train, y_train)
r2_base_train = linreg.score(X_train, y_train)
r2_base_test  = linreg.score(X_test,  y_test)
print(f'Linear regression:  train r² = {r2_base_train:.3f},  test r² = {r2_base_test:.3f}')
```

```{code-cell} ipython3
# Unregularized: polynomial features + ordinary least squares — should overfit
linreg_poly = LinearRegression()
linreg_poly.fit(X_poly_train, y_train)
r2_poly_train = linreg_poly.score(X_poly_train, y_train)
r2_poly_test  = linreg_poly.score(X_poly_test,  y_test)
print(f'Poly + OLS:         train r² = {r2_poly_train:.3f},  test r² = {r2_poly_test:.3f}')
```

```{code-cell} ipython3
# LASSO on polynomial features — sparse selection
lasso = Lasso(alpha=1.0, max_iter=5000)
lasso.fit(X_poly_train, y_train)
r2_lasso_train = lasso.score(X_poly_train, y_train)
r2_lasso_test  = lasso.score(X_poly_test,  y_test)
n_nonzero = np.sum(lasso.coef_ != 0)
print(f'Poly + LASSO (α=1): train r² = {r2_lasso_train:.3f},  test r² = {r2_lasso_test:.3f}')
print(f'                    non-zero coefficients: {n_nonzero} of {len(lasso.coef_)}')
```

The LASSO reduces training accuracy compared to unregularized polynomial regression but
improves test accuracy significantly — and with far fewer nonzero coefficients.

### Interpreting the Selected Features

A key advantage over black-box approaches is that `PolynomialFeatures` can name every
output column:

```{code-cell} ipython3
feature_names = poly.get_feature_names_out(
    [f'x{i}' for i in range(X_train.shape[1])])

coef = lasso.coef_
nonzero_idx = np.where(coef != 0)[0]
sorted_idx = nonzero_idx[np.argsort(np.abs(coef[nonzero_idx]))[::-1]]

print('Top 10 selected polynomial features (by |coefficient|):')
for i in sorted_idx[:10]:
    print(f'  {feature_names[i]:30s}  coef = {coef[i]:+.4f}')
```

The feature names reveal which interactions and squared terms the model found useful. In
engineering problems, a feature like `x5^2` or `x3 x7` may have a clear physical
interpretation (e.g., a quadratic temperature effect or a cross-flow interaction).

:::{exercise}
:label: ex-reg-lasso-poly

Compare the test $r^2$ of LASSO polynomial regression against plain linear regression
as a function of the LASSO regularization strength $\alpha$.

1. Using the same 50/50 split above, sweep $\alpha$ over `np.logspace(-2, 2, 20)`.
2. For each $\alpha$, fit `Lasso(alpha=alpha, max_iter=5000)` on `X_poly_train`
   and record train and test $r^2$ and the number of nonzero coefficients.
3. Plot train $r^2$, test $r^2$, and nonzero coefficient count vs. $\log_{10}(\alpha)$
   on two subplots (shared x-axis).
4. Identify the $\alpha$ that maximizes test $r^2$ and report the corresponding
   number of nonzero features. How does this compare to the baseline linear regression
   test $r^2$?
:::

## Symbolic Regression with `autofeat`

Polynomial features restrict transformations to integer powers of original features.
**Symbolic regression** extends this to arbitrary nonlinear combinations — ratios,
exponentials, square roots, and products thereof — and uses a built-in feature selection
step to keep only the combinations that improve prediction.

The `autofeat` library implements this workflow: it generates a large library of
nonlinear feature candidates, then applies multi-stage feature selection to reduce to a
compact, interpretable set.

```{note}
`autofeat` is not installed by default. Install it with:
`pip install autofeat`
or
`conda install -c conda-forge autofeat`
```

### Basic `AutoFeatRegressor`

```{code-cell} ipython3
try:
    from autofeat import AutoFeatRegressor
    _autofeat_available = True
except ImportError:
    print('autofeat not installed — skipping symbolic regression cells.')
    print('Install with: pip install autofeat')
    _autofeat_available = False

if _autofeat_available:
    # Use the same subsampled/split data from above
    transforms = ['1/', 'exp', 'abs', 'sqrt', '^2', '^3']
    afreg = AutoFeatRegressor(
        verbose=1,
        feateng_steps=2,
        featsel_runs=1,
        transformations=transforms,
    )
    afreg.fit(X_train, y_train)
```

The `feateng_steps=2` parameter controls how many times transformations are nested
(e.g., `sqrt(x0 * exp(x1))` requires 2 steps). Higher values create exponentially more
candidate features.

```{code-cell} ipython3
if _autofeat_available:
    # New nonlinear features selected as predictive
    print('New features generated:')
    print(afreg.new_feat_cols_)
```

```{code-cell} ipython3
if _autofeat_available:
    # All features (original + new) retained after final selection
    print('All selected features (used in model):')
    print(afreg.good_cols_)
```

```{code-cell} ipython3
if _autofeat_available:
    r2_af_test = afreg.score(X_test, y_test)
    print(f'autofeat test r²: {r2_af_test:.3f}')
```

The selected features include ratios, square roots, and products that a polynomial
expansion would not generate. Note that results vary across runs due to the random
train/test split; do not over-interpret any specific feature unless it has clear
physical meaning and is consistent across multiple runs.

### Transforming the Feature Matrix Directly

We can also access the full transformed feature matrix and build our own model:

```{code-cell} ipython3
if _autofeat_available:
    X_af_train = afreg.transform(X_train)
    X_af_test  = afreg.transform(X_test)

    linreg2 = LinearRegression()
    linreg2.fit(X_af_train, y_train)
    r2_af_linreg = linreg2.score(X_af_test, y_test)
    print(f'Linear model on autofeat features: test r² = {r2_af_linreg:.3f}')
```

If this is much lower than `afreg.score()`, it indicates overfitting because the full
transformed matrix includes features beyond those in `good_cols_`.

### Units-Aware Feature Generation

One of `autofeat`'s most powerful capabilities is **dimensional analysis**: if you
supply physical units for each input feature, it will only generate dimensionally
consistent combinations. This dramatically reduces the number of candidates and
biases the search toward physically meaningful features.

```{code-cell} ipython3
if _autofeat_available:
    # Assign physical units to the Dow features based on column name suffixes
    unit_dict = {}
    for var in dow_feature_names[:X_train.shape[1]]:
        last_word = var.split(' ')[-1]
        if last_word == 'Flow':
            unit_dict[var] = 'L/s'
        elif last_word == 'Level':
            unit_dict[var] = 'm'
        elif last_word == 'DP':
            unit_dict[var] = 'bar'
        elif last_word == 'Pressure':
            unit_dict[var] = 'bar'
        elif last_word == 'Temperature':
            unit_dict[var] = 'K'
        else:
            unit_dict[var] = ''   # dimensionless

    # Build DataFrames so autofeat knows which feature is which
    dow_input_names = dow_feature_names[:X_train.shape[1]]
    X_train_df = pd.DataFrame(X_train, columns=dow_input_names)
    X_test_df  = pd.DataFrame(X_test,  columns=dow_input_names)
```

```{code-cell} ipython3
if _autofeat_available:
    transforms_extended = ['1/', 'exp', 'log', 'abs', 'sqrt', '^2', '^3', '1+', '1-', 'exp-']
    afreg_units = AutoFeatRegressor(
        verbose=1,
        feateng_steps=2,
        featsel_runs=1,
        transformations=transforms_extended,
        units=unit_dict,
    )
    afreg_units.fit(X_train_df, y_train)
```

```{code-cell} ipython3
if _autofeat_available:
    r2_af_units = afreg_units.score(X_test_df, y_test)
    print(f'autofeat (units-aware) test r²: {r2_af_units:.3f}')
    print('\nNew features (dimensionless combinations):')
    print(afreg_units.new_feat_cols_)
```

By constraining the search to dimensionless combinations, `autofeat` finds features
that correspond to physically interpretable ratios and products — the kind of
dimensionless groups (analogous to Reynolds or Damköhler numbers) that engineers
use to characterize process behavior.

### Practical Guidance

**When is symbolic regression worth using?**

| Situation | Recommendation |
|---|---|
| Dataset is small–medium ($n < 10{,}000$) and you want an interpretable model | `autofeat` is viable; budget 5–30 min of training time |
| Features have known physical units and you want dimensionless groups | Use `autofeat` with `units=` — the dimensional constraint keeps the feature count tractable |
| You need fast training (batch re-training, CI pipelines) | Use LASSO polynomial instead; faster and more predictable |
| Dataset is large ($n > 50{,}000$) or you have $>$50 input features | Symbolic regression becomes very slow; prefer tree-based models or neural networks |
| You need to deploy a model that other engineers can inspect | Symbolic/polynomial regression produces plain algebraic expressions — ideal for documentation |

The key caution is **train/test sensitivity**: because many feature combinations give
similar $r^2$ values on the training set, different random splits may yield very
different selected features. Always run a few different splits and check whether the
same features appear consistently before drawing conclusions about physical importance.

:::{exercise}
:label: ex-reg-autofeat-units

Apply units-aware `AutoFeatRegressor` to the Dow dataset.

1. Verify that the features in `afreg_units.new_feat_cols_` are dimensionless (every
   generated feature should be a dimensionless combination of input variables, given
   the unit assignments above).
2. Compare the test $r^2$ of the units-aware model with the non-units-aware model
   and the LASSO polynomial model from the earlier exercise. Which performs best?
3. Pick one of the selected features and write a one-sentence physical interpretation
   of what it represents (e.g., a ratio of flow rates, a product of pressures).
:::

## Summary

- **Linear feature combinations** (PCA, PLS, random projections) cannot improve a
  linear model beyond what the original features can achieve; nonlinear feature
  engineering is necessary to capture interactions and higher-order effects.

- **Polynomial features** with `PolynomialFeatures` generate all monomials up to a
  specified degree. Feature count grows as $\binom{p+d}{d}$, making unregularized fitting
  impossible beyond degree 2–3 for typical engineering datasets.

- **LASSO regularization** on a polynomial feature matrix simultaneously performs
  feature selection (setting many coefficients to zero) and regularization (preventing
  overfitting), often yielding better test accuracy than ordinary linear regression.

- **Symbolic regression** (`autofeat`) generates a broader library of nonlinear
  candidates (ratios, exponentials, roots) and selects the most predictive subset.
  Results are compact and interpretable but sensitive to train/test splits; physical
  units constraints (`units=`) improve robustness and chemical interpretability.

## Additional Reading

- Kanter, J. M. & Veeramachaneni, K. (2015), "Deep Feature Synthesis: Towards
  Automating Data Science Endeavors" — foundational automated feature engineering
- Orzechowski, P. et al. (2018), "Where are we now? A large benchmark study of recent
  symbolic regression methods" — survey of symbolic regression approaches
- Tibshirani, R. (1996), "Regression Shrinkage and Selection via the Lasso,"
  *Journal of the Royal Statistical Society B* — original LASSO paper
- scikit-learn User Guide:
  [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html),
  [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
