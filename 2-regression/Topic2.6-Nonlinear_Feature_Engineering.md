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
- Describe how automated model discovery methods extend feature generation to searching for the functional form of a model, and identify representative tools used in science and in chemical engineering.
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

This is the same preparation used in Topics 2.4 and 2.5, repeated here so the chapter
runs on its own. The `is_real_and_finite` helper is needed because the raw spreadsheet
contains text placeholders in some cells, so `real_rows` keeps only the rows where every
column is a usable number. The column slicing selects the process variables as inputs
(`:-5`) and the impurity as the target (`-3`), and the features are standardized to zero
mean and unit variance. Standardization matters more than usual in this chapter: we are
about to multiply features together, and a product of two variables with very different
magnitudes would otherwise dominate the feature matrix for reasons of units rather than
physics.

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
is $\binom{40 + 2}{2} - 1 = 860$. At degree 3 it exceeds 12,000, which is larger than many
training sets. We subsample to avoid memory issues during the fitting step below. In practice you
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
subset of the polynomial features.

To see the effect clearly we fit three models on the same train/test split, changing one
thing at a time. The first is a plain linear regression on the original 40 features,
which serves as the baseline to beat:

```{code-cell} ipython3
# Baseline: plain linear regression on original features
linreg = LinearRegression()
linreg.fit(X_train, y_train)
r2_base_train = linreg.score(X_train, y_train)
r2_base_test  = linreg.score(X_test,  y_test)
print(f'Linear regression:  train r² = {r2_base_train:.3f},  test r² = {r2_base_test:.3f}')
```

The second uses the 860 polynomial features but no regularization. With that many
features relative to the number of training samples, we expect the training $r^2$ to
improve substantially while the test $r^2$ gets worse, which is the signature of
overfitting from Topic 2.3:

```{code-cell} ipython3
# Unregularized: polynomial features + ordinary least squares — should overfit
linreg_poly = LinearRegression()
linreg_poly.fit(X_poly_train, y_train)
r2_poly_train = linreg_poly.score(X_poly_train, y_train)
r2_poly_test  = linreg_poly.score(X_poly_test,  y_test)
print(f'Poly + OLS:         train r² = {r2_poly_train:.3f},  test r² = {r2_poly_test:.3f}')
```

The third uses the same polynomial features with a LASSO penalty, and also counts how
many coefficients survive:

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
output column, so a sparse model can be read as an equation. The cell below asks for
those names, finds the coefficients that LASSO did not set to zero, and sorts them by
magnitude so that the most influential terms appear first:

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

These are the nonlinear combinations that `autofeat` constructed and judged predictive.
It is worth also inspecting `good_cols_`, which is a different quantity: `new_feat_cols_`
lists only the *newly generated* features, while `good_cols_` lists every feature the
final model actually uses, including whichever of the original 40 variables survived
selection:

```{code-cell} ipython3
if _autofeat_available:
    # All features (original + new) retained after final selection
    print('All selected features (used in model):')
    print(afreg.good_cols_)
```

The distinction matters when interpreting the model, and it also explains the
overfitting comparison at the end of this section: scoring the fitted `AutoFeatRegressor`
uses only `good_cols_`, whereas transforming the data ourselves returns the full
generated matrix.

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

Supplying those units requires knowing what each column represents. The Dow column names
end in words like "Flow", "Level", or "Temperature", so the cell below assigns units by
reading that suffix and marks anything unrecognized as dimensionless. This is a
convenient shortcut rather than a rigorous one: a mislabeled column would silently
constrain the search in the wrong way, so in your own work it is worth checking the unit
assignments against the process documentation. `autofeat` also needs the feature names
themselves, which is why the arrays are wrapped in DataFrames:

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

With units supplied, we can afford a larger pool of transformations than before, because
the dimensional constraint will discard most of the combinations they generate. The
`units=` argument is the only substantive change from the earlier fit:

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

## Beyond Feature Selection: Automated Model Discovery

It is worth stepping back to see what `autofeat` does and does not do. It generates a
large library of candidate nonlinear terms and then selects a useful subset, but the
model itself is still a **linear combination of those terms**. The functional form is
therefore fixed by the transformations we allowed in the search: we chose the pool, and
the algorithm chose from it. The same is true of the LASSO polynomial models earlier in
this chapter, and of the regularized models in
[Complexity and Optimization](Topic2.3-Complexity_Optimization).

**Model discovery** (often called **symbolic regression** in its more general form)
removes that restriction and searches over the structure of the equation itself, with
the goal of returning a compact expression that a person can read, interpret, and
check against physical intuition. The difficulty is that the space of possible
equations is astronomically large, so every practical method imposes strong priors on
what a reasonable model looks like. This is the same principle as regularization,
applied to the *structure* of the model rather than to the size of its coefficients.
Three examples are worth knowing:

* **AI-Feynman** ([Udrescu & Tegmark, 2020](https://doi.org/10.1126/sciadv.aay2631),
  *Science Advances*) exploits properties that physical laws tend to possess:
  dimensional consistency, symmetry, separability into independent parts, and
  smoothness. A neural network is fit to the data first, not as the final model, but
  as a probe that can be queried to test for these properties, which then break the
  problem into simpler sub-problems recursively. The method recovered all 100 equations
  from the *Feynman Lectures on Physics*, where the best previously available software
  found 71. The code is [openly available](https://github.com/SJ001/AI-Feynman). Note
  that the dimensional-consistency idea is exactly the `units=` constraint we used with
  `autofeat` above, applied to a much larger search.

* **AI-DARWIN** ([Chakraborty, Sivaram & Venkatasubramanian,
  2021](https://doi.org/10.1016/j.compchemeng.2021.107470), *Computers & Chemical
  Engineering*) is aimed specifically at chemical engineering problems, where data are
  typically limited and noisy rather than abundant. It uses a genetic algorithm to
  evolve candidate nonlinear terms, statistical testing to retain only the terms the
  data actually support, and nonlinear regression to fit the remaining parameters. The
  result is a mechanistic-looking model rather than a black box, which is the
  motivation behind the broader argument that hybrid models are often more appropriate
  than purely data-driven ones in our field.

* **HyMech** ([Rossi, Bezzo & Barolo,
  2026](https://doi.org/10.1016/j.compchemeng.2026.109634), *Computers & Chemical
  Engineering*) builds directly on the AI-DARWIN engine and adds prior process
  knowledge to the search through physics-informed pools of allowed functions and
  variables. It addresses a situation that is common in practice: you already have a
  first-principles model, it does not quite match the plant data, and you want to know
  *where the model structure is deficient*. Conventional hybrid modeling patches such a
  mismatch with a black-box correction term, which improves predictions without
  explaining anything; HyMech instead searches for an interpretable equation for the
  correction itself.

These methods are directly relevant to industrial practice, and in fact to the dataset
used throughout this course. The Dow impurity data was originally posed by researchers
at Dow as a "data challenge" problem: given more than 40 process variables from an
integrated multi-column process, identify which of them actually drive the impurity
measured at the primary column outlet, and build an inferential sensor to predict it.
Identifying the relevant variables and producing a model that an engineer can inspect
and trust is precisely the goal of the methods above, and it is the same problem we
have been working on in this chapter with polynomial features and `autofeat`. Work in
this direction is ongoing, including methods developed jointly with industry such as
[SyMANTIC](https://doi.org/10.1021/acs.iecr.4c03503) (Muthyala et al., 2025, *Ind.
Eng. Chem. Res.*), which combines mutual-information-based feature screening with
sparse regression to keep the search tractable.

Full model discovery is beyond the scope of this course, and these tools require more
care and computational effort than the methods we have used here. They are worth
knowing about, however, because they represent the natural endpoint of the progression
in this chapter: from choosing features by hand, to generating and selecting them
automatically, to searching for the form of the model itself.

:::{exercise}
:label: ex-reg-search-space

Estimate why unrestricted equation search is impractical, and how much a physical
prior helps.

1. Consider building expressions as binary trees with $n$ internal nodes. Each internal
   node is one of $k$ binary operators ($+, -, \times, \div$, so $k = 4$) and each of
   the $n+1$ leaves is one of $p$ input variables. The number of tree shapes is the
   Catalan number $C_n = \binom{2n}{n}/(n+1)$, so the total count of expressions is
   $C_n \, k^n \, p^{\,n+1}$. Using `math.comb`, compute and print this count for
   $p = 40$ (the Dow features) and $n = 1, 2, \ldots, 6$.
2. Plot the count versus $n$ on a log scale. At what $n$ does the count exceed the
   roughly $10^3$ candidate features that `PolynomialFeatures` produced at degree 2?
3. Now suppose dimensional analysis rules out all but 1% of these expressions as
   dimensionally inconsistent. Re-plot the surviving count. Does the constraint change
   the *rate* of growth, or only the prefactor? What does your answer imply about
   relying on dimensional analysis alone to make the search tractable?
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

- **Automated model discovery** goes one step further by searching over the functional
  form of the model rather than selecting from a fixed pool of generated terms. Tools
  such as AI-Feynman, AI-DARWIN, and HyMech make the search tractable by imposing
  physical priors (dimensional consistency, symmetry, or an existing first-principles
  model), which is regularization applied to model structure.

## Additional Reading

- Kanter, J. M. & Veeramachaneni, K. (2015), "Deep Feature Synthesis: Towards
  Automating Data Science Endeavors" — foundational automated feature engineering
- Orzechowski, P. et al. (2018), "Where are we now? A large benchmark study of recent
  symbolic regression methods" — survey of symbolic regression approaches
- Tibshirani, R. (1996), "Regression Shrinkage and Selection via the Lasso,"
  *Journal of the Royal Statistical Society B* — original LASSO paper
- Udrescu, S.-M. & Tegmark, M. (2020), "AI Feynman: A physics-inspired method for
  symbolic regression," *Science Advances* 6(16), eaay2631 —
  [doi:10.1126/sciadv.aay2631](https://doi.org/10.1126/sciadv.aay2631)
- Chakraborty, A., Sivaram, A. & Venkatasubramanian, V. (2021), "AI-DARWIN: A first
  principles-based model discovery engine using machine learning," *Computers &
  Chemical Engineering* 154, 107470 —
  [doi:10.1016/j.compchemeng.2021.107470](https://doi.org/10.1016/j.compchemeng.2021.107470)
- Rossi, L., Bezzo, F. & Barolo, M. (2026), "HyMech: AI-driven framework for
  physics-informed discovery of interpretable hybrid models," *Computers & Chemical
  Engineering* 210, 109634 —
  [doi:10.1016/j.compchemeng.2026.109634](https://doi.org/10.1016/j.compchemeng.2026.109634)
- Muthyala, M. R., Sorourifar, F., Peng, Y. & Paulson, J. A. (2025), "SyMANTIC: An
  efficient symbolic regression method for interpretable and parsimonious model
  discovery," *Industrial & Engineering Chemistry Research* 64(6), 3354–3369 —
  [doi:10.1021/acs.iecr.4c03503](https://doi.org/10.1021/acs.iecr.4c03503)
- scikit-learn User Guide:
  [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html),
  [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
