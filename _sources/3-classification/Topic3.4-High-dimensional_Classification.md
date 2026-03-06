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

# High-dimensional Classification

:::{admonition} Learning Objectives
:class: tip

By the end of this chapter, you will be able to:
- Load and explore a real chemical dataset and identify class imbalance
- Apply an RBF kernel transformation to a high-dimensional feature matrix and explain its effect
- Implement a kernel-augmented SVM by hand and compare it to sklearn's `SVC`
- Use `GridSearchCV` to tune SVC hyperparameters on a held-out training set
- Evaluate classifier performance using accuracy, precision, recall, and confusion matrices
- Apply and interpret a depth-limited decision tree and read feature importances from the result
:::

The previous chapters introduced classification algorithms on simple two-dimensional
toy datasets. Here we apply these methods to a **real chemical engineering problem**:
predicting whether a given elemental combination will form a stable perovskite crystal
structure. This introduces challenges that toy data conceals — correlated features,
class imbalance, and the need to carefully separate training from test data before any
hyperparameter tuning.

Working with real data also requires thinking carefully about the **full modeling
pipeline**. In research and industrial settings, a common mistake is to use the test
set — even informally, by looking at results and adjusting the model — before final
evaluation. This form of **data leakage** produces optimistic accuracy estimates that
do not generalize. The correct procedure is:

1. Split data into training and test sets immediately.
2. Perform all preprocessing, feature selection, and hyperparameter tuning *on the
   training set only* (typically using cross-validation).
3. Evaluate the final model on the test set **once**.

We follow this protocol throughout this chapter.

## Perovskite Dataset

Perovskites (ABX$_3$ compounds) are a structurally versatile class of oxide and halide
materials with applications ranging from catalysis to photovoltaics. Whether a given
combination of A, B, and X elements adopts the perovskite structure depends on subtle
geometric and electronic factors. The dataset used here comes from Bartel et al. (2019),
who compiled 576 compounds with experimentally confirmed labels ($+1$ = perovskite,
$-1$ = non-perovskite) alongside eight numerical features:

| Feature | Description |
|---|---|
| `nA`, `nB`, `nX` | Formal oxidation states of A, B, X ions |
| `rA`, `rB`, `rX` | Ionic radii (Angstrom) |
| `t` | Goldschmidt tolerance factor |
| `tau` | New tolerance factor proposed in the paper |

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
df = pd.read_csv('data/perovskite_data.csv')
df.head(10)
```

```{code-cell} ipython3
feature_columns = ['nA', 'nB', 'nX', 'rA (Ang)', 'rB (Ang)', 'rX (Ang)', 't', 'tau']

X_perov = df[feature_columns].values
y_perov = df['exp_label'].values

print(f'Feature matrix shape: {X_perov.shape}')
print(f'Class distribution: {dict(zip(*np.unique(y_perov, return_counts=True)))}')
```

Note that the labels are $\pm 1$ rather than $0/1$. The classes are not perfectly
balanced — this matters when interpreting accuracy alone. A naive classifier that
predicts "non-perovskite" for every sample would still achieve moderate accuracy
simply by exploiting the class imbalance, without learning anything useful.
This is why precision, recall, and the confusion matrix are essential supplements
to accuracy for imbalanced binary classification tasks.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(y_perov, bins=[-1.5, -0.5, 0.5, 1.5], rwidth=0.6)
ax.set_xticks([-1, 1])
ax.set_xticklabels(['Non-perovskite (−1)', 'Perovskite (+1)'])
ax.set_ylabel('Count')
ax.set_title('Class distribution')
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
# map y in {-1, 1} to color indices {0, 1}
cidx = ((y_perov + 1) // 2).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(X_perov[:, 3], X_perov[:, 4], c=clrs[cidx], alpha=0.3)
axes[0].set_xlabel(feature_columns[3])
axes[0].set_ylabel(feature_columns[4])
axes[0].set_title('rA vs rB')

axes[1].scatter(X_perov[:, 4], X_perov[:, 5], c=clrs[cidx], alpha=0.3)
axes[1].set_xlabel(feature_columns[4])
axes[1].set_ylabel(feature_columns[5])
axes[1].set_title('rB vs rX')

axes[2].scatter(X_perov[:, 1], X_perov[:, 7], c=clrs[cidx], alpha=0.3)
axes[2].set_xlabel(feature_columns[1])
axes[2].set_ylabel(feature_columns[7])
axes[2].set_title('nB vs tau')

plt.tight_layout()
plt.show()
```

The scatter plots reveal that no single pair of features cleanly separates the two
classes. Points of both labels are heavily intermixed in all projections, although
`nB` vs `tau` (right panel) shows some tendency for perovskites to cluster at
higher $\tau$ values. This motivates the use of all features together and non-linear
decision boundaries — no simple cut in any two-dimensional projection will suffice.

This is also a reminder that high-dimensional intuition is hard: the eight features
together may define boundaries that are invisible in any pairwise projection but are
readily learned by a kernel SVM or decision tree operating on the full feature matrix.

:::{exercise}
:label: ex-cls-perov-scatter

Using the perovskite feature matrix `X_perov` and labels `y_perov`, produce scatter
plots for all pairs among the four features `rA (Ang)`, `rB (Ang)`, `t`, and `tau`.
Arrange them in a 4×4 grid (diagonal can show histograms). Identify which feature
pair appears to give the clearest visual separation between classes.
:::

---

## Kernel-based Classification

### The RBF Kernel Transformation

In Topic 3.2 we saw that kernels implicitly map data into a higher-dimensional feature
space where linear classifiers can find non-linear boundaries. Here we apply this idea
explicitly: compute the **radial basis function (RBF) kernel matrix** $K$ where

$$K_{ij} = \exp\!\left(-\gamma \|\vec{x}_i - \vec{x}_j\|^2\right)$$

Each row of $K$ is a new feature vector for sample $i$, encoding its similarity to
every other sample. The result is a $576 \times 576$ matrix, regardless of the original
number of features.

```{code-cell} ipython3
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score, confusion_matrix

X_kernel = rbf_kernel(X_perov, X_perov, gamma=0.02)
print(f'Kernel matrix shape: {X_kernel.shape}')
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(X_kernel[:, 3], X_kernel[:, 4], c=clrs[cidx], alpha=0.3)
axes[0].set_xlabel('$K_{i,3}$')
axes[0].set_ylabel('$K_{i,4}$')

axes[1].scatter(X_kernel[:, 4], X_kernel[:, 5], c=clrs[cidx], alpha=0.3)
axes[1].set_xlabel('$K_{i,4}$')
axes[1].set_ylabel('$K_{i,5}$')

axes[2].scatter(X_kernel[:, 1], X_kernel[:, 7], c=clrs[cidx], alpha=0.3)
axes[2].set_xlabel('$K_{i,1}$')
axes[2].set_ylabel('$K_{i,7}$')

axes[1].set_title('RBF Kernel ($\gamma = 0.02$) — projected coordinates')
plt.tight_layout()
plt.show()
```

### Manual SVM on Raw and Kernel-transformed Data

To build intuition, we first train our custom margin-loss SVM (from Topic 3.2) on
just two features, both before and after the kernel transform, and compare accuracy.

```{code-cell} ipython3
from scipy.optimize import minimize

def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def linear_classifier(X, w):
    p = add_intercept(X) @ w
    return np.where(p > 0, 1, -1)

def regularized_cost(w, X, y, alpha=1):
    Xb = add_intercept(X) @ w
    cost = np.sum(np.maximum(0, 1 - y * Xb))
    cost += alpha * np.linalg.norm(w[1:], 2)
    return cost
```

```{code-cell} ipython3
# SVM on original 2-feature data (rA, rB)
w_guess = np.array([-10., -4., -10.])
result = minimize(regularized_cost, w_guess, args=(X_perov[:, 3:5], y_perov, 1))
w_raw = result.x

pred_raw = linear_classifier(X_perov[:, 3:5], w_raw)
print(f'Accuracy (raw 2 features):    {accuracy_score(y_perov, pred_raw):.3f}')
```

```{code-cell} ipython3
# SVM on kernel-transformed 2 features
result_k = minimize(regularized_cost, w_guess, args=(X_kernel[:, 3:5], y_perov, 1))
w_kernel = result_k.x

pred_kernel = linear_classifier(X_kernel[:, 3:5], w_kernel)
print(f'Accuracy (kernel 2 features): {accuracy_score(y_perov, pred_kernel):.3f}')
```

The kernel-transformed version substantially outperforms the raw linear classifier
on the same two features, demonstrating that the implicit higher-dimensional mapping
allows a linear boundary in kernel space to act as a non-linear boundary in the
original feature space.

Note that both models are still trained and evaluated on the *same* data here —
we are illustrating the effect of the kernel, not measuring generalization.
True generalization requires a held-out test set, which we use in the next section.

### scikit-learn SVC

The `sklearn.svm.SVC` handles the kernel transformation internally and is far more
efficient than our manual approach. Let's compare results using 2 features vs. all 8:

```{code-cell} ipython3
from sklearn.svm import SVC

svc_2feat = SVC(kernel='rbf', gamma=100, C=1000)
svc_2feat.fit(X_perov[:, 3:5], y_perov)
print(f'SVC (2 features, train): {svc_2feat.score(X_perov[:, 3:5], y_perov):.3f}')

svc_all = SVC(kernel='rbf', gamma=100, C=1000)
svc_all.fit(X_perov, y_perov)
print(f'SVC (all features, train): {svc_all.score(X_perov, y_perov):.3f}')
```

Both achieve near-perfect *training* accuracy — a warning sign of overfitting.
With $\gamma = 100$ and $C = 1000$, the RBF kernel is extremely tight, creating
tiny decision regions around individual training points. This is classic overfitting:
the model has memorized the training data rather than learned generalizable structure.
We must evaluate on held-out data and tune hyperparameters properly.

### Hyperparameter Optimization with GridSearchCV

We perform a proper train/test split first, then search only within the training set.
`GridSearchCV` trains a model for every combination of hyperparameters in the grid,
using $k$-fold cross-validation to estimate generalization error. Because this entire
search happens inside the training set, the test set remains completely unseen until
we call `.score()` on the best estimator:

```{code-cell} ipython3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle

X_train, X_test, y_train, y_test = train_test_split(
    X_perov, y_perov, test_size=0.33, random_state=42)

X_train, y_train = shuffle(X_train, y_train, random_state=42)

sigmas = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100])
gammas = 1. / (2 * sigmas**2)
alphas = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
Cs = 1 / alphas

svc = SVC(kernel='rbf')
svc_search = GridSearchCV(svc, {'C': Cs, 'gamma': gammas}, cv=3)
svc_search.fit(X_train, y_train)

print(f'Best params:   {svc_search.best_params_}')
print(f'CV accuracy:   {svc_search.best_score_:.3f}')
```

```{code-cell} ipython3
best_svc = svc_search.best_estimator_
print(f'Test accuracy: {best_svc.score(X_test, y_test):.3f}')
```

```{code-cell} ipython3
y_pred_svc = best_svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred_svc)

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, cbar=False, fmt='d', ax=ax)
ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')
ax.set_title('SVC — Test Set Confusion Matrix')
plt.tight_layout()
plt.show()
```

### Demonstration: Precision and Recall

Accuracy alone is misleading when classes are imbalanced. Precision and recall give
a more complete picture. Recall the confusion matrix entries:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **True Positive** | TP | FN |
| **True Negative** | FP | TN |

From these counts we define:

- **Accuracy** = $\frac{TP + TN}{TP + TN + FP + FN}$ — fraction of all predictions correct
- **Precision** = $\frac{TP}{TP + FP}$ — of all predicted positives, how many are correct?
- **Recall** = $\frac{TP}{TP + FN}$ — of all true positives, how many did we catch?
- **F1 score** = $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ — harmonic mean of precision and recall

In materials screening, **recall** is often the priority: missing a promising
perovskite candidate (false negative) may be more costly than following up on a
false positive in the lab. Choosing which metric to optimize is a domain decision,
not a modeling one.

```{code-cell} ipython3
tn, fp, fn, tp = cm.ravel()

accuracy  = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * precision * recall / (precision + recall)

print(f'Accuracy:  {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall:    {recall:.3f}')
print(f'F1 score:  {f1:.3f}')
```

:::{exercise}
:label: ex-cls-perov-knn

Apply `KNeighborsClassifier` (from sklearn) to the full perovskite feature matrix
using the same train/test split (`X_train`, `X_test`, `y_train`, `y_test`).
Use `GridSearchCV` with 3-fold cross-validation to search over
$k \in \{3, 5, 10, 20, 50\}$. Print the best $k$, the CV accuracy, and the test
accuracy. Compare to the best SVC result above.
:::

---

## Decision Trees on the Perovskite Dataset

Decision trees work directly on the original feature space — no kernel transformation
needed, and no assumption of Gaussian distributions or linear separability. They are
also interpretable: we can read the learned rules directly from the tree diagram and
extract quantitative feature importance scores from the trained model.

### Overfitting and Depth Control

```{code-cell} ipython3
from sklearn.tree import DecisionTreeClassifier, plot_tree

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

cm_train = confusion_matrix(y_train, dtree.predict(X_train))
cm_test  = confusion_matrix(y_test,  dtree.predict(X_test))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_train, annot=True, cbar=False, linewidth=0.5, ax=axes[0], fmt='d')
axes[0].set_xlabel('Predicted Class')
axes[0].set_ylabel('True Class')
axes[0].set_title('Unconstrained Tree — Training Set')

sns.heatmap(cm_test, annot=True, cbar=False, linewidth=0.5, ax=axes[1], fmt='d')
axes[1].set_xlabel('Predicted Class')
axes[1].set_ylabel('True Class')
axes[1].set_title('Unconstrained Tree — Test Set')
plt.tight_layout()
plt.show()
```

The unconstrained tree memorizes the training data (perfect training accuracy) but
generalizes less well to the test set. This is the hallmark of high-variance
overfitting: the model has grown enough branches to perfectly partition every
training point, including those that are noise or outliers.
Limiting `max_depth` acts as regularization, forcing the tree to find broader
rules that capture the dominant structure rather than individual training samples:

```{code-cell} ipython3
dtree3 = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree3.fit(X_train, y_train)

cm_train3 = confusion_matrix(y_train, dtree3.predict(X_train))
cm_test3  = confusion_matrix(y_test,  dtree3.predict(X_test))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_train3, annot=True, cbar=False, linewidth=0.5, ax=axes[0], fmt='d')
axes[0].set_xlabel('Predicted Class')
axes[0].set_ylabel('True Class')
axes[0].set_title('Depth-3 Tree — Training Set')

sns.heatmap(cm_test3, annot=True, cbar=False, linewidth=0.5, ax=axes[1], fmt='d')
axes[1].set_xlabel('Predicted Class')
axes[1].set_ylabel('True Class')
axes[1].set_title('Depth-3 Tree — Test Set')
plt.tight_layout()
plt.show()

print(f'Depth-3 test accuracy: {dtree3.score(X_test, y_test):.3f}')
```

The depth-3 tree has slightly lower training accuracy but comparable (or better) test
accuracy — a clear improvement in the bias-variance trade-off. This illustrates
a general principle: a simpler model that captures the dominant signal often
generalizes better than a complex model that fits every detail of the training data.

### Feature Importance

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 5))
plot_tree(dtree3, filled=True, rounded=True, ax=ax,
          class_names=['Non-perovskite', 'Perovskite'],
          feature_names=feature_columns)
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
importances = dtree3.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(feature_columns)), importances[sorted_idx])
ax.set_xticks(range(len(feature_columns)))
ax.set_xticklabels([feature_columns[i] for i in sorted_idx], rotation=30, ha='right')
ax.set_ylabel('Gini importance')
ax.set_title('Feature importances — depth-3 decision tree')
plt.tight_layout()
plt.show()
```

:::{note}
The feature `tau` (index 7) consistently appears near the root of the decision tree
and carries the highest Gini importance. This aligns with the finding in Bartel et al.
(2019) that $\tau$ is a better predictor of perovskite stability than the traditional
Goldschmidt tolerance factor $t$. The ability to read feature importances directly
from a decision tree is one of its most practically valuable properties in materials
science and chemical engineering applications.
:::

:::{exercise}
:label: ex-cls-tree-rf

Import `RandomForestClassifier` from `sklearn.ensemble` and train it on the
perovskite training set with `n_estimators=100` and `random_state=42`.
Print the test accuracy and compare it to the depth-3 decision tree above.
Then plot the feature importances from the random forest alongside those from
the depth-3 tree and note any differences in the relative ranking of `tau` and `t`.
:::

---

## Summary

- The perovskite dataset illustrates the full classification workflow on real data:
  exploratory scatter plots, train/test splitting, hyperparameter search on the
  training set, and evaluation on held-out data.

- **Kernel transformation** maps the original feature space into a higher-dimensional
  similarity space, enabling a linear classifier to find non-linear boundaries.
  The RBF kernel with a well-chosen $\gamma$ can dramatically improve accuracy over
  a raw linear SVM on the same features. The parameter $\gamma$ controls the width
  of each Gaussian: large $\gamma$ creates narrow kernels sensitive only to very
  nearby points (risk of overfitting); small $\gamma$ creates broad kernels that
  average over large neighborhoods (risk of underfitting).

- **`GridSearchCV`** automates the search over $C$ and $\gamma$ using cross-validation
  on the training set only — the test set is never touched until final evaluation.

- **Accuracy** is a misleading metric for imbalanced datasets; **precision** and
  **recall** (and the combined F1 score) give a more complete picture of classifier
  performance.

- **Decision trees** on real data are highly prone to overfitting when unconstrained,
  but `max_depth` regularization recovers competitive generalization. Their
  interpretability — readable splitting rules and feature importance scores — makes
  them especially useful in science and engineering applications where understanding
  *why* a model makes a prediction matters as much as accuracy.

## Additional Reading

- Bartel, C. J. et al. (2019), "New tolerance factor to predict the stability of
  perovskite oxides and halides," *Science Advances* 5(2), eaav0693 — the source
  paper for this dataset
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Ch. 12
  (SVMs and kernels), Ch. 9 (Decision Trees) —
  [free PDF](https://hastie.su.domains/ElemStatLearn/)
- scikit-learn User Guide:
  [SVM](https://scikit-learn.org/stable/modules/svm.html),
  [GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html),
  [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
