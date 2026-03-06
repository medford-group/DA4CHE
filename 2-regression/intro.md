# Regression

Regression — predicting a continuous output from input features — is one of the
most widely used tools in chemical engineering data analysis. Predicting product
yield from process variables, estimating reaction rates from spectroscopic data,
and building surrogate models for expensive simulations all reduce to regression.

This module takes a machine-learning perspective: we focus on how to choose
models with appropriate complexity, how to validate them honestly, and how to
handle the high-dimensional datasets that arise in modern process data. The
**Dow Chemical distillation column dataset** appears throughout as a running
industrial example.

## Topics

- **Topic 2.1 — Non-parametric Models**: Kernel regression, k-nearest neighbor
  regression, and spline interpolation. Introduces the bias–variance tradeoff
  without parametric assumptions.

- **Topic 2.2 — Model Validation**: Train/test splits, k-fold cross-validation,
  bootstrap resampling, and outlier sensitivity. Emphasizes honest estimation of
  generalization error.

- **Topic 2.3 — Complexity Optimization**: Regularization (ridge, LASSO, elastic
  net), hyperparameter selection via BIC/AIC and cross-validation, and the danger
  of data leakage.

- **Topic 2.4 — High-dimensional Data**: Feature extraction, correlation analysis,
  and scaling strategies for datasets with many input variables.

- **Topic 2.5 — High-dimensional Regression**: Multiple linear regression,
  principal component regression (PCR), and partial least squares (PLS) for
  datasets where the number of features rivals or exceeds the number of samples.
