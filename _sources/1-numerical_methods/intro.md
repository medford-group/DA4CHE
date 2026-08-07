# Numerical Methods

Data analytics starts with numerical foundations. Before fitting models or
interpreting results, you need fluency with the computational tools and
mathematical structures that underlie every machine learning algorithm.

This module builds those foundations using Python. We begin with the core
scientific Python stack — NumPy, SciPy, Matplotlib — and work up through
linear algebra, least-squares regression, and numerical optimization. Chemical
engineering examples (Arrhenius kinetics, spectroscopic data, reaction rate
fitting) illustrate each concept throughout.

## Topics

- **Topic 1.1 — Python Basics**: NumPy arrays, vectorized operations, and the
  scientific Python ecosystem. Covers array creation, slicing, broadcasting,
  and common pitfalls (copy vs. view semantics).

- **Topic 1.2 — Linear Algebra**: Matrix operations, solving linear systems,
  orthogonalization (Gram–Schmidt), rank, and condition numbers. Foundational
  for understanding regression and dimensionality reduction.

- **Topic 1.3 — Linear Regression**: Least-squares fitting as a linear algebra
  problem. Normal equations, polynomial features, residual analysis, and
  scikit-learn's `LinearRegression` interface.

- **Topic 1.4 — Numerical Optimization**: Gradient descent, quasi-Newton
  methods (L-BFGS-B), and constrained optimization via `scipy.optimize`.
  Explores loss surfaces, convergence behavior, and regularization penalties.

- **Topic 1.5 — Nonlinear Parameter Estimation**: Least squares as maximum
  likelihood, Hessian-based confidence intervals via `autograd`, parameter
  correlations and confidence ellipses, and sloppy (unidentifiable) parameters
  when model complexity outruns the data.
