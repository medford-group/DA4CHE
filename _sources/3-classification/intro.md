# Classification

Many chemical engineering decisions are inherently discrete: a material is
stable or unstable, a sensor reading indicates normal operation or a fault, a
candidate molecule is active or inactive. Classification algorithms learn
decision boundaries that map continuous inputs to discrete class labels.

This module develops classification from first principles, moving from simple
geometric boundaries to flexible non-parametric models and finally to methods
that scale to high-dimensional feature spaces. The **perovskite stability
dataset** — a real materials-science problem — is used throughout to ground
the methods in a chemical engineering context.

## Topics

- **Topic 3.1 — Classification Basics**: Problem setup, decision boundaries,
  evaluation metrics (accuracy, precision, recall, F1, ROC/AUC), and practical
  considerations for class imbalance.

- **Topic 3.2 — Generalized Linear Models**: Logistic regression, the perceptron,
  support vector machines, and kernel methods. Covers loss functions, margin
  maximization, and the role of regularization.

- **Topic 3.3 — Alternate Classification Models**: k-nearest neighbors, Naive
  Bayes, decision trees, and random forests. Introduces ensemble methods and the
  bias–variance tradeoff in the classification setting.

- **Topic 3.4 — High-dimensional Classification**: Applying SVM and tree-based
  methods to the perovskite dataset. Covers GridSearchCV for hyperparameter
  tuning, feature importances, and the curse of dimensionality.
