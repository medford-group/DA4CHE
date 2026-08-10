#!/usr/bin/env python3
"""
Multi-class confusion matrix figure for Topic 3.1 (Classification Basics).

Replaces the scikit-learn documentation's iris example figure with a
version generated in the book's own style, for provenance. A deliberately
simple classifier (logistic regression on the two sepal features only) is
used so the matrix shows realistic off-diagonal confusion between
versicolor and virginica — the point the surrounding prose discusses.

Saves: confusion_matrix_iris.png
       (this directory and ../../3-classification/images/)

Run from settings/helper_scripts/:
    python plot_confusion_matrix_iris.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
IMGDIR = HERE / '../../3-classification/images'
plt.style.use(HERE / '../plot_style.mplstyle')

iris = load_iris()
X = iris.data[:, :2]          # sepal length + width only (harder problem)
y = iris.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0,
                                          stratify=y)
clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
cm = confusion_matrix(y_te, clf.predict(X_te))
print('confusion matrix:\n', cm)
assert cm.sum() == len(y_te)
assert (cm.sum() - np.trace(cm)) >= 4, 'expected visible off-diagonal confusion'

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cbar=True, linewidth=0.5, cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
ax.set_xlabel('Predicted class')
ax.set_ylabel('True class')
ax.set_title('Confusion matrix — iris, sepal features only')
plt.tight_layout()

for out in (HERE / 'confusion_matrix_iris.png', IMGDIR / 'confusion_matrix_iris.png'):
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')
