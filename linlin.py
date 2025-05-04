# -*- coding: utf-8 -*-
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------
# 1) LOAD & CLEAN
# ----------------------------------------------------------------
filename = 'data.csv'
print("loading:", filename)
df = pd.read_csv(filename).fillna(0)

# ----------------------------------------------------------------
# 2) EXTRACT & ENCODE TARGET
# ----------------------------------------------------------------
y_raw = df.pop("vital.status").values
le    = LabelEncoder()
y     = le.fit_transform(y_raw)    # 0/1

# ----------------------------------------------------------------
# 3) STANDARDIZE ALL ORIGINAL FEATURES
# ----------------------------------------------------------------
feature_names = df.columns.tolist()
scaler        = StandardScaler()
X_scaled      = scaler.fit_transform(df.values)

# ----------------------------------------------------------------
# 4) LINEAR REGRESSION FEATURE SELECTION
# ----------------------------------------------------------------
n_feats = 10
fs_lr   = LinearRegression()
fs_lr.fit(X_scaled, y)

importances = np.abs(fs_lr.coef_)
feat_scores = pd.Series(importances, index=feature_names) \
                .sort_values(ascending=False)

top_feats = feat_scores.index[:n_feats].tolist()
print(f"\nTop {n_feats} features by |LinearRegression coef|:")
print(feat_scores.head(n_feats))

# Subset & re-standardize selected features
X_sel        = df[top_feats].values
scaler2      = StandardScaler()
X_sel_scaled = scaler2.fit_transform(X_sel)

# ----------------------------------------------------------------
# 5) WRAP OLS INTO A “CLASSIFIER”
# ----------------------------------------------------------------
class LinRegClassifier:
    def __init__(self):
        self.lr = LinearRegression()

    def fit(self, X, y):
        self.lr.fit(X, y)
        return self

    def predict(self, X):
        y_cont = self.lr.predict(X)
        return (y_cont > 0.5).astype(int)

# ----------------------------------------------------------------
# 6) MANUAL STRATIFIED 10-FOLD CV & ACCURACY
# ----------------------------------------------------------------
cv             = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
fold_accuracies = []

for train_idx, test_idx in cv.split(X_sel_scaled, y):
    Xtr, Xte = X_sel_scaled[train_idx], X_sel_scaled[test_idx]
    ytr, yte = y[train_idx],         y[test_idx]

    clf    = LinRegClassifier().fit(Xtr, ytr)
    y_pred = clf.predict(Xte)
    fold_accuracies.append(accuracy_score(yte, y_pred))

fold_accuracies = np.array(fold_accuracies)
mean_acc        = fold_accuracies.mean()
sd_acc          = fold_accuracies.std()

print(f"\nMean Accuracy ± SD (10-fold CV): {mean_acc:.4f} ± {sd_acc:.4f}")
