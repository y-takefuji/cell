# -*- coding: utf-8 -*-
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import make_scorer, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) LOAD & CLEAN
filename = 'data.csv'
print("loading:", filename)
df = pd.read_csv(filename).fillna(0)

# 2) EXTRACT & ENCODE TARGET
y_raw = df.pop("vital.status").values
le    = LabelEncoder()
y     = le.fit_transform(y_raw)    # e.g. 0/1 classes

# 3) STANDARDIZE ALL ORIGINAL FEATURES
feature_names = df.columns.tolist()
scaler        = StandardScaler()
X_scaled      = scaler.fit_transform(df.values)
# shape == (n_samples, n_features)

# 4) LINEAR REGRESSION FEATURE SELECTION
n_feats = 10
fs_lr   = LinearRegression()
fs_lr.fit(X_scaled, y)
coefs      = fs_lr.coef_  # shape = (n_features,)
feat_scores = pd.Series(np.abs(coefs), index=feature_names)\
                .sort_values(ascending=False)

top_feats = feat_scores.index[:n_feats].tolist()
print(f"\nTop {n_feats} features by |LinearRegression coef|:")
print(feat_scores.head(n_feats))

# 5) SUBSET DATA TO THOSE TOP FEATURES & RE‐STANDARDIZE
X_sel        = df[top_feats].values
scaler2      = StandardScaler()
X_sel_scaled = scaler2.fit_transform(X_sel)

# 6) LINEAR REGRESSION “CLASSIFIER” + 10‐FOLD STRATIFIED CV
lr_model = LinearRegression()

# custom scorer: threshold at 0.5
def lr_acc(y_true, y_pred_cont):
    y_pred = (y_pred_cont > 0.5).astype(int)
    return accuracy_score(y_true, y_pred)

acc_scorer = make_scorer(lr_acc)

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=0
)

scores = cross_val_score(
    lr_model,
    X_sel_scaled,
    y,
    cv=cv,
    scoring=acc_scorer
)

mean_acc = scores.mean()
sd_acc   = scores.std()
print(f"\nMean Accuracy ± SD (10‐fold CV – LinearRegression+thr): "
      f"{mean_acc:.4f} ± {sd_acc:.4f}")
