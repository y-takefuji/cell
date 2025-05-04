# -*- coding: utf-8 -*-
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) LOAD & CLEAN
filename = 'data.csv'
print("loading:", filename)
df = pd.read_csv(filename).fillna(0)

# 2) EXTRACT & ENCODE TARGET
y_raw = df.pop("vital.status").values
le    = LabelEncoder()
y     = le.fit_transform(y_raw)

# 3) STANDARDIZE ALL ORIGINAL FEATURES
feature_names = df.columns.tolist()
scaler        = StandardScaler()
X_scaled      = scaler.fit_transform(df.values)

# 4) RANDOM-FOREST FEATURE SELECTION
n_feats = 10

# 4a) Fit a RF on all features to get importances
fs_rf = RandomForestClassifier(n_estimators=100,
                               n_jobs=-1,
                               random_state=0)
fs_rf.fit(X_scaled, y)

# 4b) Rank features by importance
importances = fs_rf.feature_importances_
feat_scores = pd.Series(importances, index=feature_names) \
                .sort_values(ascending=False)

top_feats = feat_scores.index[:n_feats].tolist()

print(f"\nTop {n_feats} features by RF importance:")
print(feat_scores.head(n_feats))

# 5) SUBSET DATA TO THOSE TOP FEATURES & RE-STANDARDIZE
X_sel        = df[top_feats].values
scaler2      = StandardScaler()
X_sel_scaled = scaler2.fit_transform(X_sel)

# 6) RANDOM FOREST + 10-FOLD STRATIFIED CV
rf = RandomForestClassifier(n_estimators=100,
                            n_jobs=-1,
                            random_state=0)
cv = StratifiedKFold(n_splits=10,
                     shuffle=True,
                     random_state=0)
scores = cross_val_score(rf,
                         X_sel_scaled,
                         y,
                         cv=cv,
                         scoring="accuracy")

mean_acc = scores.mean()
sd_acc   = scores.std()
print(f"\nMean Accuracy ± SD (10-fold CV): {mean_acc:.4f} ± {sd_acc:.4f}")
