# -*- coding: utf-8 -*-
import warnings
import logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) LOAD & CLEAN DATA
filename = 'data.csv'
print("loading:", filename)
df = pd.read_csv(filename).fillna(0)

# 2) EXTRACT TARGET VARIABLE
y = df.pop("vital.status").values  # Assuming 'vital.status' is the target variable

# 3) STANDARDIZE ALL FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)

# --- 4) SEQUENTIAL FEATURE SELECTION (SFS) ---

n_feats = 10
rf_sfs = RandomForestClassifier(
    n_estimators=50,  # Reduced number of estimators for faster performance during SFS
    n_jobs=-1,
    random_state=0
)

# Inner CV for the selector
inner_cv = StratifiedKFold(
    n_splits=5,  # 5-fold CV for feature selection
    shuffle=True,
    random_state=42
)

# Forward selection
sfs_forward = SequentialFeatureSelector(
    estimator=rf_sfs,
    n_features_to_select=n_feats,
    direction='forward',
    scoring='accuracy',
    cv=inner_cv,
    n_jobs=-1  # Use all available cores
)
sfs_forward.fit(X_scaled, y)
mask_fwd = sfs_forward.get_support()
top_fwd = [col for col, keep in zip(df.columns, mask_fwd) if keep]

print(f"\nForward-SFS selected {len(top_fwd)} features:")
print(top_fwd)

# 5) SUBSET & RE‐STANDARDIZE SELECTED FEATURES
X_sel = df[top_fwd].values
X_sel_scaled = scaler.fit_transform(X_sel)  # Rescale the selected features

# 6) RANDOM FOREST + 10-FOLD STRATIFIED CV on the selected features
rf_final = RandomForestClassifier(
    n_estimators=100,  # Full number of trees for modeling
    n_jobs=-1,
    random_state=0
)

cv_outer = StratifiedKFold(
    n_splits=10,  # 10-fold CV for final evaluation
    shuffle=True,
    random_state=54
)
scores = cross_val_score(
    rf_final,
    X_sel_scaled,
    y,
    cv=cv_outer,
    scoring="accuracy"
)

print(f"\nMean Accuracy ± SD (10-fold CV): {scores.mean():.4f} ± {scores.std():.4f}")
