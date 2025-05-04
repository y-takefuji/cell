# -*- coding: utf-8 -*-
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.ensemble           import RandomForestClassifier
from sklearn.model_selection    import StratifiedKFold, cross_val_score

# 1) LOAD & CLEAN
filename = 'data.csv'
print("loading:", filename)
df = pd.read_csv(filename).fillna(0)

# 2) EXTRACT & ENCODE TARGET
y_raw = df.pop("vital.status").values
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 3) LIST FEATURE NAMES
feature_names = df.columns.tolist()

# 4) RANK FEATURES BY SPEARMAN’S RHO WITH THE TARGET
spearman_corrs = df.apply(lambda col: spearmanr(col, y)[0])
feat_scores   = spearman_corrs.abs().sort_values(ascending=False)

top_n    = 10
top_feats = feat_scores.index[:top_n].tolist()

print(f"\nTop {top_n} original features by |Spearman correlation|:")
print(feat_scores.head(top_n))

# 5) SUBSET TO TOP FEATURES
X_sel = df[top_feats].values

# 6) STANDARDIZE SELECTED FEATURES FOR RF
scaler = StandardScaler()
X_sel_scaled = scaler.fit_transform(X_sel)

# 7) RANDOM FOREST + 10-FOLD STRATIFIED CV
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=54)
scores = cross_val_score(rf, X_sel_scaled, y, cv=cv, scoring="accuracy")

mean_acc = scores.mean()
sd_acc   = scores.std()
print(f"\nMean Accuracy ± SD (10-fold CV): {mean_acc:.4f} ± {sd_acc:.4f}")
