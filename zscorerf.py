# -*- coding: utf-8 -*-
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy  as np
import pandas as pd

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) LOAD
df = pd.read_csv("data.csv").fillna(0)

# 2) EXTRACT TARGET AND DROP IT FROM FEATURES
y_raw = df["vital.status"].values
le    = LabelEncoder()
y     = le.fit_transform(y_raw)

# now drop it
X_df = df.drop(columns=["vital.status"])

# 3) STANDARDIZE ALL FEATURES (unsupervised)
scaler_all = StandardScaler()
Xz_all     = scaler_all.fit_transform(X_df.values)
feat_names = X_df.columns.tolist()

# 4) UNSUPERVISED FEATURE SCORING: sum of absolute z‐scores
#    (no label involved)
feat_scores = pd.Series(np.abs(Xz_all).sum(axis=0),
                        index=feat_names) \
                .sort_values(ascending=False)

# 5) PICK TOP 10
top_n    = 10
top_feats = feat_scores.index[:top_n].tolist()

print(f"Top {top_n} features by unsupervised |z|‐sum:")
print(feat_scores.head(top_n))

# 6) BUILD FINAL FEATURE MATRIX
#    we can just slice Xz_all, since it's already z‐scored
idx = [feat_names.index(f) for f in top_feats]
X_sel_scaled = Xz_all[:, idx]

# 7) RANDOM FOREST + 10-FOLD STRATIFIED CV
rf = RandomForestClassifier(n_estimators=100,
                            n_jobs=-1,
                            random_state=0)
cv = StratifiedKFold(n_splits=10,
                     shuffle=True,
                     random_state=54)

scores = cross_val_score(rf,
                         X_sel_scaled,
                         y,
                         cv=cv,
                         scoring="accuracy")

print(f"\nMean Accuracy ± SD (10-fold CV): {scores.mean():.4f} ± {scores.std():.4f}")
