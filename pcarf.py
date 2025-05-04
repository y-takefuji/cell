# -*- coding: utf-8 -*-
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd

from sklearn.decomposition     import PCA
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import StratifiedKFold, cross_val_score

# 1) LOAD & CLEAN
filename = 'data.csv'
print("loading:", filename)
df = pd.read_csv(filename).fillna(0)

# 2) EXTRACT & ENCODE TARGET
y_raw = df.pop("vital.status").values
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 3) STANDARDIZE ALL ORIGINAL FEATURES
feature_names = df.columns.tolist()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)

# 4) RUN PCA
n_pcs = 10
pca = PCA(n_components=n_pcs, random_state=0)
_ = pca.fit_transform(X_scaled)

# 5) RANK ORIGINAL FEATURES BY PCA LOADINGS
#    Sum absolute loadings across the retained PCs, then sort descending.
loadings    = np.abs(pca.components_)   # shape (n_pcs, n_features)
loading_sums = loadings.sum(axis=0)     # length = n_features
feat_scores = pd.Series(loading_sums, index=feature_names) \
                .sort_values(ascending=False)

top_feats = feat_scores.index[:n_pcs].tolist()

print(f"\nTop {n_pcs} original features by |PCA loading| sum:")
print(feat_scores.head(n_pcs))

# 6) SUBSET DATA TO THOSE TOP FEATURES & RE‐STANDARDIZE
X_sel        = df[top_feats].values
scaler2      = StandardScaler()
X_sel_scaled = scaler2.fit_transform(X_sel)

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

mean_acc = scores.mean()
sd_acc   = scores.std()
print(f"\nMean Accuracy ± SD (10-fold CV): {mean_acc:.4f} ± {sd_acc:.4f}")
