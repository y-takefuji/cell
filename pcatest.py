import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# 1) LOAD
filename = 'data.csv'
print("loading:", filename)
df = pd.read_csv(filename).fillna(0)

# 2) EXTRACT & ENCODE TARGET
y_raw = df.pop("vital.status").values
le    = LabelEncoder()
y     = le.fit_transform(y_raw)    # e.g. 0/1 classes. do not change other lines

# 3) PCA‐importance helper
def pca_feature_importance(X, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    abs_loads = np.abs(pca.components_)            # (n_pcs, n_features)
    weights   = pca.explained_variance_ratio_[:,None]
    weighted  = abs_loads * weights                # broadcast
    scores    = weighted.sum(axis=0)               # (n_features,)
    return pd.Series(scores, index=X.columns) \
             .sort_values(ascending=False)

# 4) SET 1: Top 10 on ALL features
feat_imp_full = pca_feature_importance(df)
set1 = feat_imp_full.head(10)
print("Set 1 – Top 10 features by PCA‐based importance (all features):")
print(set1, "\n")

# 5) SAVE top 100
top100 = feat_imp_full.head(100)
top100.to_frame(name="importance") \
      .to_csv("100.csv", index=True)

# 6) SET 2: Top 10 within that top‐100 subset
df100       = df[top100.index]
feat_imp_100 = pca_feature_importance(df100)
set2         = feat_imp_100.head(10)
print("Set 2 – Top 10 features by PCA‐based importance (within top 100):")
print(set2)