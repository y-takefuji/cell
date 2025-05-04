import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# 1) LOAD & DROP UNUSED
df = pd.read_csv("data.csv")       # ← changed from "cdot_data.csv"

print(f"Full dataset: {df.shape[0]} samples, {df.shape[1]} features")

# 2) EXTRACT TARGET 
y = df.pop("vital.status")         # ← changed from "bid_total"
print(f"Features only: {df.shape[0]} samples, {df.shape[1]} features\n")

# Helper to compute Mutual Information–based feature importances
def mutual_info_feature_importance(X, y):
    mi_scores = mutual_info_regression(X, y, random_state=7)
    return pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# 3) SET 1: Mutual Information importances on ALL features
feat_imp_full = mutual_info_feature_importance(df, y)
set1 = feat_imp_full.head(10)

print("Set 1 – Top 10 features by Mutual Information importance (all features):")
print(set1, "\n")

# 4) SAVE top 100 features
top100 = feat_imp_full.head(100)
top100.to_frame(name="importance").to_csv("100.csv")

# 5) SET 2: Mutual Information importances on the top-100 feature subset
df100      = df[top100.index]
feat_imp_100 = mutual_info_feature_importance(df100, y)
set2         = feat_imp_100.head(10)

print("Set 2 – Top 10 features by Mutual Information importance (within top 100):")
print(set2)
