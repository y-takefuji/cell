import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# 1) LOAD & DROP UNUSED
df = pd.read_csv("data.csv")
n_samples, n_features = df.shape
print(f"Full dataset: {n_samples} samples, {n_features} features (before extracting target)")

# 2) EXTRACT TARGET
y = df.pop("vital.status")
print(f"Features only: {df.shape[0]} samples, {df.shape[1]} features (after extracting target)\n")

# 3) DECISION TREE on ALL FEATURES → get importances
dt_full = DecisionTreeRegressor(
    random_state=1
)
dt_full.fit(df, y)

# 4) RANK FEATURES BY IMPORTANCE
importances_full = pd.Series(
    dt_full.feature_importances_,
    index=df.columns
).sort_values(ascending=False)

# 5) SET 1: Top 10 from all features
set1 = importances_full.head(10)
print("Set 1 – Top 10 features by DecisionTree importance (all features):")
print(set1.to_frame(name="importance"), "\n")

# 6) Save top 100 features to 100.csv
top100 = importances_full.head(100)
top100.to_frame(name="importance").to_csv("100.csv", index=True)

# 7) DECISION TREE on the SELECTED 100 FEATURES
features100 = top100.index.tolist()
df100 = df[features100]
print(f"Top100 subset: {df100.shape[0]} samples, {df100.shape[1]} features\n")

dt_top100 = DecisionTreeRegressor(
    random_state=1
)
dt_top100.fit(df100, y)

importances_100 = pd.Series(
    dt_top100.feature_importances_,
    index=features100
).sort_values(ascending=False)

# 8) SET 2: Top 10 from the 100‐feature subset
set2 = importances_100.head(10)
print("Set 2 – Top 10 features by DecisionTree importance (within top 100):")
print(set2.to_frame(name="importance"))
