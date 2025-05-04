import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 1) LOAD & DROP UNUSED
df = pd.read_csv("data.csv")        # ← changed from "cdot_data.csv"

# print counts on the full feature set
n_samples, n_features = df.shape
print(f"Full dataset: {n_samples} samples, {n_features} features (before extracting target)")

# 2) EXTRACT TARGET
y = df.pop("vital.status")          # ← changed from "bid_total"

# print counts after popping the target
print(f"Features only: {df.shape[0]} samples, {df.shape[1]} features (after extracting target)\n")

# 3) RF on ALL FEATURES → get importances
rf_full = RandomForestRegressor(
    n_estimators=100,
    random_state=1,
    n_jobs=-1
)
rf_full.fit(df, y)

importances_full = pd.Series(
    rf_full.feature_importances_,
    index=df.columns
).sort_values(ascending=False)

# 4) SET 1: Top 10 from all features
set1 = importances_full.head(10)
print("Set 1 – Top 10 features by RF importance (all features):")
print(set1, "\n")

# 5) Save top 100 features to 100.csv
top100 = importances_full.head(100)
top100.to_frame(name="importance").to_csv("100.csv")

# 6) RF on the SELECTED 100 FEATURES
features100 = top100.index.tolist()
df100 = df[features100]

# print counts on the 100‐feature subset
print(f"Top100 subset: {df100.shape[0]} samples, {df100.shape[1]} features\n")

rf_top100 = RandomForestRegressor(
    n_estimators=100,
    random_state=1,
    n_jobs=-1
)
rf_top100.fit(df100, y)

importances_100 = pd.Series(
    rf_top100.feature_importances_,
    index=features100
).sort_values(ascending=False)

# 7) SET 2: Top 10 from the 100‐feature subset
set2 = importances_100.head(10)
print("Set 2 – Top 10 features by RF importance (within top 100):")
print(set2)
