import pandas as pd
from sklearn.linear_model import LinearRegression

# 1) LOAD & DROP UNUSED
df = pd.read_csv("data.csv")
n_samples, n_features = df.shape
print(f"Full dataset: {n_samples} samples, {n_features} features (before extracting target)")

# 2) EXTRACT TARGET
y = df.pop("vital.status")
print(f"Features only: {df.shape[0]} samples, {df.shape[1]} features (after extracting target)\n")

# 3) LINEAR REGRESSION on ALL FEATURES → get coefficients
lr_full = LinearRegression()
lr_full.fit(df, y)

# 4) RANK FEATURES BY ABSOLUTE COEFFICIENT (importance proxy)
coef_full = pd.Series(lr_full.coef_, index=df.columns)
importances_full = coef_full.abs().sort_values(ascending=False)

# 5) SET 1: Top 10 from all features
set1 = importances_full.head(10)
print("Set 1 – Top 10 features by |LinearRegression coef| (all features):")
print(set1.to_frame(name="|coef|"), "\n")

# 6) Save top 100 features to 100.csv
top100 = importances_full.head(100)
top100.to_frame(name="|coef|").to_csv("100.csv")

# 7) LINEAR REGRESSION on the SELECTED 100 FEATURES
features100 = top100.index.tolist()
df100 = df[features100]
print(f"Top100 subset: {df100.shape[0]} samples, {df100.shape[1]} features\n")

lr_top100 = LinearRegression()
lr_top100.fit(df100, y)

coef_100 = pd.Series(lr_top100.coef_, index=features100)
importances_100 = coef_100.abs().sort_values(ascending=False)

# 8) SET 2: Top 10 from the 100‐feature subset
set2 = importances_100.head(10)
print("Set 2 – Top 10 features by |LinearRegression coef| (within top 100):")
print(set2.to_frame(name="|coef|"))
