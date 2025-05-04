import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) LOAD & EXTRACT TARGET
df = pd.read_csv("data.csv").fillna(0)
y  = df.pop("vital.status")    # assume a classification task

# 2) HVGS: compute per‐feature dispersion = variance / mean
means      = df.mean(axis=0)
variances  = df.var(axis=0, ddof=1)
dispersion = (variances / means) \
                .replace([np.inf, -np.inf], np.nan) \
                .dropna() \
                .sort_values(ascending=False)

# 3) SELECT TOP 10 FEATURES
top10 = dispersion.head(10).index.tolist()
print("Top 10 features by HVGS dispersion:")
print(top10, "\n")

X_top10 = df[top10].values

# 4) 10-FOLD STRATIFIED CV WITH RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=1,
    n_jobs=-1
)
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=1
)

scores = cross_val_score(
    rf,
    X_top10,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

mean_acc = scores.mean()
sd_acc   = scores.std(ddof=1)

print(f"\nMean Accuracy ± SD (10-fold CV): {mean_acc:.4f} ± {sd_acc:.4f}")
