# -*- coding: utf-8 -*-
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) LOAD & CLEAN
filename = 'data.csv'
print("loading:", filename)
df = pd.read_csv(filename).fillna(0)

# 2) EXTRACT & ENCODE TARGET
y_raw = df.pop("vital.status").values
le    = LabelEncoder()
y     = le.fit_transform(y_raw)    # e.g. 0/1 classes
n     = y.shape[0]

# 3) STANDARDIZE ALL ORIGINAL FEATURES
feature_names = df.columns.tolist()
scaler        = StandardScaler()
X_scaled      = scaler.fit_transform(df.values)
# shape == (n_samples, n_features)

# 4) LINEAR REGRESSION FEATURE RANKING
#    Fit OLS to predict y from all X, then rank by |coef|.
lr = LinearRegression()
lr.fit(X_scaled, y)
coefs      = lr.coef_                      # shape (n_features,)
feat_scores = pd.Series(np.abs(coefs),
                        index=feature_names)\
                .sort_values(ascending=False)

# pick top‐K
n_select = 10
top_feats = feat_scores.index[:n_select].tolist()

print(f"\nTop {n_select} features by |LinearRegression coef|:")
print(feat_scores.head(n_select))

# 5) SUBSET & RE‐STANDARDIZE SELECTED FEATURES
X_sel        = df[top_feats].values
scaler2      = StandardScaler()
X_sel_scaled = scaler2.fit_transform(X_sel)

# 6) RANDOM FOREST + 10‐FOLD STRATIFIED CV
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
