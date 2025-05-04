import numpy as np
import pandas as pd

# 1) LOAD & EXTRACT TARGET
df = pd.read_csv("data.csv").fillna(0)
y  = df.pop("vital.status")    # target (not used in HVGS)

# 2) HVGS ON ALL FEATURES
means = df.mean(axis=0)
vars_ = df.var(axis=0, ddof=1)

# avoid division by zero
mask = means > 0
dispersion = (vars_[mask] / means[mask])\
               .replace([np.inf, -np.inf], np.nan)\
               .dropna()

# rank features by decreasing dispersion
disp_all = dispersion.sort_values(ascending=False)

# 3) SET 1: Top 10 by dispersion (all features)
set1 = disp_all.head(10)
print("Set 1 – Top 10 features by HVGS dispersion (all features):")
print(set1, "\n")

# 4) SAVE TOP 100 TO 100.csv
top100 = disp_all.head(100)
top100.to_frame(name="dispersion").to_csv("100.csv")

# 5) HVGS ON THE TOP‐100 SUBSET
df100 = df[top100.index]

means100 = df100.mean(axis=0)
vars100  = df100.var(axis=0, ddof=1)
mask100  = means100 > 0
disp100  = (vars100[mask100] / means100[mask100])\
              .replace([np.inf, -np.inf], np.nan)\
              .dropna()

disp100_sorted = disp100.sort_values(ascending=False)

# 6) SET 2: Top 10 by dispersion (within top 100)
set2 = disp100_sorted.head(10)
print("Set 2 – Top 10 features by HVGS dispersion (within top 100):")
print(set2)
