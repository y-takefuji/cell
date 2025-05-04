import pandas as pd

# 1) load and drop unused columns
df = pd.read_csv("data.csv")        # ← changed from cdot_data.csv

# 2) extract target
y = df.pop("vital.status")          # ← changed from bid_total

# 3) compute Spearman |ρ| for ALL features
spearman_full = df.corrwith(y, method="spearman").abs()
spearman_full = spearman_full.sort_values(ascending=False)

# 4) Set 1: top-10 from all features
set1 = spearman_full.head(10)
print("Set 1 – Top 10 features by |Spearman ρ| (all features):")
print(set1, "\n")

# 5) take top-100 by Spearman and save to 100.csv
top100 = spearman_full.head(100)
top100.to_frame(name="spearman_rho").to_csv("100.csv")

# 6) Set 2: reload the 100 and re-rank them by Spearman
#    (you can also reuse 'top100.index' directly)
features100 = pd.read_csv("100.csv", index_col=0).index.tolist()
spearman_100 = df[features100].corrwith(y, method="spearman").abs()
spearman_100 = spearman_100.sort_values(ascending=False)
set2 = spearman_100.head(10)

print("Set 2 – Top 10 features by |Spearman ρ| (within the saved 100):")
print(set2)
