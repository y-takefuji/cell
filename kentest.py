import pandas as pd

# 1) load and drop unused columns
df = pd.read_csv("data.csv")        # ← same as before

# 2) extract target
y = df.pop("vital.status")          # ← same as before

# 3) compute Kendall |τ| for ALL features
kendall_full = df.corrwith(y, method="kendall").abs()
kendall_full = kendall_full.sort_values(ascending=False)

# 4) Set 1: top-10 from all features
set1 = kendall_full.head(10)
print("Set 1 – Top 10 features by |Kendall τ| (all features):")
print(set1, "\n")

# 5) take top-100 by Kendall and save to 100.csv
top100 = kendall_full.head(100)
top100.to_frame(name="kendall_tau").to_csv("100.csv")

# 6) Set 2: reload the 100 and re-rank them by Kendall
#    (you can also reuse 'top100.index' directly)
features100 = pd.read_csv("100.csv", index_col=0).index.tolist()
kendall_100 = df[features100].corrwith(y, method="kendall").abs()
kendall_100 = kendall_100.sort_values(ascending=False)
set2 = kendall_100.head(10)

print("Set 2 – Top 10 features by |Kendall τ| (within the saved 100):")
print(set2)
