import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration

# Load the data
data = pd.read_csv('data.csv')

# Separate target and features
X = data.drop('vital.status', axis=1)
y = data['vital.status']

print(f"Original dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Define the number of features to select
k_features = 10

# Function to evaluate features
def evaluate_features(X_selected, y, method_name):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='accuracy')
    print(f"{method_name} - 5-fold CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"Individual fold scores: {[f'{s:.4f}' for s in scores]}")
    return scores.mean(), scores.std()

# 1. PCA Feature Selection
print("\n===== PCA Feature Selection =====")
def pca_feature_selection(X, n_components=10):
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    feature_importance = np.sum(np.abs(pca.components_), axis=0)
    importances = pd.Series(feature_importance, index=X.columns)
    top_features = importances.nlargest(10).index.tolist()
    return top_features, X[top_features]

pca_features, X_pca = pca_feature_selection(X)
print(f"Top {len(pca_features)} features from PCA:")
for feature in pca_features:
    print(f"- {feature}")

# 2. HVGs - Highly Variable Genes
print("\n===== HVGs Feature Selection =====")
def hvgs_feature_selection(X, top_n=10):
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    return top_features, X[top_features]

hvgs_features, X_hvgs = hvgs_feature_selection(X)
print(f"Top {len(hvgs_features)} features from HVGs:")
for feature in hvgs_features:
    print(f"- {feature}")

# 3. Feature Agglomeration
print("\n===== Feature Agglomeration Selection =====")
def feature_agglomeration(X, n_clusters=10):
    n_clusters = min(n_clusters, X.shape[1])
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(X)
    feature_to_cluster = pd.Series(agglo.labels_, index=X.columns)
    selected_features = []
    for cid in range(n_clusters):
        cluster_feats = feature_to_cluster[feature_to_cluster == cid].index
        if len(cluster_feats) > 0:
            variances = X[cluster_feats].var()
            selected_features.append(variances.idxmax())
    return selected_features, X[selected_features]

fa_features, X_fa = feature_agglomeration(X)
print(f"Top {len(fa_features)} features from Feature Agglomeration:")
for feature in fa_features:
    print(f"- {feature}")

# Evaluate each feature selection method
print("\n===== Evaluation Results =====")
pca_acc, pca_std = evaluate_features(X_pca, y, "PCA")
hvgs_acc, hvgs_std = evaluate_features(X_hvgs, y, "HVGs")
fa_acc, fa_std = evaluate_features(X_fa, y, "Feature Agglomeration")

# Compare model performances
results = {
    'PCA': pca_acc,
    'HVGs': hvgs_acc,
    'Feature Agglomeration': fa_acc
}

# Print results comparison
print("\n===== Model Comparison =====")
for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {score:.4f}")

# Determine the best model
best_score = max(results.values())
best_model_name = [name for name, score in results.items() if score == best_score][0]
print(f"\nBest model: {best_model_name} with accuracy: {best_score:.4f}")

# Create summary for easier comparison
accuracies = [pca_acc, hvgs_acc, fa_acc]
errors = [pca_std, hvgs_std, fa_std]
methods = ["PCA", "HVGs", "Feature Agglomeration"]
best_idx = np.argmax(accuracies)
best_method = methods[best_idx]
best_feats = [pca_features, hvgs_features, fa_features][best_idx]

print(f"\n===== SUMMARY =====")
print(f"Best method: {best_method} ({accuracies[best_idx]:.4f} ± {errors[best_idx]:.4f})")
print("Top features from", best_method, ":", best_feats)

# Evaluate with all features as baseline
print("\n===== Baseline (All Features) =====")
baseline_acc, baseline_std = evaluate_features(X, y, "All Features")
print(f"Improvement vs. baseline: {accuracies[best_idx] - baseline_acc:.4f}")

# Save the selected features for each method
print("\n===== Saving Selected Features =====")
feature_selections = {
    'PCA': pca_features,
    'HVGs': hvgs_features,
    'Feature Agglomeration': fa_features
}

# Print the selected features for each method
for method, features in feature_selections.items():
    print(f"\n{method} selected features:")
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature}")

# Now run again with top 9 features to check stability
print("\n\n===== EVALUATING WITH TOP 9 FEATURES =====")
k_features = 9

# Rerun PCA with top 9
def pca_feature_selection_9(X, n_components=9):
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    feature_importance = np.sum(np.abs(pca.components_), axis=0)
    importances = pd.Series(feature_importance, index=X.columns)
    top_features = importances.nlargest(9).index.tolist()
    return top_features, X[top_features]

pca_features_9, X_pca_9 = pca_feature_selection_9(X)
print(f"\nTop {len(pca_features_9)} features from PCA:")
for feature in pca_features_9:
    print(f"- {feature}")

# Rerun HVGs with top 9
hvgs_features_9 = hvgs_features[:9]  # Take first 9 from the top 10
X_hvgs_9 = X[hvgs_features_9]
print(f"\nTop {len(hvgs_features_9)} features from HVGs:")
for feature in hvgs_features_9:
    print(f"- {feature}")

# Rerun Feature Agglomeration with top 9
fa_features_9 = fa_features[:9]  # Take first 9 from the top 10
X_fa_9 = X[fa_features_9]
print(f"\nTop {len(fa_features_9)} features from Feature Agglomeration:")
for feature in fa_features_9:
    print(f"- {feature}")

# Evaluate each feature selection method with top 9 features
print("\n===== Evaluation Results (Top 9) =====")
pca_acc_9, pca_std_9 = evaluate_features(X_pca_9, y, "PCA (Top 9)")
hvgs_acc_9, hvgs_std_9 = evaluate_features(X_hvgs_9, y, "HVGs (Top 9)")
fa_acc_9, fa_std_9 = evaluate_features(X_fa_9, y, "Feature Agglomeration (Top 9)")

# Compare model performances with top 9 features
results_9 = {
    'PCA (Top 9)': pca_acc_9,
    'HVGs (Top 9)': hvgs_acc_9,
    'Feature Agglomeration (Top 9)': fa_acc_9
}

# Print results comparison for top 9
print("\n===== Model Comparison (Top 9) =====")
for model_name, score in sorted(results_9.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {score:.4f}")

# Check feature ranking stability
print("\n===== Feature Ranking Stability =====")
# For PCA
pca_stable = set(pca_features_9).issubset(set(pca_features))
print(f"PCA stability (top 9 is subset of top 10): {'Pass' if pca_stable else 'Fail'}")

# For HVGs - should be stable since we just took the first 9
hvgs_stable = hvgs_features[:9] == hvgs_features_9
print(f"HVGs stability (first 9 of top 10): {'Pass' if hvgs_stable else 'Fail'}")

# For FA - should be stable since we just took the first 9
fa_stable = fa_features[:9] == fa_features_9
print(f"Feature Agglomeration stability (first 9 of top 10): {'Pass' if fa_stable else 'Fail'}")

# Print summary table for the paper
print("\n===== Summary Table for Paper =====")
print("| Method | Top 10 accuracy±std | Top 9 accuracy±std | Consistency |")
print("|--------|-------------------|------------------|------------|")
print(f"| HVGS | {hvgs_acc:.4f} ± {hvgs_std:.4f} | {hvgs_acc_9:.4f} ± {hvgs_std_9:.4f} | {'Pass' if hvgs_stable else 'Fail'} |")
print(f"| FA | {fa_acc:.4f} ± {fa_std:.4f} | {fa_acc_9:.4f} ± {fa_std_9:.4f} | {'Pass' if fa_stable else 'Fail'} |")
print(f"| PCA | {pca_acc:.4f} ± {pca_std:.4f} | {pca_acc_9:.4f} ± {pca_std_9:.4f} | {'Pass' if pca_stable else 'Fail'} |")
