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

