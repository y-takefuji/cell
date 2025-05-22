import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data.csv')

# Separate target and features
X = data.drop('vital.status', axis=1)
y = data['vital.status']

# Convert all features to numeric (float)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Handle any missing values
X = X.fillna(X.mean())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 1. PCA - Principal Component Analysis
def pca_feature_selection(X_scaled, X, n_components=10):
    n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    feature_importance = np.sum(np.abs(pca.components_), axis=0)
    importances = pd.Series(feature_importance, index=X.columns)
    top_features = importances.nlargest(10).index.tolist()
    return top_features, X[top_features]

# 2. ICA - Independent Component Analysis
def ica_feature_selection(X_scaled, X):
    n_components = min(10, X_scaled.shape[0], X_scaled.shape[1])
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    ica.fit(X_scaled)
    feature_importance = []
    for i in range(X_scaled.shape[1]):
        X_temp = X_scaled.copy()
        X_temp[:, i] = 0
        reconstructed = ica.inverse_transform(ica.transform(X_temp))
        error = np.mean((X_scaled - reconstructed) ** 2)
        feature_importance.append(error)
    importances = pd.Series(feature_importance, index=X.columns)
    top_features = importances.nlargest(10).index.tolist()
    return top_features, X[top_features]

# 3. HVGs - Highly Variable Genes
def hvgs_feature_selection(X, top_n=10):
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    return top_features, X[top_features]

# 4. Feature Agglomeration
def feature_agglomeration(X_scaled, X, n_clusters=10):
    n_clusters = min(n_clusters, X.shape[1])
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(X_scaled)
    feature_to_cluster = pd.Series(agglo.labels_, index=X.columns)
    selected_features = []
    for cid in range(n_clusters):
        cluster_feats = feature_to_cluster[feature_to_cluster == cid].index
        if len(cluster_feats) > 0:
            variances = X[cluster_feats].var()
            selected_features.append(variances.idxmax())
    return selected_features, X[selected_features]

# Perform feature selection
print(f"Dataset shape: {X.shape}")
pca_features, X_pca = pca_feature_selection(X_scaled, X)
ica_features, X_ica = ica_feature_selection(X_scaled, X)
hvgs_features, X_hvgs = hvgs_feature_selection(X)
fa_features, X_fa = feature_agglomeration(X_scaled, X)

# Function to evaluate with Random Forest & 5-fold CV
def evaluate_features(X_selected, y, method_name):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='accuracy')
    print(f"{method_name} - 5-fold CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"Individual fold scores: {[f'{s:.4f}' for s in scores]}")
    return scores.mean(), scores.std()

print("\n===== CROSS-VALIDATION RESULTS WITH TOP 10 FEATURES (5-fold CV) =====")

print("\nPCA feature selection:")
print("Top 10 features:", pca_features)
pca_acc, pca_std = evaluate_features(X_pca, y, "PCA")

print("\nICA feature selection:")
print("Top 10 features:", ica_features)
ica_acc, ica_std = evaluate_features(X_ica, y, "ICA")

print("\nHVGS feature selection:")
print("Top 10 features:", hvgs_features)
hvgs_acc, hvgs_std = evaluate_features(X_hvgs, y, "HVGS")

print("\nFeature Agglomeration:")
print("Top 10 features:", fa_features)
fa_acc, fa_std = evaluate_features(X_fa, y, "Feature Agglomeration")

# Plot the results
methods = ['PCA', 'ICA', 'HVGS', 'Feature Agglomeration']
accuracies = [pca_acc, ica_acc, hvgs_acc, fa_acc]
errors = [pca_std, ica_std, hvgs_std, fa_std]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, accuracies, yerr=errors, capsize=8,
               color=['blue','green','red','purple'])
plt.ylabel('5-fold CV Accuracy')
plt.title('Feature Selection Methods Comparison\n(Top 10 Features, 5-fold CV)')
plt.ylim(max(0.5, min(accuracies)-0.1), min(1.0, max(accuracies)+0.1))
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, bar in enumerate(bars):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+errors[i]+0.005,
             f'{accuracies[i]:.4f}', ha='center')

plt.tight_layout()
plt.savefig('feature_selection_comparison_5fold.png')
plt.show()

# Summary
best_idx = np.argmax(accuracies)
best_method = methods[best_idx]
best_feats = [pca_features, ica_features, hvgs_features, fa_features][best_idx]

print(f"\n===== SUMMARY =====")
print(f"Best method: {best_method} ({accuracies[best_idx]:.4f} ± {errors[best_idx]:.4f})")
print("Top 10 features from", best_method, ":", best_feats)

# Baseline with all features
print("\n===== BASELINE COMPARISON =====")
baseline_acc, baseline_std = evaluate_features(X, y, "All Features")
print(f"Improvement vs. baseline: {accuracies[best_idx] - baseline_acc:.4f} "
      f"({(accuracies[best_idx]-baseline_acc)/baseline_acc*100:.2f}%)")
