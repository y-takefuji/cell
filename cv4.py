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
    # Ensure n_components doesn't exceed the number of features or samples
    n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    
    # Calculate feature importance based on the magnitude of loadings across all components
    feature_importance = np.sum(np.abs(pca.components_), axis=0)
    importances = pd.Series(feature_importance, index=X.columns)
    top_features = importances.nlargest(10).index.tolist()
    
    return top_features, X[top_features]

# 2. ICA - Independent Component Analysis
def ica_feature_selection(X_scaled, X):
    # Get number of components - don't use too many
    n_components = min(10, X_scaled.shape[0], X_scaled.shape[1])
    
    # Fit ICA
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    ica_result = ica.fit_transform(X_scaled)
    
    # Get feature importance by reconstructing the data and measuring the reconstruction error
    feature_importance = []
    
    # For each feature, measure its contribution by removing it and calculating reconstruction error
    for i in range(X_scaled.shape[1]):
        X_temp = X_scaled.copy()
        X_temp[:, i] = 0  # Zero out this feature
        reconstructed = ica.inverse_transform(ica.transform(X_temp))
        error = np.mean((X_scaled - reconstructed) ** 2)
        feature_importance.append(error)
    
    # Create Series of feature importance
    importances = pd.Series(feature_importance, index=X.columns)
    top_features = importances.nlargest(10).index.tolist()
    
    return top_features, X[top_features]

# 3. HVGs - Highly Variable Genes (features with highest variance)
def hvgs_feature_selection(X, top_n=10):
    # Calculate variance for each feature
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    return top_features, X[top_features]

# 4. Feature Agglomeration
def feature_agglomeration(X_scaled, X, n_clusters=10):
    # Ensure n_clusters doesn't exceed the number of features
    n_clusters = min(n_clusters, X.shape[1])
    
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(X_scaled)
    
    # Create a mapping from original features to clusters
    feature_to_cluster = pd.Series(agglo.labels_, index=X.columns)
    
    # For each cluster, select the feature with the highest variance
    selected_features = []
    for cluster_id in range(n_clusters):
        cluster_features = feature_to_cluster[feature_to_cluster == cluster_id].index
        if len(cluster_features) > 0:
            variances = X[cluster_features].var()
            selected_features.append(variances.idxmax())
    
    return selected_features, X[selected_features]

# Perform feature selection using all methods
print(f"Dataset shape: {X.shape}")
print("Running PCA feature selection...")
pca_features, X_pca = pca_feature_selection(X_scaled, X)

print("Running ICA feature selection...")
ica_features, X_ica = ica_feature_selection(X_scaled, X)

print("Running HVGS feature selection...")
hvgs_features, X_hvgs = hvgs_feature_selection(X)

print("Running Feature Agglomeration...")
fa_features, X_fa = feature_agglomeration(X_scaled, X)

# Function to evaluate features using Random Forest with 10-fold CV
def evaluate_features(X_selected, y, method_name):
    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Set up 10-fold cross-validation with stratification
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Perform cross-validation using ONLY the selected top 10 features
    scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='accuracy')
    
    # Print detailed results
    print(f"{method_name} - 10-fold CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"Individual fold scores: {[f'{score:.4f}' for score in scores]}")
    
    return scores.mean(), scores.std()

# Evaluate all feature selection methods
print("\n===== CROSS-VALIDATION RESULTS WITH TOP 10 FEATURES =====")

print("\nPCA feature selection:")
print("Top 10 features:", pca_features)
print(f"Shape of reduced dataset: {X_pca.shape}")
pca_acc, pca_std = evaluate_features(X_pca, y, "PCA")

print("\nICA feature selection:")
print("Top 10 features:", ica_features)
print(f"Shape of reduced dataset: {X_ica.shape}")
ica_acc, ica_std = evaluate_features(X_ica, y, "ICA")

print("\nHVGS feature selection:")
print("Top 10 features:", hvgs_features)
print(f"Shape of reduced dataset: {X_hvgs.shape}")
hvgs_acc, hvgs_std = evaluate_features(X_hvgs, y, "HVGS")

print("\nFeature Agglomeration:")
print("Top 10 features:", fa_features)
print(f"Shape of reduced dataset: {X_fa.shape}")
fa_acc, fa_std = evaluate_features(X_fa, y, "Feature Agglomeration")

# Plot the results
methods = ['PCA', 'ICA', 'HVGS', 'Feature Agglomeration']
accuracies = [pca_acc, ica_acc, hvgs_acc, fa_acc]
errors = [pca_std, ica_std, hvgs_std, fa_std]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, accuracies, yerr=errors, capsize=10, color=['blue', 'green', 'red', 'purple'])
plt.ylabel('10-fold CV Accuracy')
plt.title('Feature Selection Methods Comparison (Top 10 Features)')
plt.ylim(max(0.5, min(accuracies) - 0.1), min(1.0, max(accuracies) + 0.1))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add accuracy values on top of bars
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[i] + 0.01,
             f'{accuracies[i]:.4f}', ha='center')

plt.tight_layout()
plt.savefig('feature_selection_comparison.png')
plt.show()

# Find the best method
best_idx = np.argmax(accuracies)
best_method = methods[best_idx]
best_features = [pca_features, ica_features, hvgs_features, fa_features][best_idx]

print(f"\n===== SUMMARY =====")
print(f"Best performing method: {best_method} with accuracy: {accuracies[best_idx]:.4f} ± {errors[best_idx]:.4f}")
print(f"Top 10 features from {best_method}:")
for i, feature in enumerate(best_features, 1):
    print(f"{i}. {feature}")

# Compare to baseline using all features
print("\n===== BASELINE COMPARISON =====")
print(f"Running baseline with all {X.shape[1]} features...")
baseline_acc, baseline_std = evaluate_features(X, y, "All Features")
print(f"Improvement using {best_method}: {accuracies[best_idx] - baseline_acc:.4f} ({(accuracies[best_idx] - baseline_acc) / baseline_acc * 100:.2f}%)")
