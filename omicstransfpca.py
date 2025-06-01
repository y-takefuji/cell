import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import time

# Load the data
def load_data(filepath='data.csv'):
    print(f"Loading data from {filepath}...")
    start_time = time.time()
    
    data = pd.read_csv(filepath)
    
    # Extract target variable 'vital.status'
    if 'vital.status' not in data.columns:
        raise ValueError("Target variable 'vital.status' not found in dataset")
    
    y = data['vital.status']
    X = data.drop('vital.status', axis=1)
    
    # Ensure all features are numeric
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X = X[numeric_cols]
    
    print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Loading took {time.time() - start_time:.2f} seconds")
    
    return X, y

# Method 1: Feature selection using PCA without scaling
def method_pca_without_scaling(X, y):
    print("\nMethod 1: Feature selection using PCA without scaling")
    start_time = time.time()
    
    # Apply PCA without scaling to identify important features
    print("Applying PCA without scaling to identify important features...")
    pca = PCA(n_components=10)
    pca.fit(X)
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    print(f"Explained variance by 10 components: {cumulative_variance[-1]:.4f}")
    
    # Identify most important original features based on PCA loadings
    feature_importance = np.zeros(X.shape[1])
    for i in range(10):
        # Sum the absolute loadings across all components
        feature_importance += np.abs(pca.components_[i])
    
    # Get top 10 features
    top_indices = np.argsort(-feature_importance)[:10]
    top_features = [X.columns[i] for i in top_indices]
    
    # Create reduced dataset with selected features
    X_reduced = X[top_features]
    
    print(f"Top 10 features selected: {top_features}")
    
    # Now cross-validate on the reduced dataset
    print("Performing cross-validation on reduced dataset...")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_reduced, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    print(f"Method execution took {time.time() - start_time:.2f} seconds")
    
    return scores.mean(), scores.std(), top_features, X_reduced

# Method 2: Feature selection using PCA with scaling
def method_pca_with_scaling(X, y):
    print("\nMethod 2: Feature selection using PCA with scaling")
    start_time = time.time()
    
    # Scale the data
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to identify important features
    print("Applying PCA to scaled data to identify important features...")
    pca = PCA(n_components=10)
    pca.fit(X_scaled)
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    print(f"Explained variance by 10 components: {cumulative_variance[-1]:.4f}")
    
    # Identify most important original features based on PCA loadings
    feature_importance = np.zeros(X.shape[1])
    for i in range(10):
        # Sum the absolute loadings across all components
        feature_importance += np.abs(pca.components_[i])
    
    # Get top 10 features
    top_indices = np.argsort(-feature_importance)[:10]
    top_features = [X.columns[i] for i in top_indices]
    
    # Create reduced dataset with selected features
    X_reduced = X[top_features]  # Using original X, not X_scaled
    
    print(f"Top 10 features selected: {top_features}")
    
    # Now cross-validate on the reduced dataset
    print("Performing cross-validation on reduced dataset...")
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_reduced, y, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    print(f"Method execution took {time.time() - start_time:.2f} seconds")
    
    return scores.mean(), scores.std(), top_features, X_reduced

# Compare results
def compare_results(unscaled_acc, unscaled_std, scaled_acc, scaled_std, unscaled_features, scaled_features):
    # Compare feature sets
    common_features = set(unscaled_features).intersection(set(scaled_features))
    print(f"\nNumber of common features between methods: {len(common_features)}")
    if common_features:
        print(f"Common features: {common_features}")
    
    print("\n--- Results Summary ---")
    print(f"Method 1 (PCA without scaling) - Accuracy: {unscaled_acc:.4f} (±{unscaled_std:.4f})")
    print(f"Method 2 (PCA with scaling) - Accuracy: {scaled_acc:.4f} (±{scaled_std:.4f})")
    
    if scaled_acc > unscaled_acc:
        print("Method 2 with scaling performed better for this dataset.")
    elif unscaled_acc > scaled_acc:
        print("Method 1 without scaling performed better for this dataset.")
    else:
        print("Both methods performed equally.")

def main():
    # Load data
    X, y = load_data()
    
    # Method 1: Feature selection using PCA without scaling
    unscaled_acc, unscaled_std, unscaled_features, X_reduced_unscaled = method_pca_without_scaling(X, y)
    
    # Method 2: Feature selection using PCA with scaling
    scaled_acc, scaled_std, scaled_features, X_reduced_scaled = method_pca_with_scaling(X, y)
    
    # Compare results
    compare_results(unscaled_acc, unscaled_std, scaled_acc, scaled_std, unscaled_features, scaled_features)
    
    # Additional information about the reduced datasets
    print("\n--- Reduced Dataset Information ---")
    print(f"Method 1 reduced dataset shape: {X_reduced_unscaled.shape}")
    print(f"Method 2 reduced dataset shape: {X_reduced_scaled.shape}")

if __name__ == "__main__":
    total_start = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - total_start:.2f} seconds")
