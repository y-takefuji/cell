import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr

# Load the data
def load_data(file_path='data.csv'):
    data = pd.read_csv(file_path)
    X = data.drop('vital.status', axis=1)
    y = data['vital.status']
    return X, y

def random_forest_feature_selection(X, y, n_features=10):
    """Select top features using Random Forest importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns[indices]
    return features[:n_features], importances[indices[:n_features]]

def feature_agglomeration_selection(X, y, n_features=10):
    """Select features using feature agglomeration"""
    # Apply feature agglomeration directly without scaling
    agglo = FeatureAgglomeration(n_clusters=n_features)
    agglo.fit(X)
    
    # Calculate feature importance for each cluster
    cluster_importances = []
    for i in range(n_features):
        cluster_mask = (agglo.labels_ == i)
        feature_indices = np.where(cluster_mask)[0]
        
        # For each cluster, select the feature with highest correlation with target
        correlations = []
        for idx in feature_indices:
            corr, _ = spearmanr(X.iloc[:, idx], y)
            correlations.append((abs(corr), idx))
        
        if correlations:
            # Select the feature with highest correlation in this cluster
            max_corr_idx = max(correlations, key=lambda x: x[0])[1]
            cluster_importances.append((X.columns[max_corr_idx], max_corr_idx))
    
    # Sort by absolute correlation
    selected_features = [feat for feat, _ in cluster_importances[:n_features]]
    importance_values = [abs(spearmanr(X[feat], y)[0]) for feat in selected_features]
    
    return selected_features, importance_values

def highly_variable_gene_selection(X, y, n_features=10):
    """Select features with highest variance"""
    variances = np.var(X, axis=0)
    indices = np.argsort(variances)[::-1]
    features = X.columns[indices]
    return features[:n_features], variances[indices[:n_features]]

def spearman_correlation_selection(X, y, n_features=10):
    """Select features with highest absolute Spearman correlation with target"""
    correlations = []
    p_values = []
    
    for col in X.columns:
        corr, p_val = spearmanr(X[col], y)
        correlations.append(abs(corr))
        p_values.append(p_val)
    
    # Create a dataframe with results
    corr_df = pd.DataFrame({
        'feature': X.columns,
        'correlation': correlations,
        'p_value': p_values
    })
    
    # Sort by absolute correlation value
    corr_df = corr_df.sort_values('correlation', ascending=False).reset_index(drop=True)
    
    return corr_df['feature'].values[:n_features], corr_df['correlation'].values[:n_features]

def evaluate_feature_set(X, y, features, cv_folds=5):
    """Evaluate a feature set using cross-validation with Random Forest"""
    X_reduced = X[features]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_reduced, y, cv=cv, scoring='accuracy')
    
    return scores.mean(), scores.std()

def test_stability(X, y, feature_selection_func):
    """Test stability by removing top feature and comparing rankings"""
    # Get original top 10 features
    top_features, _ = feature_selection_func(X, y, 10)
    
    # Remove top feature
    X_reduced = X.drop(top_features[0], axis=1)
    
    # Get new top 9 features from the reduced dataset
    new_top_features, _ = feature_selection_func(X_reduced, y, 9)
    
    # Compare remaining 9 features from original top 10 (excluding the removed top feature)
    # with the new top 9 features
    original_remaining_features = set(top_features[1:10])
    new_features_set = set(new_top_features)
    
    # Calculate stability metrics
    common_features = original_remaining_features.intersection(new_features_set)
    stability_score = len(common_features) / 9  # Comparing 9 features
    
    return stability_score, list(top_features[1:10]), list(new_top_features)

def main():
    # Load data
    X, y = load_data()
    
    # Define feature selection methods
    selection_methods = {
        'Random Forest': random_forest_feature_selection,
        'Feature Agglomeration': feature_agglomeration_selection,
        'Highly Variable': highly_variable_gene_selection,
        'Spearman Correlation': spearman_correlation_selection
    }
    
    # Results storage
    results = []
    
    # For each method
    for method_name, selection_func in selection_methods.items():
        print(f"\n=== {method_name} Feature Selection ===")
        
        # Select top features
        selected_features, importance_values = selection_func(X, y, 10)
        print("Top 10 features:")
        for i, (feature, importance) in enumerate(zip(selected_features, importance_values)):
            print(f"{i+1}. {feature}: {importance:.4f}")
        
        # Evaluate with cross-validation using the top 10 features
        cv_mean_10, cv_std_10 = evaluate_feature_set(X, y, selected_features)
        print(f"Cross-validation accuracy (10 features): {cv_mean_10:.4f} ± {cv_std_10:.4f}")
        
        # Test stability after removing top feature
        stability_score, original_remaining, new_top9 = test_stability(X, y, selection_func)
        stability_status = "Stable" if stability_score >= 0.7 else "Unstable"
        
        # Evaluate with cross-validation using the top 9 features (after removing top feature)
        cv_mean_9, cv_std_9 = evaluate_feature_set(X, y, new_top9)
        
        print(f"Cross-validation accuracy (9 features): {cv_mean_9:.4f} ± {cv_std_9:.4f}")
        print(f"Stability score: {stability_score:.2f} ({stability_status})")
        print(f"Original remaining features (positions 2-10): {original_remaining}")
        print(f"New top 9 features: {new_top9}")
        
        # Store results
        results.append({
            'Method': method_name,
            'Top 10 Features': selected_features,
            'CV Accuracy (10)': cv_mean_10,
            'CV STD (10)': cv_std_10,
            'CV Accuracy (9)': cv_mean_9,
            'CV STD (9)': cv_std_9,
            'Stability Score': stability_score,
            'Stability Status': stability_status
        })
    
    # Create summary table
    results_df = pd.DataFrame(results)
    print("\n=== Summary of Results ===")
    print(results_df[['Method', 'CV Accuracy (10)', 'CV STD (10)', 
                     'CV Accuracy (9)', 'CV STD (9)', 
                     'Stability Score', 'Stability Status']])

if __name__ == "__main__":
    main()
