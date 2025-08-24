import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import FeatureAgglomeration
import xgboost as xgb
from scipy.stats import spearmanr
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('data.csv')
X = df.drop('vital.status', axis=1)
y = df['vital.status']

# Function to perform feature selection
def select_features(X, y, method='random_forest', n_features=10):
    selected_features = []
    feature_rankings = {}
    
    # Feature selection methods
    if method == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        feature_importances = model.feature_importances_
        feature_rankings = dict(zip(X.columns, feature_importances))
        indices = np.argsort(feature_importances)[::-1]
        selected_features = X.columns[indices[:n_features]].tolist()
        
    elif method == 'xgboost':
        model = xgb.XGBClassifier(random_state=42)
        model.fit(X, y)
        feature_importances = model.feature_importances_
        feature_rankings = dict(zip(X.columns, feature_importances))
        indices = np.argsort(feature_importances)[::-1]
        selected_features = X.columns[indices[:n_features]].tolist()
        
    elif method == 'feature_agglomeration':
        # Create clusters of features
        n_clusters = min(X.shape[1] // 2, 100)  # Using a reasonable number of clusters
        fa = FeatureAgglomeration(n_clusters=n_clusters)
        fa.fit(X)
        clusters = fa.labels_
        
        # Calculate variance for each feature
        variances = X.var().to_dict()
        
        # Create a list of (feature, variance, cluster) tuples
        feature_info = [(col, variances[col], clusters[i]) for i, col in enumerate(X.columns)]
        
        # Sort by variance (descending)
        feature_info.sort(key=lambda x: x[1], reverse=True)
        
        # Track selected clusters to avoid picking multiple features from same cluster
        selected_clusters = set()
        selected_features = []
        
        # Select top features across all clusters
        for feature, variance, cluster in feature_info:
            if len(selected_features) >= n_features:
                break
            
            if cluster not in selected_clusters:
                selected_features.append(feature)
                selected_clusters.add(cluster)
                feature_rankings[feature] = variance
        
        # If we need more features, allow multiple features from same cluster
        if len(selected_features) < n_features:
            remaining = [f for f, v, c in feature_info if f not in selected_features]
            for feature in remaining:
                if len(selected_features) >= n_features:
                    break
                selected_features.append(feature)
                feature_rankings[feature] = variances[feature]
        
    elif method == 'hvgs':
        variances = X.var().sort_values(ascending=False)
        feature_rankings = dict(zip(variances.index, variances))
        selected_features = variances.index[:n_features].tolist()
        
    elif method == 'spearman':
        correlations = []
        p_values = []
        for col in X.columns:
            corr, p_val = spearmanr(X[col], y)
            correlations.append(abs(corr))
            p_values.append(p_val)
            feature_rankings[col] = abs(corr)
        
        feature_corrs = pd.DataFrame({
            'feature': X.columns,
            'correlation': correlations,
            'p_value': p_values
        })
        
        feature_corrs = feature_corrs.sort_values(['correlation', 'p_value'], 
                                                ascending=[False, True])
        selected_features = feature_corrs['feature'][:n_features].tolist()
    
    return selected_features, feature_rankings

# Function to perform cross-validation
def perform_cv(X, y, features, cv_model='random_forest'):
    X_selected = X[features]
    
    if cv_model == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif cv_model == 'xgboost':
        model = xgb.XGBClassifier(random_state=42)
    
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
    return scores.mean()

# Function to calculate stability in feature selection
def calculate_stability(features_list):
    all_features = []
    for features in features_list:
        all_features.extend(features)
    
    counts = Counter(all_features)
    stability_score = sum([counts[f] for f in counts if counts[f] > 1]) / sum(counts.values())
    
    return stability_score, counts

# Define methods and their corresponding CV models
methods = {
    'random_forest': 'random_forest',
    'xgboost': 'xgboost',
    'feature_agglomeration': 'random_forest',
    'hvgs': 'random_forest',
    'spearman': 'random_forest'
}

# Step 1: Select top 10 features for each method
top10_features = {}
top10_cv_scores = {}
feature_rankings = {}

print("Top 10 Feature Selection:")
print("-" * 50)

for method, cv_model in methods.items():
    selected_features, rankings = select_features(X, y, method=method, n_features=10)
    top10_features[method] = selected_features
    feature_rankings[method] = rankings
    
    # Cross-validation with top 10 features
    cv_score = perform_cv(X, y, selected_features, cv_model=cv_model)
    top10_cv_scores[method] = cv_score
    
    print(f"{method.upper()} top 10 features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
    print(f"Cross-validation accuracy: {cv_score:.4f}")
    print("-" * 50)

# Step 2: Create reduced datasets by removing the top feature
top1_features = {}
reduced_datasets = {}
reduced_cv_scores = {}

print("\nTop 1 Feature and Reduced Dataset Creation:")
print("-" * 50)

for method in methods:
    # Get the top feature
    top1_features[method] = top10_features[method][0]
    
    # Create reduced dataset by removing the top feature
    reduced_datasets[method] = X.drop(top1_features[method], axis=1)
    
    # Cross-validate with only the top feature
    top1_cv = perform_cv(X, y, [top1_features[method]], cv_model=methods[method])
    
    print(f"{method.upper()}:")
    print(f"  Top 1 feature: {top1_features[method]}")
    print(f"  Top 1 CV accuracy: {top1_cv:.4f}")
    print(f"  Reduced dataset shape: {reduced_datasets[method].shape}")
    print("-" * 50)

# Step 3: Select top 9 features from the reduced datasets
top9_from_reduced = {}
top9_cv_scores = {}

print("\nTop 9 Features from Reduced Datasets:")
print("-" * 50)

for method, cv_model in methods.items():
    # Select top 9 features from reduced dataset
    selected_features, _ = select_features(reduced_datasets[method], y, method=method, n_features=9)
    top9_from_reduced[method] = selected_features
    
    # Cross-validation with top 9 features from reduced dataset
    cv_score = perform_cv(reduced_datasets[method], y, selected_features, cv_model=cv_model)
    top9_cv_scores[method] = cv_score
    
    print(f"{method.upper()} top 9 features from reduced dataset:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
    print(f"Cross-validation accuracy: {cv_score:.4f}")
    print("-" * 50)

# Step 4: Combine top 1 with top 9 from reduced dataset
combined_features = {}
combined_cv_scores = {}

print("\nCombined Features (Top 1 + Top 9 from reduced):")
print("-" * 50)

for method, cv_model in methods.items():
    # Combine top 1 with top 9 from reduced
    combined = [top1_features[method]] + top9_from_reduced[method]
    combined_features[method] = combined
    
    # Cross-validation with combined features
    cv_score = perform_cv(X, y, combined, cv_model=cv_model)
    combined_cv_scores[method] = cv_score
    
    print(f"{method.upper()} combined features:")
    print(f"  Top 1: {combined[0]}")
    print(f"  Next 9:")
    for i, feature in enumerate(combined[1:], 2):
        print(f"    {i}. {feature}")
    print(f"Cross-validation accuracy: {cv_score:.4f}")
    print("-" * 50)

# Calculate stability for different feature sets
print("\nFeature Selection Stability Analysis:")
print("-" * 50)

top10_stability, top10_counts = calculate_stability(top10_features.values())
print(f"Stability score for top 10 features: {top10_stability:.4f}")
print("Top 10 feature occurrence counts:")
for feature, count in top10_counts.most_common(20):
    print(f"  {feature}: {count}")
print("-" * 50)

combined_stability, combined_counts = calculate_stability(combined_features.values())
print(f"Stability score for combined features: {combined_stability:.4f}")
print("Combined feature occurrence counts:")
for feature, count in combined_counts.most_common(20):
    print(f"  {feature}: {count}")
print("-" * 50)

# Compare the performance of original top 10 vs combined features
print("\nComparison of Cross-validation Accuracy:")
print("-" * 50)
print("Method | Top 10 | Combined (Top 1 + Top 9 from reduced)")
print("-" * 50)
for method in methods:
    print(f"{method.ljust(15)} | {top10_cv_scores[method]:.4f} | {combined_cv_scores[method]:.4f}")
print("-" * 50)

# Calculate feature overlap between top 10 and combined selection
print("\nFeature Overlap Between Top 10 and Combined Selection:")
print("-" * 50)
for method in methods:
    overlap = set(top10_features[method]).intersection(set(combined_features[method]))
    overlap_percentage = len(overlap) / 10 * 100
    print(f"{method.ljust(15)}: {len(overlap)}/10 features ({overlap_percentage:.1f}%)")
    print(f"  Common features: {', '.join(overlap)}")
print("-" * 50)

# Check ranking stability for overlapping features
print("\nRanking Stability for Common Features:")
print("-" * 50)
for method in methods:
    common_features = set(top10_features[method]).intersection(set(combined_features[method]))
    if common_features:
        print(f"{method.upper()}:")
        for feature in common_features:
            top10_idx = top10_features[method].index(feature) + 1
            combined_idx = combined_features[method].index(feature) + 1
            rank_change = abs(top10_idx - combined_idx)
            print(f"  {feature}: Rank in top 10: {top10_idx}, Rank in combined: {combined_idx}, Change: {rank_change}")
    print("-" * 50)
