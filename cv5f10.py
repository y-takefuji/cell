import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.decomposition import PCA

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

# 1. Lasso Feature Selection (updated implementation)
print("\n===== Lasso Feature Selection =====")
# Convert pandas DataFrame to numpy array for consistency
X_numpy = X.values
lasso = Lasso(alpha=0.01, max_iter=1000, random_state=42)
lasso.fit(X_numpy, y)

# Get feature importance from LASSO coefficients
lasso_importance = np.abs(lasso.coef_)
top_lasso_indices = np.argsort(-lasso_importance)[:k_features]
top_lasso_features = X.columns[top_lasso_indices].tolist()
X_lasso = X[top_lasso_features]

print(f"Top {k_features} features from Lasso:")
for feature in top_lasso_features:
    print(f"- {feature}")

# 2. Logistic Regression with L1 penalty (updated implementation)
print("\n===== Logistic Regression with L1 penalty =====")
# Similarly use numpy array approach for consistency
logistic_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42)
logistic_l1.fit(X_numpy, y)

# Get feature importance from logistic coefficients (similar to Lasso approach)
logistic_importance = np.abs(logistic_l1.coef_[0])
top_logistic_indices = np.argsort(-logistic_importance)[:k_features]
top_logistic_l1_features = X.columns[top_logistic_indices].tolist()
X_logistic = X[top_logistic_l1_features]

print(f"Top {k_features} features from Logistic Regression with L1 penalty:")
for feature in top_logistic_l1_features:
    print(f"- {feature}")

# 3. PCA Feature Selection
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

# 4. HVGs - Highly Variable Genes
print("\n===== HVGs Feature Selection =====")
def hvgs_feature_selection(X, top_n=10):
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    return top_features, X[top_features]

hvgs_features, X_hvgs = hvgs_feature_selection(X)
print(f"Top {len(hvgs_features)} features from HVGs:")
for feature in hvgs_features:
    print(f"- {feature}")

# 5. Random Forest Feature Selection (RF-RF)
print("\n===== Random Forest Feature Selection =====")
def rf_feature_selection(X, y, top_n=10):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.nlargest(top_n).index.tolist()
    return top_features, X[top_features]

rf_features, X_rf = rf_feature_selection(X, y)
print(f"Top {len(rf_features)} features from Random Forest:")
for feature in rf_features:
    print(f"- {feature}")

# Evaluate each feature selection method
print("\n===== Evaluation Results =====")
lasso_acc, lasso_std = evaluate_features(X_lasso, y, "Lasso")
logistic_acc, logistic_std = evaluate_features(X_logistic, y, "Logistic")
pca_acc, pca_std = evaluate_features(X_pca, y, "PCA")
hvgs_acc, hvgs_std = evaluate_features(X_hvgs, y, "HVGs")
rf_acc, rf_std = evaluate_features(X_rf, y, "Random Forest")

# Compare model performances
results = {
    'Lasso': lasso_acc,
    'Logistic': logistic_acc,
    'PCA': pca_acc,
    'HVGs': hvgs_acc,
    'Random Forest': rf_acc
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
accuracies = [lasso_acc, logistic_acc, pca_acc, hvgs_acc, rf_acc]
errors = [lasso_std, logistic_std, pca_std, hvgs_std, rf_std]
methods = ["Lasso", "Logistic", "PCA", "HVGs", "Random Forest"]
best_idx = np.argmax(accuracies)
best_method = methods[best_idx]
best_feats = [top_lasso_features, top_logistic_l1_features, pca_features, hvgs_features, rf_features][best_idx]

print(f"\n===== SUMMARY =====")
print(f"Best method: {best_method} ({accuracies[best_idx]:.4f} ± {errors[best_idx]:.4f})")
print("Top 10 features from", best_method, ":", best_feats)
