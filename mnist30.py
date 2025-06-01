import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from scipy.stats import spearmanr
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data
y = mnist.target.astype('int')

# Since the entire dataset is very large, we'll use a random subset
# Adjust sample_size based on your available computational resources
sample_size = 70000  # Using 10,000 samples
np.random.seed(42)
indices = np.random.choice(X.shape[0], sample_size, replace=False)

# Properly sample from pandas DataFrame
X_sample = X.iloc[indices].values
y_sample = y[indices]

# Create feature names for better reference
string_features = [f'pixel_{i}' for i in range(X_sample.shape[1])]

print(f"Dataset shape: {X_sample.shape}, Target classes: {np.unique(y_sample)}")

# 1. Feature Agglomeration
print("\nApplying Feature Agglomeration...")
agglo = FeatureAgglomeration(n_clusters=30)  # Changed to 30 clusters
agglo.fit(X_sample)
feature_importances_agglo = np.zeros(len(string_features))
for i in range(30):  # Changed to 30
    feature_importances_agglo[np.where(agglo.labels_ == i)[0][0]] = 1
selected_features_agglo_idx = np.argsort(feature_importances_agglo)[-30:]  # Top 30
selected_features_agglo = [string_features[i] for i in selected_features_agglo_idx]

# 2. Generalized Linear Mixed Model (using GLM as a simpler alternative)
print("Applying Generalized Linear Model...")
# For MNIST with many features, we'll use a simplified approach
glm_importance = []
for i in range(0, X_sample.shape[1], 10):  # Sample every 10th feature to speed up
    try:
        # Binary target for GLM: is it digit 5 or not
        binary_target = (y_sample == 5).astype(int)
        model = GLM(binary_target, sm.add_constant(X_sample[:, i]), family=Binomial())
        result = model.fit(disp=0)
        glm_importance.append(abs(result.params[1]))
    except:
        glm_importance.append(0)
    
# Fill the remaining positions with zeros
full_glm_importance = np.zeros(X_sample.shape[1])
for idx, val in enumerate(glm_importance):
    full_glm_importance[idx * 10] = val
    
selected_features_glm_idx = np.argsort(full_glm_importance)[-30:]  # Top 30
selected_features_glm = [string_features[i] for i in selected_features_glm_idx]

# 3. Highly Variable Feature Selection
print("Applying Highly Variable Feature Selection...")
variance = np.var(X_sample, axis=0)
selected_features_hvgs_idx = np.argsort(variance)[-30:]  # Top 30
selected_features_hvgs = [string_features[i] for i in selected_features_hvgs_idx]

# 4. PCA
print("Applying PCA...")
pca = PCA(n_components=30)  # Changed to 30 components
pca.fit(X_sample)
pca_importance = np.sum(np.abs(pca.components_), axis=0)
selected_features_pca_idx = np.argsort(pca_importance)[-30:]  # Top 30
selected_features_pca = [string_features[i] for i in selected_features_pca_idx]

# 5. ICA
print("Applying ICA...")
ica = FastICA(n_components=30, random_state=42, max_iter=1000)  # Changed to 30 components
ica.fit(X_sample)
ica_importance = np.sum(np.abs(ica.components_), axis=0)
selected_features_ica_idx = np.argsort(ica_importance)[-30:]  # Top 30
selected_features_ica = [string_features[i] for i in selected_features_ica_idx]

# 6. Spearman Correlation
print("Applying Spearman Correlation...")
spearman_importance = []
# Calculate correlation for every 10th feature to save time
for i in range(0, X_sample.shape[1], 10):
    correlation, _ = spearmanr(X_sample[:, i], y_sample)
    spearman_importance.append(abs(correlation))

# Fill the remaining positions with zeros
full_spearman_importance = np.zeros(X_sample.shape[1])
for idx, val in enumerate(spearman_importance):
    full_spearman_importance[idx * 10] = val
    
selected_features_spearman_idx = np.argsort(full_spearman_importance)[-30:]  # Top 30
selected_features_spearman = [string_features[i] for i in selected_features_spearman_idx]

# Create union of all selected features indices
all_selected_indices = list(set(
    selected_features_agglo_idx.tolist() + 
    selected_features_glm_idx.tolist() + 
    selected_features_hvgs_idx.tolist() + 
    selected_features_pca_idx.tolist() + 
    selected_features_ica_idx.tolist() +
    selected_features_spearman_idx.tolist()
))

all_selected_features = [string_features[i] for i in all_selected_indices]

print(f"\nTotal unique features selected: {len(all_selected_features)}")

# Dictionary to store cross-validation results
cv_results = {}

# Perform cross-validation using Random Forest for each feature selection method
methods_indices = {
    'Feature Agglomeration': selected_features_agglo_idx,
    'GLM': selected_features_glm_idx,
    'HVGS': selected_features_hvgs_idx,
    'PCA': selected_features_pca_idx,
    'ICA': selected_features_ica_idx,
    'Spearman': selected_features_spearman_idx
}

methods = {
    'Feature Agglomeration': selected_features_agglo,
    'GLM': selected_features_glm,
    'HVGS': selected_features_hvgs,
    'PCA': selected_features_pca,
    'ICA': selected_features_ica,
    'Spearman': selected_features_spearman
}

# Stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for method_name, indices in methods_indices.items():
    print(f"\nEvaluating {method_name} selected features:")
    
    # Create dataset with only selected features
    X_selected = X_sample[:, indices]
    
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf, X_selected, y_sample, cv=skf, scoring='accuracy')
    
    # Store results
    cv_results[method_name] = {
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'all_scores': cv_scores,
        'selected_features': methods[method_name]
    }
    
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Selected features: {', '.join(methods[method_name][:5])}... (first 5 shown)")

# Print overall best method
best_method = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_accuracy'])
print("\n" + "="*50)
print(f"Best feature selection method: {best_method}")
print(f"Best accuracy: {cv_results[best_method]['mean_accuracy']:.4f} ± {cv_results[best_method]['std_accuracy']:.4f}")
print(f"Selected features (first 5 shown): {', '.join(methods[best_method][:5])}...")

# Train final model using the best set of features
print("\nTraining final model with best features...")
best_indices = methods_indices[best_method]
X_best = X_sample[:, best_indices]
final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
final_scores = cross_val_score(final_rf, X_best, y_sample, cv=5, scoring='accuracy')
print(f"Final model accuracy: {final_scores.mean():.4f} ± {final_scores.std():.4f}")

# Feature importance of the best model
print("\nFeature importance of final model (top 10):")
final_rf.fit(X_best, y_sample)
feature_importance = pd.DataFrame({
    'Feature': methods[best_method],
    'Importance': final_rf.feature_importances_
}).sort_values('Importance', ascending=False).head(10)  # Showing top 10 features
print(feature_importance)
