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
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('data.csv')

# Separate target and features
y = df['vital.status']
string_features = [col for col in df.columns if col != 'vital.status']
X = df[string_features]

print(f"Dataset shape: {X.shape}, Target: {y.value_counts().to_dict()}")

# 1. Feature Agglomeration
print("\nApplying Feature Agglomeration...")
agglo = FeatureAgglomeration(n_clusters=10)  # Changed to 10 clusters
agglo.fit(X)
feature_importances_agglo = np.zeros(len(string_features))
for i in range(10):  # Changed to 10
    feature_importances_agglo[np.where(agglo.labels_ == i)[0][0]] = 1
selected_features_agglo = [string_features[i] for i in np.argsort(feature_importances_agglo)[-10:]]  # Top 10

# 2. Generalized Linear Mixed Model (using GLM as a simpler alternative)
print("Applying Generalized Linear Model...")
glm_importance = []
for i in range(X.shape[1]):
    try:
        model = GLM(y == y.unique()[0], sm.add_constant(X.iloc[:, i]), family=Binomial())
        result = model.fit(disp=0)
        glm_importance.append(abs(result.params[1]))
    except:
        glm_importance.append(0)
selected_features_glm = [string_features[i] for i in np.argsort(glm_importance)[-10:]]  # Top 10

# 3. Highly Variable Gene Selection (adapting for general features)
print("Applying Highly Variable Feature Selection...")
variance = np.var(X, axis=0)
selected_features_hvgs = [string_features[i] for i in np.argsort(variance)[-10:]]  # Top 10

# 4. PCA
print("Applying PCA...")
pca = PCA(n_components=10)  # Changed to 10 components
pca.fit(X)
pca_importance = np.sum(np.abs(pca.components_), axis=0)
selected_features_pca = [string_features[i] for i in np.argsort(pca_importance)[-10:]]  # Top 10

# 5. ICA
print("Applying ICA...")
ica = FastICA(n_components=10, random_state=42)  # Changed to 10 components
ica.fit(X)
ica_importance = np.sum(np.abs(ica.components_), axis=0)
selected_features_ica = [string_features[i] for i in np.argsort(ica_importance)[-10:]]  # Top 10

# 6. Spearman Correlation
print("Applying Spearman Correlation...")
spearman_importance = []
# Convert target to numeric if needed
y_numeric = pd.factorize(y)[0]
for col in X.columns:
    correlation, _ = spearmanr(X[col], y_numeric)
    spearman_importance.append(abs(correlation))
selected_features_spearman = [string_features[i] for i in np.argsort(spearman_importance)[-10:]]  # Top 10

# Create union of all selected features and remove duplicates
all_selected_features = list(set(
    selected_features_agglo + 
    selected_features_glm + 
    selected_features_hvgs + 
    selected_features_pca + 
    selected_features_ica +
    selected_features_spearman
))

print(f"\nTotal unique features selected: {len(all_selected_features)}")

# Dictionary to store cross-validation results
cv_results = {}

# Perform cross-validation using Random Forest for each feature selection method
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

for method_name, features in methods.items():
    print(f"\nEvaluating {method_name} selected features:")
    
    # Create dataset with only selected features
    X_selected = X[features]
    
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf, X_selected, y, cv=skf, scoring='accuracy')
    
    # Store results
    cv_results[method_name] = {
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'all_scores': cv_scores,
        'selected_features': features
    }
    
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Selected features: {', '.join(features)}")

# Print overall best method
best_method = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_accuracy'])
print("\n" + "="*50)
print(f"Best feature selection method: {best_method}")
print(f"Best accuracy: {cv_results[best_method]['mean_accuracy']:.4f} ± {cv_results[best_method]['std_accuracy']:.4f}")
print(f"Selected features: {', '.join(methods[best_method])}")

# Train final model using the best set of features
print("\nTraining final model with best features...")
best_features = methods[best_method]
X_best = X[best_features]
final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
final_scores = cross_val_score(final_rf, X_best, y, cv=5, scoring='accuracy')
print(f"Final model accuracy: {final_scores.mean():.4f} ± {final_scores.std():.4f}")

# Feature importance of the best model
print("\nFeature importance of final model:")
final_rf.fit(X_best, y)
feature_importance = pd.DataFrame({
    'Feature': best_features,
    'Importance': final_rf.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)
