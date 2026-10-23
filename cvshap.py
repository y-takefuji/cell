"""
Feature selection + leakage-free cross-validation pipeline
-----------------------------------------------------------
Data      : data.csv  (binary target column 'vital.status', values 0/1)
            ground0 = alive (0), ground1 = death (1)
Methods   : RF, RF-SHAP, XGB, XGB-SHAP, GMM, HVGS, Spearman, DEA

Workflow
  1. Rank all features on the FULL feature set; take top-9.
  2. Leakage-free 5-fold CV using top-9 features.
     Feature selection is re-done on the TRAINING part of each fold,
     then the matching classifier is fit on those columns and scored
     on the held-out part.
     CV classifier:
       RF, RF-SHAP            -> RandomForestClassifier
       XGB, XGB-SHAP          -> XGBClassifier
       GMM, HVGS, Spearman    -> RandomForestClassifier
       DEA                    -> RandomForestClassifier
  3. Top-8 features (all methods):
       Remove the highest-ranked feature, refit the scorer on the
       reduced dataset, and reselect top-8.

DEA (death-associated features):
  - Binary features : Fisher's exact test, ground1 vs ground0
  - Continuous      : Welch's t-test,      ground1 vs ground0
  - Multiple testing: Benjamini-Hochberg adjusted p-values (p_adj)
  - Score           : -p_adj  (most significant death-associated feature = highest)

No scaling, no normalization, no transformation of any kind.
Default model settings are used throughout.
SHAP: RF-SHAP and XGB-SHAP compute SHAP values on 100 randomly
selected instances (fixed seed), not the full dataset.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    shap = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

RANDOM_STATE = 42
N_SPLITS = 5
N_SHAP_SAMPLES = 100
TOP_K_MAIN = 9          # top-k from full feature set
TOP_K_REDUCED = 8       # top-k from reduced dataset
TARGET = "vital.status"
DATA_PATH = "data.csv"
RESULT_PATH = "result.csv"

GROUP_0 = 0             # ground0 = alive
GROUP_1 = 1             # ground1 = death

np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
def load_data(path=DATA_PATH, target=TARGET):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}.")
    df = df.dropna(subset=[target])
    y = df[target].astype(int)
    if not set(y.unique()).issubset({GROUP_0, GROUP_1}):
        raise ValueError("Target must be binary (0/1).")
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    X = X.fillna(X.median())
    return X, y


# ---------------------------------------------------------------------
# 2. Feature-scoring functions (higher score = more important)
# ---------------------------------------------------------------------
def score_rf(X, y):
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns)


def score_rf_shap(X, y):
    """RF-SHAP: mean |SHAP| on 100 randomly selected instances."""
    if shap is None:
        raise ImportError("Install shap: pip install shap")
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)

    n = min(N_SHAP_SAMPLES, len(X))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=n, replace=False)
    X_sample = X.iloc[idx]

    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]
    elif np.ndim(sv) == 3:
        sv = sv[:, :, 1]
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns)


def score_xgb(X, y):
    if XGBClassifier is None:
        raise ImportError("Install xgboost: pip install xgboost")
    xgb = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X, y)
    return pd.Series(xgb.feature_importances_, index=X.columns)


def score_xgb_shap(X, y):
    """XGB-SHAP: mean |SHAP| on 100 randomly selected instances."""
    if XGBClassifier is None or shap is None:
        raise ImportError("Install xgboost and shap.")
    xgb = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X, y)

    n = min(N_SHAP_SAMPLES, len(X))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=n, replace=False)
    X_sample = X.iloc[idx]

    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns)


def score_gmm(X, y=None):
    """GMM: each raw column is fit with a default 2-component
    GaussianMixture; score = |difference of component means|."""
    scores = {}
    for col in X.columns:
        v = X[col].values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
        gmm.fit(v)
        means = np.sort(gmm.means_.ravel())
        scores[col] = abs(means[1] - means[0])
    return pd.Series(scores)


def score_hvgs(X, y=None):
    """HVGS: score = variance of each raw feature."""
    return X.var()


def score_spearman(X, y):
    """Spearman: absolute rank correlation of each raw feature with target."""
    out = {}
    for col in X.columns:
        rho, _ = stats.spearmanr(X[col], y)
        out[col] = abs(rho) if not np.isnan(rho) else 0.0
    return pd.Series(out)


def score_dea(X, y):
    """DEA (death-associated features), following the association analysis:
      - binary feature    : Fisher's exact test, ground1 vs ground0
      - continuous feature: Welch's t-test,      ground1 vs ground0
      - Benjamini-Hochberg correction across all features (p_adj)
    Score = -p_adj, so the most significant death-associated features
    rank highest. No log or other transformation is applied."""
    y = np.asarray(y)
    X_g1 = X[y == GROUP_1]      # ground1 = death
    X_g0 = X[y == GROUP_0]      # ground0 = alive

    names, p_values = [], []
    for col in X.columns:
        x1 = X_g1[col].dropna()
        x0 = X_g0[col].dropna()
        if len(x0) < 2 or len(x1) < 2:
            continue

        unique_vals = pd.unique(X[col].dropna())
        is_binary = set(unique_vals).issubset({0, 1})

        if is_binary:
            a = (x1 == 1).sum()
            b = (x1 == 0).sum()
            c = (x0 == 1).sum()
            d = (x0 == 0).sum()
            _, p = stats.fisher_exact([[a, b], [c, d]])
        else:
            _, p = stats.ttest_ind(x1, x0, equal_var=False)

        names.append(col)
        p_values.append(p)

    if not names:
        return pd.Series(dtype=float)

    p_adj = multipletests(np.asarray(p_values), method="fdr_bh")[1]
    scores = pd.Series(-p_adj, index=names)

    # Features that could not be tested are placed last
    missing = [c for c in X.columns if c not in names]
    if missing:
        scores = pd.concat([scores, pd.Series(-np.inf, index=missing)])
    return scores


SCORERS = {
    "RF": score_rf,
    "RF-SHAP": score_rf_shap,
    "XGB": score_xgb,
    "XGB-SHAP": score_xgb_shap,
    "GMM": score_gmm,
    "HVGS": score_hvgs,
    "Spearman": score_spearman,
    "DEA": score_dea,
}

# Classifier used for cross-validation, per method
CV_CLASSIFIER = {
    "RF": "RF",
    "RF-SHAP": "RF",
    "XGB": "XGB",
    "XGB-SHAP": "XGB",
    "GMM": "RF",
    "HVGS": "RF",
    "Spearman": "RF",
    "DEA": "RF",
}


def make_cv_classifier(method):
    """Return a fresh default classifier matching the method's CV assignment."""
    if CV_CLASSIFIER[method] == "XGB":
        if XGBClassifier is None:
            raise ImportError("Install xgboost: pip install xgboost")
        return XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)


def rank_features(X, y, method):
    """Feature names sorted by score, highest first."""
    scores = SCORERS[method](X, y)
    return scores.sort_values(ascending=False, kind="mergesort").index.tolist()


def select_top_k(X, y, k, method):
    return rank_features(X, y, method)[:k]


# ---------------------------------------------------------------------
# 3. Cross-validation (feature selection repeated inside each fold)
# ---------------------------------------------------------------------
def ci95(scores):
    scores = np.asarray(scores)
    n = len(scores)
    mean = scores.mean()
    se = scores.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t * se, mean + t * se


def cv_accuracy(X, y, k, method):
    """For each fold: select top-k on training part, fit matching classifier
    on those columns, score on held-out part."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    accs = []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        selected = select_top_k(X_tr, y_tr, k, method)

        clf = make_cv_classifier(method)
        clf.fit(X_tr[selected], y_tr)
        accs.append(clf.score(X_te[selected], y_te))
    return np.asarray(accs)


# ---------------------------------------------------------------------
# 4. Top-8 derivation (same procedure for every method)
# ---------------------------------------------------------------------
def top8_features(X, y, method, ranked_full):
    """Remove the highest-ranked feature, refit the scorer on the reduced
    dataset, and reselect top-8."""
    highest = ranked_full[0]
    X_reduced = X.drop(columns=[highest])
    return select_top_k(X_reduced, y, TOP_K_REDUCED, method)


# ---------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------
def main():
    X, y = load_data()
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} numeric features")

    rows = []
    for method in SCORERS:
        try:
            ranked_full = rank_features(X, y, method)
            top9 = ranked_full[:TOP_K_MAIN]
            top8 = top8_features(X, y, method, ranked_full)

            acc9 = cv_accuracy(X, y, k=TOP_K_MAIN, method=method)
            m9, lo9, hi9 = ci95(acc9)

            rows.append({
                "method": method,
                "CV_classifier": CV_CLASSIFIER[method],
                "CV_accuracy_top9": round(m9, 3),
                "CI95": f"[{lo9:.3f}, {hi9:.3f}]",
                "top_9_features": "; ".join(top9),
                "top_8_features": "; ".join(top8),
            })
            print(f"[{method}] ({CV_CLASSIFIER[method]}) top-9 CV acc = "
                  f"{m9:.3f} 95% CI [{lo9:.3f}, {hi9:.3f}]")
        except Exception as e:
            print(f"[{method}] skipped: {e}")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved summary to {RESULT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()