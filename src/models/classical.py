"""
Classical ML models for behavioral fingerprinting.

All models expose a unified interface:
    fit(X, y, feature_names=None) → self
    predict_proba(X) → np.ndarray of scores in [0, 1]
    get_feature_importance() → dict or None

Available algorithms
~~~~~~~~~~~~~~~~~~~~
    random_forest    – RF classifier (binary, balanced class weight)
    xgboost          – XGBoost gradient boosting
    lightgbm         – LightGBM (fast, often best on tabular data)
    isolation_forest – One-class anomaly detection (trained on genuine only)
    svm              – SVM with RBF kernel
    gradient_boost   – sklearn GradientBoostingClassifier
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BehaviorModel:
    """Common interface for all behavioral fingerprinting models."""

    def __init__(self, name: str, params: Optional[dict] = None):
        self.name = name
        self.params: dict = params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return None


# ---------------------------------------------------------------------------
# Concrete models
# ---------------------------------------------------------------------------

class RandomForestModel(BehaviorModel):
    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        if params:
            defaults.update(params)
        super().__init__("RandomForest", defaults)

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        Xs = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(Xs, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def get_feature_importance(self):
        if self.model is not None and self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return None


class XGBoostModel(BehaviorModel):
    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )
        if params:
            defaults.update(params)
        super().__init__("XGBoost", defaults)

    def fit(self, X, y, feature_names=None):
        import xgboost as xgb
        self.feature_names = feature_names
        Xs = self.scaler.fit_transform(X)
        pos_weight = float((y == 0).sum()) / max(1, (y == 1).sum())
        p = {**self.params, "scale_pos_weight": pos_weight}
        self.model = xgb.XGBClassifier(**p)
        self.model.fit(Xs, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def get_feature_importance(self):
        if self.model is not None and self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return None


class LightGBMModel(BehaviorModel):
    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.07,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        if params:
            defaults.update(params)
        super().__init__("LightGBM", defaults)

    def fit(self, X, y, feature_names=None):
        import lightgbm as lgb
        self.feature_names = feature_names
        Xs = self.scaler.fit_transform(X)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(Xs, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def get_feature_importance(self):
        if self.model is not None and self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return None


class GradientBoostModel(BehaviorModel):
    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        if params:
            defaults.update(params)
        super().__init__("GradientBoost", defaults)

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        Xs = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(**self.params)
        self.model.fit(Xs, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def get_feature_importance(self):
        if self.model is not None and self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return None


class IsolationForestModel(BehaviorModel):
    """
    One-class anomaly detection: trained ONLY on genuine sessions.
    Scores impostors as anomalies.
    """
    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            n_estimators=150,
            contamination=0.1,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )
        if params:
            defaults.update(params)
        super().__init__("IsolationForest", defaults)

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        X_genuine = X[y == 1]
        Xs = self.scaler.fit_transform(X_genuine)
        self.model = IsolationForest(**self.params)
        self.model.fit(Xs)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        raw = self.model.decision_function(Xs)
        lo, hi = raw.min(), raw.max()
        if hi > lo:
            return (raw - lo) / (hi - lo)
        return np.full(len(raw), 0.5)


class SVMModel(BehaviorModel):
    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
        if params:
            defaults.update(params)
        super().__init__("SVM", defaults)

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        Xs = self.scaler.fit_transform(X)
        self.model = SVC(**self.params)
        self.model.fit(Xs, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    "random_forest":   RandomForestModel,
    "xgboost":         XGBoostModel,
    "lightgbm":        LightGBMModel,
    "gradient_boost":  GradientBoostModel,
    "isolation_forest": IsolationForestModel,
    "svm":             SVMModel,
}


def build_model(algorithm: str, params: Optional[dict] = None) -> BehaviorModel:
    """Factory function: create a model by algorithm name."""
    if algorithm not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[algorithm](params=params)
