#!/usr/bin/env python3
"""
Standalone experiment runner — called by the research agent via Bash.

Usage
-----
    python run_experiment.py \
        --dataset experiments/.../feature_dataset.csv \
        --algorithm xgboost \
        --params '{"n_estimators": 200, "max_depth": 6}' \
        --feature-groups keystroke,mouse,navigation,temporal,derived \
        --cv-folds 5 \
        --label "xgb_deep"

Prints a single JSON line to stdout with the experiment results.
All warnings/logs go to stderr so stdout stays clean for JSON parsing.
"""

import argparse
import json
import sys
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.dataset.features import filter_features
from src.evaluation.metrics import evaluate_model
from src.models.classical import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",        required=True, help="Path to feature_dataset.csv")
    p.add_argument("--algorithm",      default="random_forest")
    p.add_argument("--params",         default="{}", help="JSON hyperparameters")
    p.add_argument("--feature-groups", default="keystroke,mouse,navigation,temporal,derived",
                   help="Comma-separated groups to include")
    p.add_argument("--cv-folds",       type=int, default=5)
    p.add_argument("--label",          default="experiment")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load dataset ──────────────────────────────────────────────────
    try:
        feat_df = pd.read_csv(args.dataset)
    except Exception as e:
        print(json.dumps({"error": f"Cannot load dataset: {e}"}))
        sys.exit(1)

    # ── Build feature config ──────────────────────────────────────────
    active_groups = {g.strip().lower() for g in args.feature_groups.split(",")}
    feature_config = {
        "use_keystroke":  "keystroke"  in active_groups,
        "use_mouse":      "mouse"      in active_groups,
        "use_navigation": "navigation" in active_groups,
        "use_temporal":   "temporal"   in active_groups,
        "use_derived":    "derived"    in active_groups,
    }

    filtered = filter_features(feat_df, feature_config)
    meta_cols = ["user_id", "target_user", "actual_user", "is_genuine", "session_id"]
    feat_cols = [c for c in filtered.columns if c not in meta_cols]

    X = filtered[feat_cols].fillna(0).values.astype(np.float32)
    y = filtered["is_genuine"].values.astype(int)

    if len(feat_cols) == 0:
        print(json.dumps({"error": "No features after filtering — check --feature-groups"}))
        sys.exit(1)

    # ── Parse hyperparameters ─────────────────────────────────────────
    try:
        model_params = json.loads(args.params)
    except json.JSONDecodeError:
        model_params = {}

    # ── Cross-validation ──────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    all_scores, all_labels = [], []
    fold_eers = []
    importances_list = []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        try:
            model = build_model(args.algorithm, params=dict(model_params))
            model.fit(X_tr, y_tr, feature_names=feat_cols)
            scores = model.predict_proba(X_te)
            all_scores.extend(scores.tolist())
            all_labels.extend(y_te.tolist())

            fold_m = evaluate_model(y_te, scores)
            fold_eers.append(fold_m["eer"])

            imp = model.get_feature_importance()
            if imp:
                importances_list.append(imp)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}))
            sys.exit(1)

    # ── Aggregate metrics ─────────────────────────────────────────────
    metrics = evaluate_model(np.array(all_labels), np.array(all_scores))

    # Average feature importances across folds
    avg_importance: dict = {}
    if importances_list:
        for feat in importances_list[0]:
            vals = [d.get(feat, 0.0) for d in importances_list]
            avg_importance[feat] = round(float(np.mean(vals)), 6)

    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15]

    result = {
        "label":         args.label,
        "algorithm":     args.algorithm,
        "model_params":  model_params,
        "feature_config": feature_config,
        "n_features":    len(feat_cols),
        "metrics": {k: round(float(v), 5) for k, v in metrics.items()},
        "fold_eer_std":  round(float(np.std(fold_eers)), 5),
        "top_features":  [{"feature": f, "importance": v} for f, v in top_features],
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
