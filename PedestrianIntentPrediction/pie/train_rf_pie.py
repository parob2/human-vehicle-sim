#!/usr/bin/env python3
"""Train Ma & Rong (2022) 7-feature Random Forest on extracted PIE features."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pie.config_pie import (
    DECISION_THRESHOLD,
    DEFAULT_FEATURES_JSONL,
    DEFAULT_MODEL_PATH_PIE,
    FEATURE_NAMES_PIE,
    MODEL_VERSION_PIE,
    RF_CLASS_WEIGHT,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    TARGET_KEY_PIE,
    TRAIN_TEST_SPLIT,
)
from pie.pie_dataset_loader import load_pie_features_from_jsonl
from split_metrics import compute_binary_metrics, split_train_test_by_group, summarize_split


def tune_n_estimators(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_estimators_list: List[int],
    cv_folds: int = 10,
) -> Dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    n_splits = min(cv_folds, len(unique_groups))
    if n_splits < 2:
        return {"best_n_estimators": RF_N_ESTIMATORS, "cv_scores": {}}

    cv = GroupKFold(n_splits=n_splits)
    scores: Dict[int, float] = {}
    for n_est in n_estimators_list:
        fold_acc: List[float] = []
        for tr_idx, te_idx in cv.split(X, y, groups):
            rf = RandomForestClassifier(
                n_estimators=n_est,
                class_weight=RF_CLASS_WEIGHT,
                random_state=RF_RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X[tr_idx], y[tr_idx])
            proba = rf.predict_proba(X[te_idx])[:, 1]
            pred = (proba >= DECISION_THRESHOLD).astype(int)
            fold_acc.append(float(compute_binary_metrics(y[te_idx], pred, proba)["accuracy"]))
        if fold_acc:
            scores[n_est] = float(np.mean(fold_acc))

    if not scores:
        return {"best_n_estimators": RF_N_ESTIMATORS, "cv_scores": {}}

    best_n = max(scores, key=scores.get)
    return {"best_n_estimators": best_n, "cv_scores": {str(k): v for k, v in scores.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ma-Rong RF on PIE 7-feature JSONL")
    parser.add_argument("--features-jsonl", action="append", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_MODEL_PATH_PIE.parent))
    parser.add_argument("--n-estimators", type=int, default=RF_N_ESTIMATORS)
    parser.add_argument("--tune-estimators", action="store_true", help="10-fold GroupKFold sweep (paper §3.2)")
    parser.add_argument("--test-size", type=float, default=TRAIN_TEST_SPLIT)
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=RF_RANDOM_STATE)
    args = parser.parse_args()

    jsonl_paths = args.features_jsonl if args.features_jsonl else [str(DEFAULT_FEATURES_JSONL)]
    X, y, groups, meta = load_pie_features_from_jsonl(jsonl_paths)
    print(f"Loaded {len(y)} PIE samples, positive_rate={np.mean(y):.3f}, dim={X.shape[1]}")

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Need both classes for binary RF; got labels {dict(zip(*np.unique(y, return_counts=True)))}. "
            "Include PIE sets with intention_prob <= 0.5 (e.g. set02, set03, set04)."
        )

    tune_result = None
    n_estimators = args.n_estimators
    if args.tune_estimators:
        est_list = list(range(10, 501, 10))
        tune_result = tune_n_estimators(X, y, groups, n_estimators_list=est_list)
        n_estimators = int(tune_result["best_n_estimators"])
        print(f"CV tuning: best n_estimators={n_estimators} (accuracy metric, paper §3.2)")

    X_train, X_test, y_train, y_test, train_idx, test_idx = split_train_test_by_group(
        X, y, groups, test_size=args.test_size, random_state=args.random_state, stratify=True
    )
    split_info = summarize_split(groups, y, train_idx, test_idx)
    if split_info["group_leakage"]:
        raise RuntimeError(f"Group leakage: {split_info['leaked_groups']}")
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            f"Stratified split missing a class: train={np.unique(y_train)} test={np.unique(y_test)}. "
            "Add more negative clips (set02/video_0001, set04/video_0001)."
        )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=RF_CLASS_WEIGHT,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    train_pred = (train_proba >= args.threshold).astype(int)
    test_pred = (test_proba >= args.threshold).astype(int)

    train_metrics = compute_binary_metrics(y_train, train_pred, train_proba)
    test_metrics = compute_binary_metrics(y_test, test_pred, test_proba)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "rf_ma_rong_pie.joblib")
    payload = {
        "model": model,
        "model_type": "ma_rong_pie_7",
        "version": MODEL_VERSION_PIE,
        "feature_dim": X.shape[1],
        "feature_names": FEATURE_NAMES_PIE,
        "target_key": TARGET_KEY_PIE,
        "threshold": args.threshold,
        "n_estimators": n_estimators,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "split_summary": split_info,
        "tune_result": tune_result,
        "dataset": "PIE",
        "paper_reference": {
            "accuracy": 0.895,
            "precision": 0.975,
            "recall": 0.951,
            "f1": 0.963,
            "roc_auc": 0.992,
            "n_estimators_paper": 320,
        },
    }
    joblib.dump(payload, model_path)

    metrics_path = os.path.join(args.out_dir, "train_metrics_pie.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": model_path,
                "n_estimators": n_estimators,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "split_summary": split_info,
                "tune_result": tune_result,
            },
            f,
            indent=2,
        )

    print(f"Model saved: {model_path}")
    print(
        f"Test accuracy={test_metrics['accuracy']:.3f} "
        f"recall={test_metrics['recall']:.3f} "
        f"precision={test_metrics['precision']:.3f} "
        f"f1={test_metrics['f1']:.3f} "
        f"roc_auc={test_metrics.get('roc_auc', 'n/a')}"
    )


if __name__ == "__main__":
    main()
