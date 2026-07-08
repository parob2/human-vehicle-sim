#!/usr/bin/env python3
"""Train Ma & Rong (2022) 9-feature PIE RF (7 paper features + trajectory heading)."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pie.config_pie import (
    DECISION_THRESHOLD,
    DEFAULT_FEATURES_JSONL_HEADING,
    DEFAULT_MODEL_PATH_PIE_HEADING,
    FEATURE_NAMES_PIE_HEADING,
    MODEL_VERSION_PIE_HEADING,
    RF_CLASS_WEIGHT,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    TARGET_KEY_PIE,
    TRAIN_TEST_SPLIT,
)
from pie.pie_dataset_loader import load_pie_heading_features_from_jsonl
from pie.train_rf_pie import tune_n_estimators
from split_metrics import compute_binary_metrics, split_train_test_by_group, summarize_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ma-Rong PIE RF with trajectory heading")
    parser.add_argument("--features-jsonl", action="append", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_MODEL_PATH_PIE_HEADING.parent))
    parser.add_argument("--n-estimators", type=int, default=RF_N_ESTIMATORS)
    parser.add_argument("--tune-estimators", action="store_true")
    parser.add_argument("--test-size", type=float, default=TRAIN_TEST_SPLIT)
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=RF_RANDOM_STATE)
    args = parser.parse_args()

    jsonl_paths = args.features_jsonl if args.features_jsonl else [str(DEFAULT_FEATURES_JSONL_HEADING)]
    X, y, groups, meta = load_pie_heading_features_from_jsonl(jsonl_paths)
    print(
        f"Loaded {len(y)} PIE heading samples, positive_rate={np.mean(y):.3f}, dim={X.shape[1]}"
    )

    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes for binary RF.")

    tune_result = None
    n_estimators = args.n_estimators
    if args.tune_estimators:
        est_list = list(range(10, 501, 10))
        tune_result = tune_n_estimators(X, y, groups, n_estimators_list=est_list)
        n_estimators = int(tune_result["best_n_estimators"])
        print(f"CV tuning: best n_estimators={n_estimators}")

    X_train, X_test, y_train, y_test, train_idx, test_idx = split_train_test_by_group(
        X, y, groups, test_size=args.test_size, random_state=args.random_state, stratify=True
    )
    split_info = summarize_split(groups, y, train_idx, test_idx)
    if split_info["group_leakage"]:
        raise RuntimeError(f"Group leakage: {split_info['leaked_groups']}")
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            f"Stratified split missing a class: train={np.unique(y_train)} test={np.unique(y_test)}"
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
    model_path = os.path.join(args.out_dir, "rf_ma_rong_pie_heading.joblib")
    payload = {
        "model": model,
        "model_type": "ma_rong_pie_9_heading",
        "version": MODEL_VERSION_PIE_HEADING,
        "feature_dim": int(X.shape[1]),
        "feature_names": FEATURE_NAMES_PIE_HEADING,
        "target_key": TARGET_KEY_PIE,
        "threshold": args.threshold,
        "n_estimators": n_estimators,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "split_summary": split_info,
        "tune_result": tune_result,
        "dataset": "PIE_full_heading" if "full" in str(jsonl_paths[0]) else "PIE_heading",
        "baseline_reference": {
            "model": "rf_ma_rong_pie.joblib",
            "feature_dim": 7,
            "test_accuracy": 0.949,
            "test_precision": 0.943,
            "test_recall": 1.0,
            "test_f1": 0.971,
            "test_roc_auc": 0.983,
            "test_pr_auc": 0.997,
        },
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

    metrics_path = os.path.join(args.out_dir, "train_metrics_pie_heading.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": model_path,
                "model_type": "ma_rong_pie_9_heading",
                "feature_names": FEATURE_NAMES_PIE_HEADING,
                "n_estimators": n_estimators,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "split_summary": split_info,
                "tune_result": tune_result,
                "baseline_reference": payload["baseline_reference"],
                "paper_reference": payload["paper_reference"],
            },
            f,
            indent=2,
        )

    cm = test_metrics["confusion_matrix"]
    print(f"Model saved: {model_path}")
    print(
        f"Test accuracy={test_metrics['accuracy']:.3f} "
        f"precision={test_metrics['precision']:.3f} "
        f"recall={test_metrics['recall']:.3f} "
        f"f1={test_metrics['f1']:.3f} "
        f"roc_auc={test_metrics.get('roc_auc', 'n/a'):.3f} "
        f"pr_auc={test_metrics.get('pr_auc', 'n/a'):.3f}"
    )
    print(f"Test confusion: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")


if __name__ == "__main__":
    main()
