#!/usr/bin/env python3
"""Evaluate Ma-Rong 9-feature PIE RF (7 paper features + trajectory heading)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Must set Agg before pyplot import (avoids tkinter crash on headless / threaded exit).
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import learning_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pie.config_pie import (
    DEFAULT_FEATURES_JSONL_HEADING,
    DEFAULT_MODEL_PATH_PIE_HEADING,
    DEFAULT_PIE_HEADING_EVAL_OUT,
    TRAIN_TEST_SPLIT,
)
from pie.pie_dataset_loader import load_pie_heading_features_from_jsonl
from split_metrics import compute_binary_metrics, split_train_test_by_group, summarize_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Ma-Rong PIE heading RF model")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH_PIE_HEADING))
    parser.add_argument("--features-jsonl", action="append", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_PIE_HEADING_EVAL_OUT))
    parser.add_argument("--test-size", type=float, default=TRAIN_TEST_SPLIT)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    payload = joblib.load(args.model)
    threshold = float(payload.get("threshold", 0.5))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_paths = args.features_jsonl if args.features_jsonl else [str(DEFAULT_FEATURES_JSONL_HEADING)]
    X, y, groups, meta = load_pie_heading_features_from_jsonl(jsonl_paths)

    X_train, X_test, y_train, y_test, train_idx, test_idx = split_train_test_by_group(
        X, y, groups, test_size=args.test_size, random_state=args.random_state, stratify=True
    )
    split_info = summarize_split(groups, y, train_idx, test_idx)

    model = clone(payload["model"])
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= threshold).astype(int)
    train_metrics = compute_binary_metrics(y_train, train_pred, train_proba)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = compute_binary_metrics(y_test, pred, proba)

    if len(np.unique(y_test)) > 1:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
        ax.set_title("ROC — Ma-Rong PIE 9-feature RF (heading)")
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=160)
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax)
        ax.set_title("PR curve — Ma-Rong PIE 9-feature RF (heading)")
        plt.tight_layout()
        plt.savefig(out_dir / "pr_curve.png", dpi=160)
        plt.close()

    train_sizes = np.linspace(0.1, 1.0, 8)
    try:
        sizes_abs, train_scores, val_scores = learning_curve(
            clone(model),
            X_train,
            y_train,
            cv=3,
            train_sizes=train_sizes,
            scoring="accuracy",
            n_jobs=1,
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sizes_abs, np.mean(train_scores, axis=1), "o-", label="train accuracy")
        ax.plot(sizes_abs, np.mean(val_scores, axis=1), "o-", label="cv accuracy")
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning curve — PIE Ma-Rong 9-feature RF")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "learning_curve.png", dpi=160)
        plt.close()
    except Exception as ex:
        print(f"[warn] learning curve skipped: {ex}")

    cm = metrics["confusion_matrix"]
    report = {
        "evaluation_protocol": "70/30 stratified pedestrian holdout; retrain on train split",
        "model_type": payload.get("model_type"),
        "feature_names": payload.get("feature_names"),
        "split_summary": split_info,
        "train_metrics": train_metrics,
        "test_metrics": metrics,
        "baseline_reference": payload.get("baseline_reference", {}),
        "paper_reference": payload.get("paper_reference", {}),
        "test_confusion": {
            "TN": cm[0][0],
            "FP": cm[0][1],
            "FN": cm[1][0],
            "TP": cm[1][1],
        },
    }
    with open(out_dir / "eval_metrics_pie_heading.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Wrote plots and metrics to {out_dir}")


if __name__ == "__main__":
    main()
