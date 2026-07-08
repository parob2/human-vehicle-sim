"""Leakage-safe group splits and imbalanced-class metrics for offline RF eval."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

TRAIN_TEST_SPLIT = 0.30
RANDOM_STATE = 42


def make_person_group_id(meta: Dict[str, Any]) -> str:
    """One group per tracked pedestrian within a source JSON file."""
    source = str(meta.get("source", meta.get("source_name", "unknown")))
    person_id = str(meta.get("person_id", "unknown"))
    return f"{source}::{person_id}"


def make_person_group_ids(meta: List[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([make_person_group_id(m) for m in meta])


def _group_positive_frame_count(y: np.ndarray, groups: np.ndarray, group_id: str) -> int:
    return int(np.sum(y[groups == group_id] == 1))


def _groups_with_any_positive(y: np.ndarray, groups: np.ndarray) -> List[str]:
    return [g for g in np.unique(groups) if _group_positive_frame_count(y, groups, g) > 0]


def split_train_test_by_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float = TRAIN_TEST_SPLIT,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/test arrays and indices; all frames from one group stay together."""
    if stratify and len(np.unique(y)) >= 2:
        return _split_train_test_by_group_stratified(
            X, y, groups, test_size=test_size, random_state=random_state
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        train_idx,
        test_idx,
    )


def _split_train_test_by_group_stratified(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hold out whole pedestrian tracks while balancing crossing-intent groups.

    A group is treated as positive if it contains *any* positive frame, not only
    when most of its frames are positive. This keeps enough positive frames in
    the test split for reliable metrics on imbalanced RSU data.
    """
    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(groups)

    pos_groups = _groups_with_any_positive(y, groups)
    neg_groups = [g for g in unique_groups if g not in set(pos_groups)]

    pos_groups = sorted(
        pos_groups,
        key=lambda g: _group_positive_frame_count(y, groups, g),
        reverse=True,
    )
    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    n_test_pos = max(1, int(round(len(pos_groups) * test_size))) if pos_groups else 0
    n_test_neg = max(1, int(round(len(neg_groups) * test_size))) if neg_groups else 0

    if len(pos_groups) <= 1:
        n_test_pos = 1 if pos_groups else 0
    elif n_test_pos >= len(pos_groups):
        n_test_pos = max(1, len(pos_groups) - 1)

    if len(neg_groups) <= 1:
        n_test_neg = 1 if neg_groups else 0
    elif n_test_neg >= len(neg_groups):
        n_test_neg = max(1, len(neg_groups) - 1)

    test_group_set = set(pos_groups[:n_test_pos] + neg_groups[:n_test_neg])
    test_idx = np.asarray([i for i, g in enumerate(groups) if g in test_group_set])
    train_idx = np.asarray([i for i, g in enumerate(groups) if g not in test_group_set])

    if len(train_idx) == 0 or len(test_idx) == 0 or np.sum(y[test_idx]) == 0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        train_idx,
        test_idx,
    )


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Any]:
    """Primary ranking metric: PR-AUC (average precision). Also reports recall at threshold."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp = cm[0]
    fn, tp = cm[1]
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "num_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
    }
    if len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        # Backward-compatible alias used in older reports.
        out["auc"] = out["roc_auc"]
    return out


def summarize_split(
    groups: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Any]:
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    train_pos_groups = _groups_with_any_positive(y[train_idx], groups[train_idx])
    test_pos_groups = _groups_with_any_positive(y[test_idx], groups[test_idx])
    return {
        "num_train_samples": int(len(train_idx)),
        "num_test_samples": int(len(test_idx)),
        "num_train_groups": int(len(train_groups)),
        "num_test_groups": int(len(test_groups)),
        "num_train_positives": int(np.sum(y[train_idx])),
        "num_test_positives": int(np.sum(y[test_idx])),
        "num_train_pos_groups": int(len(train_pos_groups)),
        "num_test_pos_groups": int(len(test_pos_groups)),
        "train_positive_rate": float(np.mean(y[train_idx])),
        "test_positive_rate": float(np.mean(y[test_idx])),
        "group_leakage": bool(overlap),
        "leaked_groups": sorted(overlap) if overlap else [],
    }


def tune_threshold_for_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    min_precision: float = 0.0,
) -> Dict[str, Any]:
    """Pick threshold on train split maximizing recall subject to optional precision floor."""
    best: Dict[str, Any] = {
        "threshold": 0.5,
        "recall": 0.0,
        "precision": 0.0,
        "f1": 0.0,
        "pr_auc": 0.0,
    }
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (y_proba >= threshold).astype(int)
        metrics = compute_binary_metrics(y_true, pred, y_proba)
        if metrics["precision"] + 1e-9 < min_precision:
            continue
        if metrics["recall"] > best["recall"] or (
            metrics["recall"] == best["recall"] and metrics["f1"] > best["f1"]
        ):
            best = {
                "threshold": float(threshold),
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "pr_auc": metrics.get("pr_auc", 0.0),
            }
    return best
