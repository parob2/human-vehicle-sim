#!/usr/bin/env python3
"""
Analyze eval_sim JSONL run logs — thesis metrics per module and end-to-end.

Metrics:
  - Perception (YOLO): detection rate, precision, recall
  - RF intent: accuracy, precision, recall, F1, confusion matrix
  - Hazard decision policy: decision accuracy vs scenario ground truth
  - E2E safety: reaction time, safe stop rate, false brake rate
  - ETSI V2X: hazard-to-DENM generation, end-to-end, and network latency; CPM rate

Usage:
  python analyze.py --captures captures/
  python analyze.py captures/S2/run_*/run.jsonl
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:
    yaml = None

EVAL_SIM_DIR = Path(__file__).resolve().parent
COLLISION_M = 0.5
ANALYSIS_INTERVAL_S = 0.25


def load_jsonl(path: Path) -> Tuple[List[dict], Optional[dict]]:
    rows: List[dict] = []
    metadata: Optional[dict] = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record_type") == "run_metadata":
                metadata = rec
                continue
            rows.append(rec)
    return rows, metadata


def load_config() -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    with open(EVAL_SIM_DIR / "scenarios.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_rf_model_id(path: Path, metadata: Optional[dict]) -> str:
    if metadata and metadata.get("rf_model_id"):
        return str(metadata["rf_model_id"])
    parts = path.parts
    try:
        cap_idx = parts.index("captures")
        if cap_idx + 1 < len(parts):
            candidate = parts[cap_idx + 1]
            if not re.fullmatch(r"S[0-9]+", candidate):
                return candidate
    except ValueError:
        pass
    return "unknown"


def infer_scenario_id(path: Path, metadata: Optional[dict], rows: List[dict]) -> str:
    if metadata and metadata.get("scenario_id"):
        sid = str(metadata["scenario_id"])
        return sid.split("_run")[0] if "_run" in sid else sid
    for part in path.parts:
        if re.fullmatch(r"S[0-9]+", part):
            return part
    if rows and rows[0].get("scenario_id"):
        sid = str(rows[0]["scenario_id"])
        return sid.split("_run")[0] if "_run" in sid else sid
    return "unknown"


def analysis_steps(rows: List[dict]) -> List[dict]:
    out = []
    prev_key = None
    for r in rows:
        ped_p = r.get("ped_perception") or {}
        if ped_p.get("analysis_skipped"):
            continue
        key = (
            ped_p.get("detected"),
            ped_p.get("intent_pred"),
            ped_p.get("in_zone_vis"),
            ped_p.get("ped_phase"),
        )
        if key != prev_key:
            out.append(r)
            prev_key = key
    return out


def gt_ped_present(r: dict) -> bool:
    """True when CARLA reports a pedestrian (any non-empty ped_gt block)."""
    gt = r.get("ped_gt")
    return bool(gt)


def perception_metrics(rows: List[dict]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for r in analysis_steps(rows):
        ped_p = r.get("ped_perception") or {}
        detected = bool(ped_p.get("detected"))
        present = gt_ped_present(r)
        if present and detected:
            tp += 1
        elif present and not detected:
            fn += 1
        elif not present and detected:
            fp += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    detection_rate = recall
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "detection_rate": round(detection_rate, 4) if detection_rate is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "analysis_frames": total,
    }


def _confusion_from_pairs(pairs: List[tuple]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for actual, pred in pairs:
        if actual and pred:
            tp += 1
        elif actual and not pred:
            fn += 1
        elif not actual and pred:
            fp += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else None
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": round(acc, 4) if acc is not None else None,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def rf_intent_metrics(rows: List[dict], *, label_cross: bool) -> Dict[str, Any]:
    """
    RF intent: proba >= threshold while pedestrian is off zone.

    Uses rf_proba_raw + rf_threshold when available (works on old and new logs).
    Falls back to intent_pred for logs that have neither.
    """
    pairs: List[tuple] = []
    for r in analysis_steps(rows):
        ped_p = r.get("ped_perception") or {}
        if not gt_ped_present(r):
            continue
        if bool(ped_p.get("in_zone_vis")):
            continue
        raw_p = ped_p.get("rf_proba_raw")
        threshold = float(ped_p.get("rf_threshold") or 0.5)
        if raw_p is not None:
            predicted = float(raw_p) >= threshold
        else:
            predicted = bool(ped_p.get("intent_pred"))
        pairs.append((label_cross, predicted))

    result = _confusion_from_pairs(pairs)
    result["label_crossing"] = label_cross
    return result


def _hazard_decision_row(r: dict) -> dict:
    """Read hazard decision from a JSONL row (supports legacy ``priority`` key)."""
    return r.get("hazard_decision") or r.get("priority") or {}


def hazard_decision_metrics(rows: List[dict], *, expects_brake: bool) -> Dict[str, Any]:
    """Check whether brake/denm decisions match scenario ground truth."""
    correct = 0
    total = 0
    for r in analysis_steps(rows):
        brake = float(r.get("ego_brake_level") or 0) > 0
        denm_sent = bool((r.get("denm") or {}).get("sent"))
        hd = _hazard_decision_row(r)
        hazard_active = bool(hd.get("hazard_active", hd.get("send_denm")))
        gt = r.get("ped_gt") or {}
        if not gt_ped_present(r) and not expects_brake:
            total += 1
            if not brake:
                correct += 1
            continue
        if not gt_ped_present(r):
            continue
        total += 1
        if expects_brake:
            in_zone = bool(gt.get("in_zone") or (r.get("ped_perception") or {}).get("in_zone_vis"))
            if in_zone:
                if brake:
                    correct += 1
            else:
                if not brake or hazard_active or denm_sent:
                    correct += 1
        else:
            if not brake and not hazard_active:
                correct += 1
    acc = correct / total if total else None
    return {"correct": correct, "total": total, "decision_accuracy": round(acc, 4) if acc is not None else None}


def reaction_time_s(rows: List[dict]) -> Optional[float]:
    t_intent = None
    t_brake = None
    for r in rows:
        ped_p = r.get("ped_perception") or {}
        t = float(r["sim_time_s"])
        if t_intent is None and (ped_p.get("intent_pred") or ped_p.get("intent_confirmed")):
            t_intent = t
        if t_brake is None and float(r.get("ego_brake_level") or 0) > 0:
            t_brake = t
    if t_intent is None or t_brake is None:
        return None
    return round(max(0.0, t_brake - t_intent), 4)


def safe_stop(rows: List[dict], *, min_dist_m: float = 0.5) -> bool:
    min_d = None
    for r in rows:
        d = (r.get("safety") or {}).get("min_centroid_distance_m")
        if d is not None:
            min_d = d if min_d is None else min(min_d, d)
    if min_d is None:
        return False
    collision = min_d < COLLISION_M
    return not collision and min_d >= min_dist_m


def had_false_brake(rows: List[dict]) -> bool:
    for r in rows:
        if float(r.get("ego_brake_level") or 0) > 0:
            gt = r.get("ped_gt") or {}
            if not gt.get("in_zone") and not (r.get("ped_perception") or {}).get("in_zone_vis"):
                hd = _hazard_decision_row(r)
                if not hd.get("path_conflict"):
                    return True
    return any(float(r.get("ego_brake_level") or 0) > 0 for r in rows)


def _hazard_detected(r: dict) -> bool:
    """
    Hazard activation for V2X latency (runtime perception only).

    Matches RSU ``hazard_event_active``: intent_confirmed or in_zone_vis.
    ``ped_gt.in_zone`` is ground truth for scenario labelling, not used here.
    """
    ped_p = r.get("ped_perception") or {}
    if ped_p.get("analysis_skipped"):
        return False
    hd = _hazard_decision_row(r)
    if "hazard_active" in hd:
        return bool(hd.get("hazard_active"))
    return bool(ped_p.get("intent_confirmed") or ped_p.get("in_zone_vis"))


def _denm_sent_tick(r: dict) -> bool:
    v2x = r.get("v2x") or {}
    denm = r.get("denm") or {}
    return bool(v2x.get("denm_sent") or denm.get("sent"))


def _denm_received_tick(r: dict) -> bool:
    return bool((r.get("v2x") or {}).get("ego_denm_active_rx"))


def _tick_time_s(r: dict, block: str) -> float:
    nested = r.get(block) or {}
    if nested.get("sim_time_s") is not None:
        return float(nested["sim_time_s"])
    return float(r["sim_time_s"])


def v2x_latency_metrics(rows: List[dict]) -> Dict[str, Optional[float]]:
    """
    Per-run V2X latencies (one hazard event chain):

      L_gen = t_first_DENM_sent - t_hazard
      L_e2e = t_DENM_received - t_hazard
      L_net = t_DENM_received - t_first_DENM_sent
    """
    t_hazard: Optional[float] = None
    t_sent: Optional[float] = None
    t_rx: Optional[float] = None

    for r in rows:
        if t_hazard is None and _hazard_detected(r):
            t_hazard = float(r["sim_time_s"])
        if t_sent is None and _denm_sent_tick(r):
            t_sent = _tick_time_s(r, "denm")
        if t_rx is None and _denm_received_tick(r):
            t_rx = float(r["sim_time_s"])

    def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return round(max(0.0, b - a), 4)

    return {
        "hazard_detection_s": round(t_hazard, 4) if t_hazard is not None else None,
        "first_denm_sent_s": round(t_sent, 4) if t_sent is not None else None,
        "first_denm_received_s": round(t_rx, 4) if t_rx is not None else None,
        "generation_s": _delta(t_hazard, t_sent),
        "end_to_end_s": _delta(t_hazard, t_rx),
        "network_s": _delta(t_sent, t_rx),
    }


def _latency_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": None, "max": None, "median": None, "n": 0}
    return {
        "mean": round(statistics.mean(values), 4),
        "max": round(max(values), 4),
        "median": round(statistics.median(values), 4),
        "n": len(values),
    }


def etsi_metrics(rows: List[dict], *, requirements: dict) -> Dict[str, Any]:
    denm_count = 0
    cpm_times: List[float] = []

    for r in rows:
        if _denm_sent_tick(r):
            denm_count += 1

        cpm = r.get("cpm") or {}
        if cpm.get("sent") and cpm.get("sim_time_s") is not None:
            cpm_times.append(float(cpm["sim_time_s"]))

    duration = float(rows[-1]["sim_time_s"]) - float(rows[0]["sim_time_s"]) if len(rows) > 1 else 0.0
    cpm_count = len(set(round(t, 2) for t in cpm_times))
    cpm_rate_hz = cpm_count / duration if duration > 0 else None

    lat = v2x_latency_metrics(rows)
    gen_s = lat.get("generation_s")

    denm_max_s = float(requirements.get("denm_generation_max_s", 1.0))
    cpm_min_hz = float(requirements.get("cpm_min_hz", 1.0))

    return {
        "denm_count": denm_count,
        "cpm_count": cpm_count,
        "cpm_rate_hz": round(cpm_rate_hz, 2) if cpm_rate_hz is not None else None,
        "v2x_latency": lat,
        "generation_latency_pass": gen_s is not None and gen_s < denm_max_s,
        "cpm_rate_pass": cpm_rate_hz is not None and cpm_rate_hz >= cpm_min_hz,
    }


def analyze_run(path: Path, cfg: dict) -> Dict[str, Any]:
    rows, metadata = load_jsonl(path)
    if not rows:
        return {"file": str(path), "error": "empty"}

    sid = infer_scenario_id(path, metadata, rows)
    model_id = infer_rf_model_id(path, metadata)
    sc = (cfg.get("scenarios") or {}).get(sid) or {}
    gt = sc.get("ground_truth") or {}
    defaults = cfg.get("defaults") or {}
    etsi_req = cfg.get("etsi_requirements") or {}
    category = sc.get("category", "unknown")
    label_cross = bool(gt.get("crossing_intent"))
    expects_brake = bool(gt.get("expects_brake"))

    perception = perception_metrics(rows)
    rf = rf_intent_metrics(rows, label_cross=label_cross)
    hazard_decision = hazard_decision_metrics(rows, expects_brake=expects_brake)
    rt = reaction_time_s(rows)
    etsi = etsi_metrics(rows, requirements=etsi_req)

    e2e = {
        "reaction_time_s": rt,
        "safe_stop": safe_stop(rows, min_dist_m=float(defaults.get("safe_stop_min_dist_m", 0.5))),
        "false_brake": had_false_brake(rows) if category == "non_crossing" else False,
        "category": category,
    }

    return {
        "file": str(path),
        "scenario_id": sid,
        "rf_model_id": model_id,
        "rf_model_path": (metadata or {}).get("pip_rf_model") or (metadata or {}).get("rf_intent_model"),
        "scenario_name": sc.get("name"),
        "category": category,
        "tick_count": len(rows),
        "perception": perception,
        "rf_intent": rf,
        "hazard_decision": hazard_decision,
        "e2e": e2e,
        "etsi": etsi,
        "random_seed": metadata.get("random_seed") if metadata else None,
    }


def aggregate(per_run: List[dict], cfg: dict) -> Dict[str, Any]:
    defaults = cfg.get("defaults") or {}

    def _agg_confusion(key: str, prefix: str = "") -> dict:
        tp_k = f"{prefix}TP"
        fp_k = f"{prefix}FP"
        tn_k = f"{prefix}TN"
        fn_k = f"{prefix}FN"
        agg = {tp_k: 0, fp_k: 0, tn_k: 0, fn_k: 0}
        for r in per_run:
            block = r.get(key) or {}
            for k in agg:
                agg[k] += int(block.get(k, 0))
        tp, fp, tn, fn = agg[tp_k], agg[fp_k], agg[tn_k], agg[fn_k]
        total = tp + fp + tn + fn
        return {
            **agg,
            f"{prefix}precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
            f"{prefix}recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
            f"{prefix}accuracy": round((tp + tn) / total, 4) if total else None,
        }

    crossing = [r for r in per_run if r.get("category") == "crossing"]
    non_crossing = [r for r in per_run if r.get("category") == "non_crossing"]

    reaction_times = [r["e2e"]["reaction_time_s"] for r in crossing if r.get("e2e", {}).get("reaction_time_s") is not None]
    safe_stops = sum(1 for r in crossing if r.get("e2e", {}).get("safe_stop"))
    false_brakes = sum(1 for r in non_crossing if r.get("e2e", {}).get("false_brake"))

    hdp_correct = sum(
        (r.get("hazard_decision") or r.get("priority") or {}).get("correct", 0)
        for r in per_run
    )
    hdp_total = sum(
        (r.get("hazard_decision") or r.get("priority") or {}).get("total", 0)
        for r in per_run
    )

    def _collect_latency(key: str) -> List[float]:
        out: List[float] = []
        for r in per_run:
            v = ((r.get("etsi") or {}).get("v2x_latency") or {}).get(key)
            if v is not None:
                out.append(float(v))
        return out

    rf_agg = _agg_confusion("rf_intent")
    prec, rec = rf_agg.get("precision"), rf_agg.get("recall")
    f1 = round(2 * prec * rec / (prec + rec), 4) if prec and rec and (prec + rec) else None

    return {
        "runs_total": len(per_run),
        "perception_aggregate": _agg_confusion("perception"),
        "rf_intent_aggregate": {**rf_agg, "f1": f1},
        "hazard_decision_accuracy": round(hdp_correct / hdp_total, 4) if hdp_total else None,
        "hazard_decisions": {"correct": hdp_correct, "total": hdp_total},
        "reaction_time_s": {
            "mean": round(statistics.mean(reaction_times), 4) if reaction_times else None,
            "max": round(max(reaction_times), 4) if reaction_times else None,
            "n": len(reaction_times),
        },
        "safe_stop_rate": round(safe_stops / len(crossing), 4) if crossing else None,
        "safe_stops": {"success": safe_stops, "total": len(crossing)},
        "false_brake_rate": round(false_brakes / len(non_crossing), 4) if non_crossing else None,
        "false_brakes": {"events": false_brakes, "total": len(non_crossing)},
        "v2x_latency_aggregate": {
            "generation_latency_s": _latency_stats(_collect_latency("generation_s")),
            "end_to_end_latency_s": _latency_stats(_collect_latency("end_to_end_s")),
            "network_latency_s": _latency_stats(_collect_latency("network_s")),
        },
        "collision_threshold_m": defaults.get("collision_threshold_m", COLLISION_M),
    }


def aggregate_by_model(per_run: List[dict], cfg: dict) -> Dict[str, dict]:
    by_model: Dict[str, List[dict]] = {}
    for r in per_run:
        mid = r.get("rf_model_id") or "unknown"
        by_model.setdefault(mid, []).append(r)
    return {mid: aggregate(runs, cfg) for mid, runs in sorted(by_model.items())}


def write_outputs(
    out_dir: Path,
    per_run: List[dict],
    aggregate_data: dict,
    per_model: Optional[Dict[str, dict]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_run.json", "w") as f:
        json.dump(per_run, f, indent=2)
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(aggregate_data, f, indent=2)
    if per_model:
        with open(out_dir / "aggregate_by_model.json", "w") as f:
            json.dump(per_model, f, indent=2)

    with open(out_dir / "per_run.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "rf_model_id", "scenario_id", "category", "perception_recall", "rf_accuracy",
            "hazard_decision_accuracy", "reaction_time_s", "safe_stop", "false_brake",
            "gen_latency_s", "e2e_latency_s", "net_latency_s", "cpm_hz",
        ])
        for r in per_run:
            lat = (r.get("etsi") or {}).get("v2x_latency") or {}
            w.writerow([
                r.get("file"), r.get("rf_model_id"), r.get("scenario_id"), r.get("category"),
                (r.get("perception") or {}).get("recall"),
                (r.get("rf_intent") or {}).get("accuracy"),
                ((r.get("hazard_decision") or r.get("priority") or {}).get("decision_accuracy")),
                (r.get("e2e") or {}).get("reaction_time_s"),
                (r.get("e2e") or {}).get("safe_stop"),
                (r.get("e2e") or {}).get("false_brake"),
                lat.get("generation_s"),
                lat.get("end_to_end_s"),
                lat.get("network_s"),
                (r.get("etsi") or {}).get("cpm_rate_hz"),
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze eval_sim JSONL logs")
    parser.add_argument("jsonl", nargs="*", help="JSONL file(s) or glob")
    parser.add_argument("--captures", default=None, help="Scan eval_sim/captures for run.jsonl")
    parser.add_argument("--out-dir", default=str(EVAL_SIM_DIR / "results"))
    args = parser.parse_args()

    paths: List[Path] = []
    if args.captures:
        paths.extend(Path(p) for p in glob.glob(str(Path(args.captures) / "**" / "run.jsonl"), recursive=True))
    for p in args.jsonl:
        paths.extend(Path(x) for x in glob.glob(p))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No JSONL files found. Run simulations first or pass --captures.")

    cfg = load_config()
    per_run = [analyze_run(p, cfg) for p in paths]
    per_run = [r for r in per_run if "error" not in r]
    agg = aggregate(per_run, cfg)
    per_model = aggregate_by_model(per_run, cfg)

    out_dir = Path(args.out_dir)
    write_outputs(out_dir, per_run, agg, per_model)
    print(json.dumps({"out_dir": str(out_dir), "aggregate": agg, "by_model": per_model}, indent=2))


if __name__ == "__main__":
    main()
