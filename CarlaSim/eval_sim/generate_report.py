#!/usr/bin/env python3
"""
Generate thesis evaluation summary table and markdown report from eval_sim results.

Usage:
  python generate_report.py
  python generate_report.py --results results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

EVAL_SIM_DIR = Path(__file__).resolve().parent


def pct(v, digits=1) -> str:
    if v is None:
        return "—"
    return f"{100 * v:.{digits}f}%"


def sec(v, digits=2) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f} s"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _fmt_latency_stats(stats: dict) -> str:
    if not stats or stats.get("mean") is None:
        return "—"
    return (
        f"mean {sec(stats['mean'])}, max {sec(stats['max'])}, "
        f"median {sec(stats['median'])} (n={stats.get('n', 0)})"
    )


def build_summary_table(aggregate: dict, hazard_decision_eval: dict, cfg: dict) -> list[dict]:
    perc = aggregate.get("perception_aggregate") or {}
    rf = aggregate.get("rf_intent_aggregate") or {}
    v2x = aggregate.get("v2x_latency_aggregate") or {}

    hdp_acc = aggregate.get("hazard_decision_accuracy")
    if hdp_acc is None:
        hdp_acc = aggregate.get("priority_decision_accuracy")
    if hazard_decision_eval.get("decision_accuracy") is not None:
        hdp_acc = hazard_decision_eval["decision_accuracy"]

    rt = aggregate.get("reaction_time_s") or {}
    gen = v2x.get("generation_latency_s") or {}
    e2e = v2x.get("end_to_end_latency_s") or {}
    net = v2x.get("network_latency_s") or {}

    rows = [
        {"metric": "Pedestrian Recall", "result": pct(perc.get("recall"))},
        {"metric": "RF Intent Accuracy", "result": pct(rf.get("accuracy"))},
        {"metric": "Hazard Decision Accuracy", "result": pct(hdp_acc)},
        {"metric": "Mean Reaction Time", "result": sec(rt.get("mean"))},
        {"metric": "Max Reaction Time", "result": sec(rt.get("max"))},
        {"metric": "Safe Stop Rate", "result": pct(aggregate.get("safe_stop_rate"))},
        {"metric": "False Brake Rate", "result": pct(aggregate.get("false_brake_rate"))},
        {
            "metric": "Hazard-to-DENM generation latency",
            "result": _fmt_latency_stats(gen),
        },
        {
            "metric": "End-to-end warning latency",
            "result": _fmt_latency_stats(e2e),
        },
        {
            "metric": "DENM network latency",
            "result": _fmt_latency_stats(net),
        },
    ]
    return rows


def build_etsi_table(aggregate: dict, cfg: dict) -> list[dict]:
    req = cfg.get("etsi_requirements") or {}
    v2x = aggregate.get("v2x_latency_aggregate") or {}
    gen = v2x.get("generation_latency_s") or {}
    lat = gen.get("mean")
    return [
        {
            "kpi": "Hazard-to-DENM generation",
            "requirement": f"< {req.get('denm_generation_max_s', 1.0)} s (mean)",
            "result": _fmt_latency_stats(gen),
            "pass": lat is not None and lat < float(req.get("denm_generation_max_s", 1.0)),
        },
        {
            "kpi": "End-to-end warning latency",
            "requirement": "report mean / max / median",
            "result": _fmt_latency_stats(v2x.get("end_to_end_latency_s") or {}),
            "pass": None,
        },
        {
            "kpi": "DENM network latency",
            "requirement": "report mean / max / median",
            "result": _fmt_latency_stats(v2x.get("network_latency_s") or {}),
            "pass": None,
        },
        {
            "kpi": "CPM update rate",
            "requirement": f"≥ {req.get('cpm_min_hz', 1.0)} Hz",
            "result": "see per-run",
            "pass": None,
        },
    ]


def markdown_report(cfg: dict, aggregate: dict, hazard_decision_eval: dict, summary: list, etsi_table: list) -> str:
    setup = cfg.get("experimental_setup") or {}
    scenarios = cfg.get("scenarios") or {}
    rf = aggregate.get("rf_intent_aggregate") or {}
    perc = aggregate.get("perception_aggregate") or {}

    lines = [
        "# Evaluation Results (eval_sim)",
        "",
        "## 5.1 Experimental Setup",
        "",
        f"- CARLA version: {setup.get('carla_version', '—')}",
        f"- Weather: {setup.get('weather', '—')}",
        f"- Fixed seed: {setup.get('fixed_seed', '—')}",
        f"- Camera: {setup.get('camera', '—')}",
        f"- Scenarios: S1–S6 ({len(scenarios)} defined in `scenarios.yaml`)",
        "",
        "## 5.2 Perception Validation (YOLO Pose)",
        "",
        f"- Detection rate (recall): {pct(perc.get('recall'))}",
        f"- Precision: {pct(perc.get('precision'))}",
        f"- Recall: {pct(perc.get('recall'))}",
        "",
        "## 5.3 Intent Prediction Validation (Random Forest)",
        "",
        f"- Accuracy: {pct(rf.get('accuracy'))}",
        f"- Precision: {pct(rf.get('precision'))}",
        f"- Recall: {pct(rf.get('recall'))}",
        f"- F1-score: {pct(rf.get('f1'))}",
        "",
        "| | Pred Cross | Pred No Cross |",
        "|---|---:|---:|",
        f"| Actual Cross | {rf.get('TP', '—')} | {rf.get('FN', '—')} |",
        f"| Actual No Cross | {rf.get('FP', '—')} | {rf.get('TN', '—')} |",
        "",
        "## 5.4 End-to-End System Evaluation",
        "",
    ]

    rt = aggregate.get("reaction_time_s") or {}
    safe = aggregate.get("safe_stops") or {}
    fb = aggregate.get("false_brakes") or {}
    lines.extend([
        f"- Mean reaction time (intent → brake): {sec(rt.get('mean'))}",
        f"- Maximum reaction time: {sec(rt.get('max'))}",
        f"- Safe stops: {safe.get('success', '—')} / {safe.get('total', '—')} "
        f"({pct(aggregate.get('safe_stop_rate'))})",
        f"- False braking (non-crossing runs): {fb.get('events', '—')} / {fb.get('total', '—')} "
        f"({pct(aggregate.get('false_brake_rate'))})",
        "",
        "## 5.5 Hazard Decision Policy (offline + runtime)",
        "",
    ])
    hdp = hazard_decision_eval
    lines.append(
        f"- Offline decision accuracy: {hdp.get('correct', '—')} / {hdp.get('total', '—')} "
        f"({pct(hdp.get('decision_accuracy'))})"
    )
    lines.extend([
        "",
        "## 5.6 V2X Warning Latency",
        "",
        "| Metric | Definition | Mean | Max | Median |",
        "|--------|------------|-----:|----:|-------:|",
    ])
    v2x = aggregate.get("v2x_latency_aggregate") or {}
    for label, key in (
        ("Hazard-to-DENM generation", "generation_latency_s"),
        ("End-to-end warning", "end_to_end_latency_s"),
        ("DENM network", "network_latency_s"),
    ):
        stats = v2x.get(key) or {}
        lines.append(
            f"| {label} | see thesis definitions | "
            f"{sec(stats.get('mean'))} | {sec(stats.get('max'))} | {sec(stats.get('median'))} |"
        )
    lines.extend([
        "",
        "## 5.7 ETSI Compliance",
        "",
        "| KPI | Requirement | Result |",
        "|-----|-------------|--------|",
    ])
    for row in etsi_table:
        status = "✓" if row.get("pass") else ("—" if row.get("pass") is None else "✗")
        lines.append(f"| {row['kpi']} | {row['requirement']} | {row['result']} {status} |")

    lines.extend([
        "",
        "## Summary Table",
        "",
        "| Metric | Result |",
        "|--------|--------|",
    ])
    for row in summary:
        lines.append(f"| {row['metric']} | {row['result']} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval_sim thesis report")
    parser.add_argument("--results", default=str(EVAL_SIM_DIR / "results"))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if yaml is None:
        raise SystemExit("PyYAML required")

    with open(EVAL_SIM_DIR / "scenarios.yaml") as f:
        cfg = yaml.safe_load(f)

    aggregate = load_json(results_dir / "aggregate.json")
    hdp_eval_path = results_dir / "hazard_decision_eval.json"
    legacy_eval_path = results_dir / "priority_eval.json"
    hazard_decision_eval = load_json(
        hdp_eval_path if hdp_eval_path.exists() else legacy_eval_path
    )

    summary = build_summary_table(aggregate, hazard_decision_eval, cfg)
    etsi_table = build_etsi_table(aggregate, cfg)
    md = markdown_report(cfg, aggregate, hazard_decision_eval, summary, etsi_table)

    with open(out_dir / "summary_table.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "evaluation_report.md", "w") as f:
        f.write(md)

    print(f"Wrote {out_dir / 'evaluation_report.md'}")
    print(f"Wrote {out_dir / 'summary_table.json'}")


if __name__ == "__main__":
    main()
