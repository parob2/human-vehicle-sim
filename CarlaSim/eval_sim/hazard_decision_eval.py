#!/usr/bin/env python3
"""
Offline hazard decision policy accuracy (no CARLA).

Usage:
  python hazard_decision_eval.py
  python hazard_decision_eval.py --out results/hazard_decision_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_SIM_DIR = Path(__file__).resolve().parent
CARLA_DIR = EVAL_SIM_DIR.parent
if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: pip install pyyaml")

from hazard_decision_policy import compute_hazard_decision  # noqa: E402


def load_cases() -> tuple[dict, list]:
    with open(EVAL_SIM_DIR / "hazard_decision_cases.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("map_context") or {}, cfg.get("cases") or []


def eval_case(case: dict, map_ctx: dict) -> dict:
    ped = dict(case.get("ped") or {})
    ego = dict(case.get("ego") or {})
    expect = case.get("expect") or {}
    result = compute_hazard_decision(ped, ego, map_ctx)

    checks = {}
    for key, expected in expect.items():
        actual = result.get(key)
        if isinstance(expected, float):
            checks[key] = abs(float(actual or 0) - expected) < 1e-6
        else:
            checks[key] = actual == expected

    passed = all(checks.values())
    return {
        "id": case.get("id"),
        "description": case.get("description"),
        "passed": passed,
        "checks": checks,
        "result": {k: result.get(k) for k in expect},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hazard decision policy offline evaluation")
    parser.add_argument("--out", default=str(EVAL_SIM_DIR / "results" / "hazard_decision_eval.json"))
    args = parser.parse_args()

    map_ctx, cases = load_cases()
    results = [eval_case(c, map_ctx) for c in cases]
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    report = {
        "decision_accuracy": round(passed / total, 4) if total else None,
        "correct": passed,
        "total": total,
        "cases": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    if passed < total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
