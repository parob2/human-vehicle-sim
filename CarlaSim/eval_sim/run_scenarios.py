#!/usr/bin/env python3
"""
Run thesis scenario matrix from eval_sim/scenarios.yaml (requires CARLA server).

Each scenario is run once per RF model defined in rf_models.yaml.

Usage:
  python run_scenarios.py --list
  python run_scenarios.py --list-models
  python run_scenarios.py --scenario S2 --model pie_9_heading --runs 5
  python run_scenarios.py --all --all-models --runs 5 --analyze
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

EVAL_SIM_DIR = Path(__file__).resolve().parent
CARLA_DIR = EVAL_SIM_DIR.parent
SIM_ROOT = CARLA_DIR.parent
MAIN = CARLA_DIR / "main.py"
CAPTURE_DIR = EVAL_SIM_DIR / "captures"

# Active test model (medium — faster than x, less VRAM).
_YOLO_POSE_MODEL = "yolov8m-pose.pt"
_YOLO_SEARCH_PATHS = (
    CARLA_DIR / _YOLO_POSE_MODEL,
    CARLA_DIR.parent / _YOLO_POSE_MODEL,
    SIM_ROOT / _YOLO_POSE_MODEL,
    SIM_ROOT / "PedestrianIntentPrediction" / _YOLO_POSE_MODEL,
)


def resolve_yolo_pose_model() -> str:
    """Absolute path to cached pose weights (no network download if cached locally)."""
    for p in _YOLO_SEARCH_PATHS:
        if p.is_file():
            return str(p.resolve())
    return str(_YOLO_SEARCH_PATHS[0])


def load_yaml(name: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML required: pip install pyyaml")
    with open(EVAL_SIM_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scenarios() -> dict:
    return load_yaml("scenarios.yaml")


def load_rf_models() -> dict:
    return load_yaml("rf_models.yaml")


def resolve_model_path(raw: str) -> str:
    p = Path(raw)
    if p.is_file():
        return str(p.resolve())
    for base in (SIM_ROOT, CARLA_DIR.parent):
        candidate = base / raw
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"RF model not found: {raw}")


def resolve_rf_model(model_id: str, models_cfg: dict) -> dict:
    models = models_cfg.get("models") or {}
    if model_id not in models:
        raise KeyError(f"Unknown RF model {model_id!r}")
    spec = dict(models[model_id])
    raw_path = spec.get("model_path", "")
    spec["model_path_resolved"] = resolve_model_path(raw_path) if raw_path else ""
    env = {}
    for k, v in (spec.get("env") or {}).items():
        val = str(v).replace("{model_path}", spec["model_path_resolved"])
        env[str(k)] = val
    spec["env_resolved"] = env
    return spec


# Detection gates aligned with main.py / dataset generation (override via shell env).
_YOLO_EVAL_DEFAULTS = {
    "YOLO_CONF": "0.15",
    "YOLO_MIN_BBOX_H_PX": "14",
    "YOLO_MIN_ASPECT_RATIO": "0.82",
    "YOLO_CONF_NEW_TRACK_MIN": "0.08",
    "YOLO_IMGSZ": "1280",
}


def build_env(
    scenario_id: str,
    scenario: dict,
    model_id: str,
    model_spec: dict,
    run_idx: int,
    base_seed: int,
    defaults: dict,
) -> dict:
    env = os.environ.copy()
    env["SIM_RUN_LOGGER"] = "1"
    env["SIM_VALIDATION_MODE"] = "1"
    env["SIM_SPAWN_PRESETS"] = "eval_sim.spawn_presets"
    env["SIM_CAPTURE_DIR"] = str(CAPTURE_DIR / model_id)
    env["SIM_RF_MODEL_ID"] = model_id
    env["SIM_MAX_DURATION_S"] = str(defaults.get("sim_max_duration_s", 120))
    env["SIM_RANDOM_SEED"] = str(base_seed + run_idx)
    env["SIM_SCENARIO_ID"] = f"{scenario_id}_run{run_idx}"
    env["YOLO_POSE_MODEL"] = resolve_yolo_pose_model()
    for key, val in _YOLO_EVAL_DEFAULTS.items():
        env.setdefault(key, val)
    for k, v in (scenario.get("env") or {}).items():
        env[str(k)] = str(v)
    for k, v in model_spec.get("env_resolved", {}).items():
        env[str(k)] = str(v)
    walker = scenario.get("walker") or {}
    if walker.get("spawn_preset"):
        env["WALKER_SPAWN_PRESET"] = str(walker["spawn_preset"])
    if walker.get("spawn_delay_s") is not None:
        env["WALKER_START_DELAY_S"] = str(walker["spawn_delay_s"])
    if walker.get("speed_ms") is not None:
        env["WALKER_SPEED_MS"] = str(walker["speed_ms"])
    if walker.get("enabled") is False or walker.get("spawn_preset") == "none":
        env["DISABLE_WALKER_SPAWN"] = "1"
        env["WALKER_SPAWN_PRESET"] = "none"
    return env


def find_run_jsonl_files() -> list[Path]:
    return sorted(CAPTURE_DIR.rglob("run.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval_sim thesis scenario × RF model matrix")
    parser.add_argument("--scenario", default=None, help="S1..S6")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default=None, help="RF model id from rf_models.yaml")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--analyze", action="store_true", help="Run analyze.py after simulations")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--base-seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_scenarios()
    models_cfg = load_rf_models()
    defaults = cfg.get("defaults") or {}
    model_defaults = models_cfg.get("defaults") or {}
    scenarios = cfg.get("scenarios") or {}
    all_models = models_cfg.get("models") or {}
    runs = args.runs if args.runs is not None else int(
        defaults.get("runs_per_scenario", model_defaults.get("runs_per_scenario", 5))
    )
    base_seed = args.base_seed if args.base_seed is not None else int(
        defaults.get("base_seed", model_defaults.get("base_seed", 42))
    )

    if args.list_models:
        for mid, spec in all_models.items():
            try:
                resolved = resolve_rf_model(mid, models_cfg)
                path = resolved.get("model_path_resolved", "—")
            except FileNotFoundError as ex:
                path = f"MISSING ({ex})"
            print(
                f"{mid}: {spec.get('name')} [{spec.get('feature_dim')} features] — {path}"
            )
        return

    if args.list:
        for sid, sc in scenarios.items():
            gt = sc.get("ground_truth") or {}
            print(
                f"{sid}: {sc.get('name')} [{sc.get('category')}] — "
                f"cross={gt.get('crossing_intent')} brake={gt.get('expects_brake')}"
            )
        return

    if args.all:
        selected_scenarios = list(scenarios.keys())
    elif args.scenario:
        selected_scenarios = [args.scenario]
    else:
        parser.error("Specify --scenario S1 or --all")

    if args.all_models:
        selected_models = list(all_models.keys())
    elif args.model:
        selected_models = [args.model]
    else:
        parser.error("Specify --model pie_9_heading (etc.) or --all-models")

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    for model_id in selected_models:
        if model_id not in all_models:
            print(f"Unknown RF model {model_id}", file=sys.stderr)
            continue
        try:
            model_spec = resolve_rf_model(model_id, models_cfg)
        except FileNotFoundError as ex:
            print(f"Skipping {model_id}: {ex}", file=sys.stderr)
            continue

        print(
            f"\n######## RF model: {model_id} — {model_spec.get('name')} "
            f"({model_spec.get('feature_dim')} features) ########",
            flush=True,
        )

        for sid in selected_scenarios:
            if sid not in scenarios:
                print(f"Unknown scenario {sid}", file=sys.stderr)
                continue
            sc = scenarios[sid]
            for run_idx in range(runs):
                env = build_env(sid, sc, model_id, model_spec, run_idx, base_seed, defaults)
                print(f"\n=== {sid} / {model_id} run {run_idx + 1}/{runs} ===", flush=True)
                print(f"  SIM_SCENARIO_ID={env.get('SIM_SCENARIO_ID')}", flush=True)
                print(f"  WALKER_SPAWN_PRESET={env.get('WALKER_SPAWN_PRESET')}", flush=True)
                print(f"  PIP_RF_MODEL={env.get('PIP_RF_MODEL') or '(unset)'}", flush=True)
                print(f"  RF_INTENT_MODEL={env.get('RF_INTENT_MODEL') or '(unset)'}", flush=True)
                print(
                    f"  YOLO model={env.get('YOLO_POSE_MODEL')} conf={env.get('YOLO_CONF')} "
                    f"min_bbox_h={env.get('YOLO_MIN_BBOX_H_PX')} aspect={env.get('YOLO_MIN_ASPECT_RATIO')}",
                    flush=True,
                )
                rc = subprocess.run([sys.executable, str(MAIN)], env=env, cwd=str(CARLA_DIR))
                if rc.returncode != 0:
                    print(f"Run failed with exit code {rc.returncode}", file=sys.stderr)

    if args.analyze:
        logs = find_run_jsonl_files()
        if not logs:
            print("No JSONL logs found under eval_sim/captures/", file=sys.stderr)
            return
        subprocess.run(
            [sys.executable, str(EVAL_SIM_DIR / "analyze.py"), "--captures", str(CAPTURE_DIR)],
            cwd=str(EVAL_SIM_DIR),
        )


if __name__ == "__main__":
    main()
