# eval_sim — Thesis Simulation Evaluation

Batch runner for scenarios **S1–S6** defined in `scenarios.yaml`. RF models in `rf_models.yaml`:

| Model ID | Features | Artifact |
|----------|----------|----------|
| `pie_7` | 7 (Ma & Rong paper) | `PedestrianIntentPrediction/pie/models/rf_ma_rong_pie.joblib` |
| `pie_9_heading` | 9 (+ trajectory heading) | `PedestrianIntentPrediction/pie/models/rf_ma_rong_pie_heading.joblib` |

## Scenarios (S1–S6)

| ID | Spawn preset | Intent |
|----|--------------|--------|
| S1 | `sidewalk_walk` | No crossing — pedestrian walks along sidewalk |
| S2 | `nearside_off` | Normal nearside crossing (default `main.py` geometry) |
| S3 | `talking_pair` | Two stationary pedestrians — false-positive test |
| S4 | `slow_crossing` | Very slow crossing (0.35 m/s) |
| S5 | `multi_cross` | Three pedestrians crossing simultaneously |
| S6 | `running_cross` | Fast crossing (2.8 m/s) — late braking |

Presets live in `eval_sim/spawn_presets.py` (loaded via `SIM_SPAWN_PRESETS=eval_sim.spawn_presets`).

## Quick Start

Requires CARLA server on `localhost:2000` with map `trail24`.

```bash
cd CarlaSim/eval_sim

python3 run_scenarios.py --list
python3 run_scenarios.py --list-models
python3 run_scenarios.py --scenario S2 --model pie_7 --runs 3
python3 run_scenarios.py --scenario S2 --model pie_9_heading --runs 3
python3 run_scenarios.py --all --model pie_9_heading --runs 3 --analyze
python3 hazard_decision_eval.py
python3 generate_report.py --results results/
```

## Pipeline

```
run_scenarios.py  →  captures/{model_id}/S*/run_*/run.jsonl
       ↓
analyze.py        →  results/per_run.json, aggregate.json
       ↓
generate_report.py → results/evaluation_report.md
```

## Environment

`run_scenarios.py` sets `SIM_SPAWN_PRESETS=eval_sim.spawn_presets`, `PIP_RF_MODEL`, and YOLO gates automatically. See repo-root `.env.example` for overrides.

Captures and results are gitignored; regenerate locally or keep archived copies outside the repo.
