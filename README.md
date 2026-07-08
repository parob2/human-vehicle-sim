# human-vehicle-sim

Simulation-based pipeline to bridge the human–vehicle communication gap: infer pedestrian crossing intent from roadside perception, disseminate it over V2X (DENM/CPM), and evaluate cooperative vehicle and infrastructure responses in CARLA on the `trail24` map ([5GoIng HD map](https://www.thi.de/en/research/research-at-thi/project-5going/)).

The repo combines PIE-trained crossing-intent prediction with a closed-loop CARLA scenario — RSU camera → intent model → V2X messages → ego braking and hazard signaling — plus batch evaluation (S1–S6) to compare models and scenarios offline and in simulation.

## Layout

```
human-vehicle-sim/
├── PedestrianIntentPrediction/   # PIE feature extraction, training, offline eval
├── CarlaSim/                     # CARLA simulation + eval_sim batch runner
├── PIE_dataset/                  # PIE dataset (local only — not in repo; see below)
├── .env.example                  # Environment variable template
└── requirements.txt
```

## External dependencies (not shipped in this repo)

| Dependency | How to obtain |
|------------|----------------|
| CARLA 0.9.16 + `trail24` map | Install CARLA per the [0.9.16 quick start guide](https://carla.readthedocs.io/en/0.9.16/start_quickstart/); add the custom `trail24` map from [5GoIng First Mile Test Field HD map](https://www.thi.de/en/research/research-at-thi/project-5going/). Run the server on `localhost:2000`. |
| `yolov8m-pose.pt` | Download from [Ultralytics](https://docs.ultralytics.com); place in `CarlaSim/` |
| `rf_ma_rong_pie_heading.joblib` | Train via `PedestrianIntentPrediction/pie/train_rf_pie_heading.py`, or copy a trained artifact into `pie/models/` |
| PIE dataset (annotations + videos) | Clone [aras62/PIE](https://github.com/aras62/PIE) into `PIE_dataset/`, then `bash PedestrianIntentPrediction/pie/download_pie_videos.sh set02 set04` (~2.5 GB minimal clips) |

Gitignored outputs — regenerate locally: simulation captures (`simulation_captures/`, `eval_sim/captures/`, `eval_sim/results/`) and, for offline PIE eval/retraining, the feature cache via `PedestrianIntentPrediction/pie/run_full_pipeline.sh` (see [PedestrianIntentPrediction/README.md](PedestrianIntentPrediction/README.md)).

## Setup

```bash
cd human-vehicle-sim
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set CARLA_ROOT, optional REPO_ROOT
# Install CARLA: https://carla.readthedocs.io/en/0.9.16/start_quickstart/

# PIE dataset (not in repo): clone official annotations, then download videos
git clone https://github.com/aras62/PIE.git PIE_dataset
bash PedestrianIntentPrediction/pie/download_pie_videos.sh set02 set04   # ~2.5 GB minimal
```

Place `yolov8m-pose.pt` in `CarlaSim/` (or set `YOLO_POSE_MODEL` in `.env`).

## Offline PIE model evaluation

```bash
cd PedestrianIntentPrediction
python3 pie/evaluate_pie_heading.py
```

Requires `pie/models/rf_ma_rong_pie_heading.joblib` and `pie/features_cache/pie_ma_rong_features_full_heading.jsonl` (see [PedestrianIntentPrediction/README.md](PedestrianIntentPrediction/README.md)).

## CARLA simulation

```bash
cd CarlaSim
python3 main.py
```

Default RF: `PedestrianIntentPrediction/pie/models/rf_ma_rong_pie_heading.joblib`.

### Pipeline (`main.py`)

RSU RGB camera → YOLOv8 pose (ByteTrack) → PIE 9-feature RF crossing-intent → RSU transmits DENM/CPM over `sensor.other.v2x_custom`. Ego OBU receives V2X and applies graded TTC braking; ARI robot light follows ego CAM + received DENM via `hazard_decision_policy`.

Optional CARLA walkers provide ground-truth comparison (`DISABLE_WALKER_SPAWN=1` to disable). Walker geometry comes from spawn presets (`WALKER_SPAWN_PRESET`; default module `eval.spawn_presets`, batch eval uses `eval_sim.spawn_presets`).

Ego navigation: `EGO_NAV_MODE=lane` (default), `simple`, or `basic` (BasicAgent; requires `CARLA_ROOT` / `CARLA_PYTHONAPI_CARLA`).

### Common environment variables

Copy `.env.example` to `.env` for paths. Additional toggles (defaults in `sim_config.py`):

| Variable | Purpose |
|----------|---------|
| `YOLO_POSE_MODEL`, `YOLO_IMGSZ`, `YOLO_CONF`, `YOLO_DEVICE` | Pose detection / VRAM tuning |
| `PIP_RF_MODEL`, `PIP_RF_THRESHOLD` | Intent model path and decision threshold |
| `EGO_NAV_MODE`, `EGO_NAV_TARGET_SPEED_KMH` | Ego route following |
| `WALKER_SPAWN_PRESET`, `SIM_SPAWN_PRESETS` | Walker spawn geometry (`eval.spawn_presets` vs `eval_sim.spawn_presets`) |
| `RSU_MASTER_RANGE_M` | RSU master arbitration range (m) |
| `SIM_RUN_LOGGER`, `SIM_SCENARIO_ID`, `SIM_CAPTURE_DIR` | Validation logging |

For VRAM pressure (CARLA + YOLO on one GPU): lower `YOLO_IMGSZ` (default 1280), set `YOLO_DEVICE=cpu`, or `YOLO_HALF=0`.

## Scenario batch evaluation (thesis S1–S6)

```bash
cd CarlaSim/eval_sim
python3 run_scenarios.py --list
python3 run_scenarios.py --scenario S2 --model pie_9_heading --runs 3
python3 run_scenarios.py --all --model pie_9_heading --runs 3 --analyze
python3 generate_report.py --results results/
```

See [CarlaSim/eval_sim/README.md](CarlaSim/eval_sim/README.md) for scenario definitions and the analysis pipeline.

## PIE dataset

The [PIE dataset](http://data.nvision2.eecs.yorku.ca/PIE_dataset/) (Rasouli et al., ICCV 2019) is **not included** in this repository. Clone the official annotations from [github.com/aras62/PIE](https://github.com/aras62/PIE) into `PIE_dataset/`, then download video clips (see [PedestrianIntentPrediction/README.md](PedestrianIntentPrediction/README.md)). The entire `PIE_dataset/` directory is gitignored so you can keep a local copy without committing third-party data.

## References

**Ma & Rong (2022)** — crossing-intent Random Forest (7/9-feature Ma-Rong fusion). *World Electric Vehicle Journal* 13(8):158. [doi:10.3390/wevj13080158](https://doi.org/10.3390/wevj13080158)

```bibtex
@article{ma2022pedestrian,
  author         = {Ma, Jun and Rong, Wenhui},
  title          = {{Pedestrian} {Crossing} {Intention} {Prediction} {Method} {Based} on {Multi}-{Feature} {Fusion}},
  journal        = {World Electric Vehicle Journal},
  volume         = {13},
  year           = {2022},
  number         = {8},
  articleno      = {158},
  issn           = {2032-6653},
  doi            = {10.3390/wevj13080158}
}
```

**Rasouli et al. (2019)** — PIE dataset ([official source](https://github.com/aras62/PIE); not shipped in this repo). ICCV 2019, pp. 6261–6270. [doi:10.1109/ICCV.2019.00636](https://doi.org/10.1109/ICCV.2019.00636)

```bibtex
@inproceedings{rasouli2019pie,
  author    = {Rasouli, Amir and Kotseruba, Iuliia and Kunic, Toni and Tsotsos, John K.},
  booktitle = {2019 IEEE/CVF International Conference on Computer Vision (ICCV)},
  title     = {{PIE}: {A} {Large}-{Scale} {Dataset} and {Models} for {Pedestrian} {Intention} {Estimation} and {Trajectory} {Prediction}},
  year      = {2019},
  pages     = {6261--6270},
  doi       = {10.1109/ICCV.2019.00636}
}
```

**5GoIng (2025)** — `trail24` CARLA environment (First Mile Test Field HD map). [Project page](https://www.thi.de/en/research/research-at-thi/project-5going/) (accessed 2026-06-12).

```bibtex
@misc{5going_hdmap_2025,
  title  = {{HD} {Map} of the {First} {Mile} {Test} {Field}},
  author = {{5GoIng Research and Development Project}},
  year   = {2025},
  url    = {https://www.thi.de/en/research/research-at-thi/project-5going/},
  note   = {Accessed: 2026-06-12}
}
```
