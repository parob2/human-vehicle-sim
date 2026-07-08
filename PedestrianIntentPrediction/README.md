# PedestrianIntentPrediction — PIE pipeline

Ma & Rong (2022) pedestrian crossing-intent Random Forest on the [PIE dataset](http://data.nvision2.eecs.yorku.ca/PIE_dataset/), 9-feature heading variant used in CARLA simulation.

## PIE dataset (not included in this repo)

The [PIE dataset](http://data.nvision2.eecs.yorku.ca/PIE_dataset/) is third-party work by Rasouli et al. (ICCV 2019, [MIT license](https://github.com/aras62/PIE/blob/master/LICENSE)). It is **not** shipped in this repository — obtain it from the official source and keep it locally.

### 1. Annotations and utilities

From the repository root:

```bash
git clone https://github.com/aras62/PIE.git PIE_dataset
```

This provides `annotations/`, `annotations_attributes/`, `annotations_vehicle/`, `camera_params/`, and `utilities/` under `PIE_dataset/`.

Official sources:

- Dataset page: [http://data.nvision2.eecs.yorku.ca/PIE_dataset/](http://data.nvision2.eecs.yorku.ca/PIE_dataset/)
- Annotations repo: [https://github.com/aras62/PIE](https://github.com/aras62/PIE)

### 2. Video clips

Videos are **not** in the annotations repo (~74 GB total). Download into `PIE_dataset/PIE_clips/`:

```bash
bash pie/download_pie_videos.sh set02 set04   # minimal ~2.5 GB
# or all sets:
bash pie/download_pie_videos.sh
```

Alternatively, download from the [YorkU server](http://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/) or [Google Drive](https://drive.google.com/drive/folders/180MXX1z3aicZMwYu2pCM0TamzUKT0L16?usp=drive_link).

Override `PIE_ROOT` in repo-root `.env` if the dataset lives elsewhere.

## Quick start

```bash
cd PedestrianIntentPrediction

# Offline evaluation (70/30 pedestrian holdout)
python3 pie/evaluate_pie_heading.py

# Retrain (requires pie/features_cache/pie_ma_rong_features_full_heading.jsonl)
python3 pie/train_rf_pie_heading.py

# Full extraction pipeline (requires PIE_clips/ videos)
bash pie/run_full_pipeline.sh
```

## Key paths

| Artifact | Path |
|----------|------|
| Trained model (9-feature) | `pie/models/rf_ma_rong_pie_heading.joblib` |
| Trained model (7-feature) | `pie/models/rf_ma_rong_pie.joblib` |
| PIE dataset (local only) | `../PIE_dataset/` |
| Training metrics | `pie/models/train_metrics_pie_heading.json` |

## Environment

See repo-root `.env.example`.

## CARLA runtime adapter

`pie/carla_adapter.py` builds live 7- or 9-feature vectors for `CarlaSim/main.py` from RSU YOLO pose detections.

## Optional tooling

| Script | Purpose |
|--------|---------|
| `pie/plot_feature_importance.py` | Feature-importance plots from eval output |
