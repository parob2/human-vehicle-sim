#!/usr/bin/env python3
"""Bar chart of per-feature MDI for the pie_9_heading RF.

Outputs PNG (300 dpi) and PDF in pie/figures/.

    python3 pie/plot_feature_importance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pie.config_pie import DEFAULT_PIE_HEADING_EVAL_OUT

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_STEM = OUT_DIR / "feature_importance_pie_9_heading"

GROUP_COLORS = {
    "spatial": "#b6d7a8",
    "kinematic": "#ffe599",
    "pose": "#b4a7d6",
}
EDGE_COLOR = "#555555"
TEXT_COLOR = "#333333"


def main() -> None:
    json_path = DEFAULT_PIE_HEADING_EVAL_OUT / "feature_importance_full.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))

    features = sorted(data["features"], key=lambda f: f["importance"])
    names = [f["name"] for f in features]
    values = [f["importance"] for f in features]
    colors = [GROUP_COLORS[f["group"]] for f in features]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(
        names,
        values,
        color=colors,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        height=0.65,
    )

    ax.set_xlabel("MDI")
    ax.set_title("Feature importance — pie_9_heading")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR)
    ax.set_xlim(0, max(values) * 1.12)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
        )

    ax.legend(
        handles=[
            Patch(facecolor=GROUP_COLORS["spatial"], edgecolor=EDGE_COLOR, label="Spatial"),
            Patch(facecolor=GROUP_COLORS["kinematic"], edgecolor=EDGE_COLOR, label="Kinematic"),
            Patch(facecolor=GROUP_COLORS["pose"], edgecolor=EDGE_COLOR, label="Pose"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=8,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_STEM}.png and {OUT_STEM}.pdf")


if __name__ == "__main__":
    main()
