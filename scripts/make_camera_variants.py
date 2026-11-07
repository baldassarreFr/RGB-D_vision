"""Apply the D435i camera model to every scene of the dataset, in every variant.

Usage (after scripts/generate_dataset.py):
    python scripts/make_camera_variants.py [--dataset D]                  # Step 1 variants
    python scripts/make_camera_variants.py [--dataset D] --set step2      # Step 2 variants
    python scripts/make_camera_variants.py [--dataset D] --figure-only    # only redraw the figure
For each scene folder writes depth_<variant>.png next to depth.png (the ideal depth).
  step1: the variants in camera_model.VARIANTS; random numbers from (scene seed, variant number).
  step2: the variants in camera_model.STEP2_VARIANTS, written to NEW files (depth_realistic_*.png);
         no Step 1 file is touched. They use the same random numbers as "realistic", so each
         differs from "realistic" only by its extra effect.
Also writes, in the dataset folder (step2: names with "_step2"):
  camera_variants_summary.json   per variant: depth kept, typical error on table and parts
  camera_variants.png            one scene in every variant (close-up + 3D points from the front)
"""
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from rgbd_sim.camera import backproject, camera_to_world
from rgbd_sim.camera_model import STEP2_VARIANTS, VARIANTS, apply_camera
from rgbd_sim.scene import load_scene, save_depth

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dataset", default="outputs/dataset", help="dataset folder")
parser.add_argument("--set", choices=["step1", "step2"], default="step1")
parser.add_argument("--figure-only", action="store_true", help="first scene only, write no depth files")
args = parser.parse_args()
scene_dirs = sorted(os.path.join(args.dataset, "scenes", d)
                    for d in os.listdir(os.path.join(args.dataset, "scenes")))
if args.figure_only:
    scene_dirs = scene_dirs[:1]
REALISTIC_INDEX = list(VARIANTS).index("realistic")
if args.set == "step1":
    VARIANT_SET = {name: (i, settings) for i, (name, settings) in enumerate(VARIANTS.items())}
    suffix = ""
else:
    # "realistic" is recomputed for the figure only (never written); every Step 2 variant uses
    # the random numbers of "realistic"
    VARIANT_SET = {"realistic": (REALISTIC_INDEX, VARIANTS["realistic"]),
                   **{name: (REALISTIC_INDEX, settings) for name, settings in STEP2_VARIANTS.items()}}
    suffix = "_step2"

# ---------- 1. Every scene, every variant ----------
stats = {name: {"kept": [], "err_table": [], "err_parts": []} for name in VARIANT_SET}
example = {}
t0 = time.time()
for folder in scene_dirs:
    scene = load_scene(folder)
    ideal, label = scene["depth"], scene["label"]
    for name, (i, settings) in VARIANT_SET.items():
        rng = np.random.default_rng([scene["meta"]["seed"], i])
        depth = apply_camera(scene, rng=rng, **settings)
        write = name != "ideal" and not args.figure_only and (args.set == "step1" or name in STEP2_VARIANTS)
        if write:
            save_depth(depth, os.path.join(folder, f"depth_{name}.png"))
        both = (ideal > 0) & (depth > 0)
        stats[name]["kept"].append(both.sum() / (ideal > 0).sum())
        error = np.abs(depth - ideal) * 1000
        stats[name]["err_table"].append(np.mean(error[both & (label == 1)]))
        parts = both & (label >= 2)
        stats[name]["err_parts"].append(np.mean(error[parts]) if parts.any() else np.nan)
        if folder == scene_dirs[0]:
            example[name] = depth
    print(f"{folder}: done   [{time.time() - t0:.0f} s]")

# ---------- 2. Summary over all scenes ----------
summary = {}
print(f"\n{'variant':18s} {'depth kept':>11s} {'mean error table':>17s} {'mean error parts':>17s}")
for name, st in stats.items():
    summary[name] = {"settings": VARIANT_SET[name][1],
                     "depth_kept_percent": 100 * float(np.mean(st["kept"])),
                     "mean_abs_error_table_mm": float(np.mean(st["err_table"])),
                     "mean_abs_error_parts_mm": float(np.nanmean(st["err_parts"]))}
    s = summary[name]
    print(f"{name:18s} {s['depth_kept_percent']:10.2f}% {s['mean_abs_error_table_mm']:14.2f} mm"
          f" {s['mean_abs_error_parts_mm']:14.2f} mm")
print("(error = |measured - ideal| on pixels that have depth in both; mean over scenes)")
if not args.figure_only:
    with open(os.path.join(args.dataset, f"camera_variants{suffix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

# ---------- 3. Figure: the first scene in every variant ----------
scene = load_scene(scene_dirs[0])
meta, label = scene["meta"], scene["label"]
K = np.array(meta["camera"]["K"])
extrinsic = np.array(meta["camera"]["extrinsic_world_to_camera"])
rows, cols = np.where(label >= 2)
r = slice(max(rows.min() - 15, 0), rows.max() + 15)
c = slice(max(cols.min() - 15, 0), cols.max() + 15)
ideal = scene["depth"]
vmin, vmax = ideal[r, c][ideal[r, c] > 0].min(), ideal[r, c].max()
fig, ax = plt.subplots(2, len(example), figsize=(3.6 * len(example), 6.5), squeeze=False)
for j, (name, d) in enumerate(example.items()):
    ax[0, j].imshow(np.where(d > 0, d, np.nan)[r, c], cmap="viridis", vmin=vmin, vmax=vmax,
                    interpolation="nearest")
    ax[0, j].set_title(name)
    ax[0, j].axis("off")
    pts = camera_to_world(backproject(d, K), extrinsic)[r, c][(d > 0)[r, c]][::2] * 1000
    ax[1, j].scatter(pts[:, 0], pts[:, 2], s=0.2, c="k")
    ax[1, j].set_ylim(-20, 60)
    ax[1, j].set_xlabel("world x [mm]")
ax[1, 0].set_ylabel("height z [mm]")
fig.suptitle(f"{scene_dirs[0]}: depth (top; white = no depth) and 3D points seen from the front (bottom)")
plt.tight_layout()
plt.savefig(os.path.join(args.dataset, f"camera_variants{suffix}.png"), dpi=110, bbox_inches="tight")
print(f"figure: {os.path.join(args.dataset, f'camera_variants{suffix}.png')}")
