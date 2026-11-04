"""Apply the D435i camera model to every scene of the dataset, in every variant.

Usage (after scripts/generate_dataset.py):  python scripts/make_camera_variants.py
For each scene folder writes depth_<variant>.png next to depth.png (the ideal depth), for the
variants in camera_model.VARIANTS. Same seed -> same result: the random numbers of a scene and
variant come from (scene seed, variant number).
Also writes, in the dataset folder:
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
from rgbd_sim.camera_model import VARIANTS, apply_camera
from rgbd_sim.scene import load_scene, save_depth

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dataset", default="outputs/dataset", help="dataset folder")
args = parser.parse_args()
scene_dirs = sorted(os.path.join(args.dataset, "scenes", d)
                    for d in os.listdir(os.path.join(args.dataset, "scenes")))

# ---------- 1. Every scene, every variant ----------
stats = {name: {"kept": [], "err_table": [], "err_parts": []} for name in VARIANTS}
example = {}
t0 = time.time()
for folder in scene_dirs:
    scene = load_scene(folder)
    ideal, label = scene["depth"], scene["label"]
    for i, (name, settings) in enumerate(VARIANTS.items()):
        rng = np.random.default_rng([scene["meta"]["seed"], i])
        depth = apply_camera(scene, rng=rng, **settings)
        if name != "ideal":
            save_depth(depth, os.path.join(folder, f"depth_{name}.png"))
        both = (ideal > 0) & (depth > 0)
        stats[name]["kept"].append(both.sum() / (ideal > 0).sum())
        error = np.abs(depth - ideal) * 1000
        stats[name]["err_table"].append(np.mean(error[both & (label == 1)]))
        stats[name]["err_parts"].append(np.mean(error[both & (label >= 2)]))
        if folder == scene_dirs[0]:
            example[name] = depth
    print(f"{folder}: done   [{time.time() - t0:.0f} s]")

# ---------- 2. Summary over all scenes ----------
summary = {}
print(f"\n{'variant':10s} {'depth kept':>11s} {'mean error table':>17s} {'mean error parts':>17s}")
for name, st in stats.items():
    summary[name] = {"settings": VARIANTS[name],
                     "depth_kept_percent": 100 * float(np.mean(st["kept"])),
                     "mean_abs_error_table_mm": float(np.mean(st["err_table"])),
                     "mean_abs_error_parts_mm": float(np.mean(st["err_parts"]))}
    s = summary[name]
    print(f"{name:10s} {s['depth_kept_percent']:10.2f}% {s['mean_abs_error_table_mm']:14.2f} mm"
          f" {s['mean_abs_error_parts_mm']:14.2f} mm")
print("(error = |measured - ideal| on pixels that have depth in both; mean over scenes)")
with open(os.path.join(args.dataset, "camera_variants_summary.json"), "w") as f:
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
plt.savefig(os.path.join(args.dataset, "camera_variants.png"), dpi=110)
