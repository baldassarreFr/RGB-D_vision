"""Figure of the part/table segmentation of one evaluation scene in conditions A, B and C.

Usage (after scripts/run_segment_experiment.py):
    python scripts/make_seg_figure.py [--seed S] [--eval outputs/dataset_eval] [--exp outputs/segmentation]
Scene: the evaluation scene with seed S; without --seed, the same scene as fig_scene.png (the first
evaluation scene with both part types). Thresholds: chosen.json, as in the experiment
(A = t tuned on ideal, on ideal depth; B = t tuned on ideal, on realistic depth;
C = t tuned on realistic, on realistic depth). Writes docs/figures/fig_seg_example.png.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from rgbd_sim.scene import load_depth, load_scene
from rgbd_sim.segment import evaluate, segment_parts

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--eval", default="outputs/dataset_eval")
parser.add_argument("--exp", default="outputs/segmentation")
parser.add_argument("--figdir", default="docs/figures")
parser.add_argument("--seed", type=int, help="seed of the evaluation scene to show")
args = parser.parse_args()
with open(os.path.join(args.exp, "chosen.json")) as f:
    chosen = json.load(f)
t_ideal, t_real = chosen["tuned_on_ideal"]["t_mm"], chosen["tuned_on_realistic"]["t_mm"]

scenes = sorted(os.listdir(os.path.join(args.eval, "scenes")))
name = f"{args.seed:06d}" if args.seed is not None else next(s for s in scenes
            if {o["class"] for o in load_scene(os.path.join(args.eval, "scenes", s))["meta"]["objects"]}
            == {"profile", "bracket"})
folder = os.path.join(args.eval, "scenes", name)
scene = load_scene(folder)
label, meta = scene["label"], scene["meta"]
K = np.array(meta["camera"]["K"])
is_part = label >= 2

r, c = np.where(is_part)
m = 25
crop = (slice(max(r.min() - m, 0), r.max() + m + 1), slice(max(c.min() - m, 0), c.max() + m + 1))

# pixel classes: 0 table not marked, 1 part marked, 2 table marked as part, 3 part missed, 4 no depth
COLOURS = ["#f2f1ee", "#52514e", "#e34948", "#7fb3ea", "white"]
cmap = ListedColormap(COLOURS)
panels = [(f"A  ideal depth, t = {t_ideal} mm", "ideal", t_ideal),
          (f"B  realistic depth, t = {t_ideal} mm", "realistic", t_ideal),
          (f"C  realistic depth, t = {t_real} mm", "realistic", t_real)]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, (title, variant, t_mm) in zip(axes, panels):
    depth = load_depth(folder, variant)
    mask, _ = segment_parts(depth, K, t_mm / 1000, seed=0)
    res = evaluate(mask, label, meta)
    cls = np.zeros(label.shape, int)
    cls[mask & is_part] = 1
    cls[mask & ~is_part] = 2
    cls[~mask & is_part] = 3
    cls[(depth <= 0) & ~is_part] = 4
    ax.imshow(cls[crop], cmap=cmap, vmin=0, vmax=len(COLOURS) - 1, interpolation="nearest")
    found = sum(p[3] for p in res["parts"])
    ax.set_title(f"{title}\nparts found {found}/{len(res['parts'])}, pixel precision "
                 f"{100 * res['precision']:.0f} %", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
legend = [Patch(color=COLOURS[1], label="part, marked as part"),
          Patch(color=COLOURS[2], label="table, marked as part"),
          Patch(color=COLOURS[3], label="part, not marked"),
          Patch(facecolor=COLOURS[0], edgecolor="#b0afab", label="table, not marked"),
          Patch(facecolor="white", edgecolor="#b0afab", label="no depth")]
fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"Part/table segmentation of evaluation scene {name}", fontsize=10)
plt.tight_layout(rect=(0, 0.05, 1, 1))
os.makedirs(args.figdir, exist_ok=True)
out = os.path.join(args.figdir, "fig_seg_example.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("written", out)
