"""Check of the part/table segmentation on one scene, ideal and realistic depth.

Usage: python checks/check_segment.py [scene folder]   (default: outputs/dataset/scenes/000000)

For t = 2, 4, 6, 8, 12 mm and both depths:
1. fitted table plane vs the true one: angle between the normals and difference in height at the
   camera's line of sight (expected ~0 on ideal depth);
2. per part: share of its pixels marked "part" (found if >= 50 %);
3. pixel precision: share of the "part" pixels that really are a part (expected 1.00 on ideal).

Figure: outputs/check_segment_<scene>.png (close-up, one panel per depth and t):
  blue = part pixel marked part, orange = part pixel missed, green = table pixel marked part.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from rgbd_sim.scene import load_depth, load_scene
from rgbd_sim.segment import evaluate, segment_parts, true_table_plane

BLUE, ORANGE, AQUA = (0.16, 0.47, 0.84), (0.92, 0.41, 0.2), (0.11, 0.69, 0.48)
T_MM = [2, 4, 6, 8, 12]

folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset/scenes/000000"
scene = load_scene(folder)
meta, label = scene["meta"], scene["label"]
K = np.array(meta["camera"]["K"])
n_true, d_true = true_table_plane(meta)
print(folder)

panels = []
for variant in ("ideal", "realistic"):
    depth = load_depth(folder, variant)
    for t in T_MM:
        mask, (n, d) = segment_parts(depth, K, t / 1000)
        angle = np.degrees(np.arccos(np.clip(n @ n_true, -1, 1)))
        res = evaluate(mask, label, meta)
        parts = ", ".join(f"{c[0]}{i} {100 * r:.0f}%" + ("" if ok else " MISSED") for i, c, r, ok in res["parts"])
        print(f"{variant:9s} t = {t:2d} mm: plane angle {angle:.2f} deg, offset {1000 * (d - d_true):+.2f} mm | "
              f"found {100 * res['found_rate']:.0f} %, precision {res['precision']:.3f}, F1 {res['f1']:.3f}\n"
              f"          per part: {parts}")
        panels.append((variant, t, mask, depth))

rows, cols = np.where(label >= 2)
r = slice(max(rows.min() - 25, 0), rows.max() + 25)
c = slice(max(cols.min() - 25, 0), cols.max() + 25)
is_part = label >= 2
fig, ax = plt.subplots(2, len(T_MM), figsize=(3.4 * len(T_MM), 6.2))
for k, (variant, t, mask, depth) in enumerate(panels):
    a = ax[k // len(T_MM), k % len(T_MM)]
    img = np.ones(label.shape + (3,)) * 0.92
    img[depth == 0] = 1.0
    img[mask & is_part] = BLUE
    img[~mask & is_part] = ORANGE
    img[mask & ~is_part] = AQUA
    a.imshow(img[r, c], interpolation="nearest")
    a.set_title(f"{variant}, t = {t} mm", fontsize=10)
    a.axis("off")
fig.suptitle(f"{folder}: blue = part found, orange = part missed, green = table marked as part", fontsize=11)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
name = f"outputs/check_segment_{os.path.basename(os.path.normpath(folder))}.png"
plt.savefig(name, dpi=120, bbox_inches="tight")
print(f"figure: {name}")
