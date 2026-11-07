"""Step 2 check: holes on aluminium seen obliquely.

Usage: python checks/check_holes.py [scene folder]   (default: outputs/dataset/scenes/000000)
For the critical angles of STEP2_VARIANTS (75, 45, 30 deg) on one scene:
1. only aluminium pixels lose depth, never the table;
2. every removed pixel is seen more obliquely than the critical angle, every kept aluminium
   pixel less (the rule is applied exactly);
3. share of each part's pixels removed (more for smaller angles).
Figure: outputs/check_holes_<scene>.png (histogram of the viewing angle of the aluminium pixels, with the
critical angles; close-up of the ideal depth with the removed pixels in red for each angle).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from rgbd_sim.camera import incidence_angle
from rgbd_sim.camera_model import STEP2_VARIANTS, aluminium_holes_mask
from rgbd_sim.scene import load_scene

folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset/scenes/000000"
scene = load_scene(folder)
meta = scene["meta"]
label, depth = scene["label"], scene["depth"]
angle = np.degrees(incidence_angle(depth, scene["normals"], np.array(meta["camera"]["K"]),
                                   np.array(meta["camera"]["extrinsic_world_to_camera"])))
parts = label >= 2
print(folder)
masks = {}
angles = [v["holes_deg"] for v in STEP2_VARIANTS.values() if "holes_deg" in v]
for crit in angles:
    holes = aluminium_holes_mask(scene, crit)
    masks[crit] = holes
    on_table = (holes & (label == meta["table_instance_id"])).sum()
    wrong_removed = (holes & ~(angle > crit)).sum()
    wrong_kept = (parts & ~holes & (depth > 0) & (angle > crit)).sum()
    per_part = ", ".join(f"{o['class'][0]}{o['instance_id']} {100 * holes[label == o['instance_id']].mean():.0f}%"
                         for o in meta["objects"] if (label == o["instance_id"]).any())
    print(f"critical {crit} deg: removed {holes.sum()} px ({100 * holes[parts].mean():.1f} % of the parts); "
          f"on the table {on_table} (expected 0); rule violated: removed {wrong_removed}, kept {wrong_kept} "
          f"(expected 0, 0)\n    per part: {per_part}")

rows, cols = np.where(parts)
r = slice(max(rows.min() - 15, 0), rows.max() + 15)
c = slice(max(cols.min() - 15, 0), cols.max() + 15)
fig, ax = plt.subplots(1, 4, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1, 1]})
ax[0].hist(angle[parts & (depth > 0)], bins=45, range=(0, 90), color="#9a9893")
for crit in masks:
    ax[0].axvline(crit, color="#eb6834", ls="--")
    ax[0].text(crit, ax[0].get_ylim()[1], f" {crit}", color="#eb6834", va="top", fontsize=9)
ax[0].set_xlabel("viewing angle [deg] (0 = seen face-on)")
ax[0].set_ylabel("aluminium pixels")
ax[0].set_title("viewing angle of the aluminium pixels\n(dashed: critical angles)", fontsize=10)
grey = np.where(depth > 0, depth, np.nan)
for a, (crit, holes) in zip(ax[1:], masks.items()):
    a.imshow(grey[r, c], cmap="Greys_r", interpolation="nearest")
    overlay = np.zeros(holes[r, c].shape + (4,))
    overlay[holes[r, c]] = (0.92, 0.41, 0.2, 1.0)
    a.imshow(overlay, interpolation="nearest")
    a.set_title(f"critical angle {crit} deg\n{100 * holes[parts].mean():.0f} % of part pixels removed", fontsize=10)
    a.axis("off")
fig.suptitle(f"{folder}: aluminium pixels seen more obliquely than the critical angle lose their depth "
             f"(orange)", fontsize=11)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
name = f"outputs/check_holes_{os.path.basename(os.path.normpath(folder))}.png"
plt.savefig(name, dpi=120, bbox_inches="tight")
print(f"figure: {name}")
