"""Check the stereo-shadow effect on one scene.

Usage: python checks/check_shadows.py [scene folder]   (default: outputs/dataset/scenes/000000)

Checks:
1. invalid band on the left of the image: its width in each row must equal the disparity
   fx * b / z of the first point the right imager can see (about 24-46 px in our scenes);
2. every shadow beside an object must have its occluder immediately to its right (the right
   imager looks "around" the right side of an object, missing the strip just left of it);
3. how many pixels lose their depth.

Figure: outputs/check_shadows.png (whole image and close-up; red = depth removed).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from rgbd_sim import d435
from rgbd_sim.camera_model import VISIBILITY_TOLERANCE_M, stereo_shadow_mask
from rgbd_sim.scene import load_scene

folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset/scenes/000000"
scene = load_scene(folder)
depth, label, meta = scene["depth"], scene["label"], scene["meta"]
fx = meta["camera"]["K"][0][0]
valid = depth > 0
shadow = stereo_shadow_mask(scene)
print(f"{folder}")

# ---------- 1. Invalid band on the left ----------
errors = []
for row in range(depth.shape[0]):
    if not valid[row, 0] or label[row, 0] != 1:          # only rows that start on the table
        continue
    kept = np.where(valid[row] & ~shadow[row])[0]
    if len(kept) == 0:
        continue
    width = kept[0]                                      # columns removed from the left edge
    predicted = fx * d435.BASELINE_M / depth[row, width]
    errors.append(width - predicted)
errors = np.array(errors)
print(f"left band: {len(errors)} rows, width - disparity = {errors.mean():+.2f} px on average, "
      f"max |difference| {np.abs(errors).max():.2f} px   (expected within ~1 px)")

# ---------- 2. Every shadow must have its occluder just to its right ----------
band_width = fx * d435.BASELINE_M / np.where(valid, depth, np.inf)      # disparity in px
max_disparity = fx * d435.BASELINE_M / depth[valid].min()
cols_idx = np.arange(depth.shape[1])
# The occluder only has to exist and be closer; it may itself be hidden from the right imager
# by something even closer (a chain of occlusions), so all pixels with depth are candidates.
checked = ok = 0
failures = []
for row in range(depth.shape[0]):
    for col in np.where(shadow[row] & (cols_idx >= band_width[row]))[0]:   # skip the left band
        reach = int(np.ceil(max_disparity - band_width[row, col])) + 1
        window = slice(col + 1, col + 1 + reach)
        closer = valid[row, window] & (depth[row, window] < depth[row, col] - VISIBILITY_TOLERANCE_M)
        checked += 1
        ok += closer.any()
        if not closer.any():
            failures.append((row, col, reach))
print(f"shadows beside objects: {checked} px; occluder found to the right in "
      f"{100 * ok / checked:.1f} %   (expected ~100 %)")
for row, col, reach in failures[:10]:                     # show the exceptions, if any
    right = depth[row, col + 1:col + 1 + reach] * 1000
    print(f"  no occluder found: row {row}, col {col}, instance {label[row, col]}, "
          f"depth {depth[row, col] * 1000:.1f} mm; next {reach} px to the right [mm]: "
          + " ".join(f"{d:.1f}" if d > 0 else "-" for d in right))
for obj in meta["objects"]:
    print(f"  instance {obj['instance_id']} {obj['class']:8s}: "
          f"{(shadow & (label == obj['instance_id'])).sum():4d} of its own pixels removed")

# ---------- 3. How much is lost ----------
print(f"pixels with depth: {valid.sum()}; removed: {shadow.sum()} "
      f"({100 * shadow.sum() / valid.sum():.1f} %), of which parts: "
      f"{(shadow & (label >= 2)).sum()} px")

# ---------- Figure ----------
grey = 0.4 + 0.6 * (depth - depth[valid].min()) / np.ptp(depth[valid])     # near = dark, far = light
rgb = np.dstack([np.clip(np.where(valid, grey, 1.0), 0, 1)] * 3)               # no depth = white
rgb[shadow] = [1, 0, 0]
rows, cols = np.where(label >= 2)
r = slice(max(rows.min() - 15, 0), rows.max() + 15)
c = slice(max(cols.min() - 15, 0), cols.max() + 15)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].imshow(rgb, interpolation="nearest")
ax[0].set_title("Depth (grey) and depth removed by stereo shadows (red)")
ax[1].imshow(rgb[r, c], interpolation="nearest")
ax[1].set_title("Close-up: shadows on the left of each part")
for a in ax:
    a.axis("off")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/check_shadows.png", dpi=110)
