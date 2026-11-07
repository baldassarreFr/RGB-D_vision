"""
Look at the noise effect on one scene, in its two variants, and check that it is right.

Usage: python checks/check_noise.py [scene folder]   (default: outputs/dataset/scenes/000000)
Variants: base only | base x aluminium.

Checks:
1. the noise added has exactly the intended size: (noisy - ideal) / sigma must have std ~1;
2. typical sigma on the table, on the tops of the parts and on their steep faces.

Figure outputs/check_noise.png, one column per variant:
  row 1 depth (close-up), row 2 error added [mm], row 3 the 3D points seen from the front
  (height vs left-right): the "trembling" shows as thickness of the lines.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from rgbd_sim.camera import backproject, camera_to_world, incidence_angle
from rgbd_sim.camera_model import add_noise, noise_sigma
from rgbd_sim.scene import load_scene

folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset/scenes/000000"
scene = load_scene(folder)
depth, label, meta = scene["depth"], scene["label"], scene["meta"]
K = np.array(meta["camera"]["K"])
extrinsic = np.array(meta["camera"]["extrinsic_world_to_camera"])
valid = depth > 0

VARIANTS = {"base": False, "base x aluminium": True}     # name: material factor on/off

# Pixel groups for the summary: table, part faces seen at < 45 deg (tops), at >= 45 deg (steep)
# (the angle is only used to group pixels; the noise does not depend on it)
angle = np.degrees(incidence_angle(depth, scene["normals"], K, extrinsic))
groups = {"table": label == 1,
          "parts < 45 deg": (label >= 2) & (angle < 45),
          "parts >= 45 deg": (label >= 2) & (angle >= 45)}

results = {"ideal": depth}
print(f"{folder}")
print(f"{'variant':18s} {'std of z-score':>14s}   " + "   ".join(f"{g:>16s}" for g in groups))
for name, use_material in VARIANTS.items():
    sigma = noise_sigma(scene, material=use_material)
    noisy = add_noise(depth, sigma, np.random.default_rng(0))   # same random numbers for all
    results[name] = noisy
    z_score = (noisy - depth)[valid] / sigma[valid]
    medians = [np.median(sigma[g]) * 1000 for g in groups.values()]
    print(f"{name:18s} {z_score.std():14.3f}   " + "   ".join(f"{m:13.1f} mm" for m in medians))
print("(z-score std must be ~1.000; the three columns are the median sigma of each group)")

# ---------- Figure ----------
rows, cols = np.where(label >= 2)
r = slice(max(rows.min() - 15, 0), rows.max() + 15)
c = slice(max(cols.min() - 15, 0), cols.max() + 15)
fig, ax = plt.subplots(3, len(results), figsize=(4 * len(results), 9), squeeze=False)
ideal_pts = camera_to_world(backproject(depth, K), extrinsic)[r, c][valid[r, c]] * 1000
x_range = (ideal_pts[:, 0].min() - 10, ideal_pts[:, 0].max() + 10)   # same x-axis everywhere
for j, (name, d) in enumerate(results.items()):
    crop = np.where(valid, d, np.nan)[r, c]
    ax[0, j].imshow(crop, cmap="viridis", vmin=np.nanmin(depth[r, c][valid[r, c]]),
                    vmax=np.nanmax(depth[r, c]), interpolation="nearest")
    ax[0, j].set_title(name)
    err = np.where(valid, (d - depth) * 1000, np.nan)[r, c]
    im = ax[1, j].imshow(err, cmap="coolwarm", vmin=-10, vmax=10, interpolation="nearest")
    # front view of the 3D points in the close-up: world x (left-right) vs z (height)
    pts = camera_to_world(backproject(d, K), extrinsic)[r, c][valid[r, c]][::3] * 1000
    ax[2, j].scatter(pts[:, 0], pts[:, 2], s=0.2, c="k")
    ax[2, j].set_ylim(-20, 60)
    ax[2, j].set_xlim(*x_range)
    ax[2, j].set_xlabel("world x [mm]")
    for a in ax[:2, j]:
        a.axis("off")
ax[1, 0].set_ylabel("error [mm]")
ax[2, 0].set_ylabel("height z [mm]")
fig.colorbar(im, ax=ax[1, :].tolist(), label="error added [mm] (clipped at +/-10)", shrink=0.8)
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/check_noise.png", dpi=110, bbox_inches="tight")
