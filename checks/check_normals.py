"""
Check the saved normals and the incidence angles they give (used by the Step 2 holes).

Usage: python checks/check_normals.py [scene folder]    (default: outputs/dataset/scenes/000000)
Run scripts/generate_dataset.py first (the scenes must contain normals.npz).

Checks:
1. the table's normals point straight up (0, 0, 1);
2. all normals have length 1;
3. incidence angle on the table: ~4 deg at the bottom of the image (seen almost from
   above) up to ~60 deg at the far edge; 32.5 deg where the optical axis meets the table.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from rgbd_sim.camera import incidence_angle
from rgbd_sim.scene import load_scene

folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset/scenes/000000"
scene = load_scene(folder)
meta, label, normals, depth = scene["meta"], scene["label"], scene["normals"], scene["depth"]
K = np.array(meta["camera"]["K"])
extrinsic = np.array(meta["camera"]["extrinsic_world_to_camera"])
print(f"{folder}: normals {normals.shape} {normals.dtype}")

# ---------- 1-2. The normals themselves ----------
table = label == meta["table_instance_id"]
deviation = np.abs(normals[table] - [0, 0, 1]).max()
lengths = np.linalg.norm(normals[label > 0], axis=-1)
print(f"table normals: max difference from (0,0,1) = {deviation:.4f}   (expected ~0)")
print(f"normal lengths: {lengths.min():.4f} .. {lengths.max():.4f}   (expected ~1)")

# ---------- 3. Incidence angles ----------
angle = np.degrees(incidence_angle(depth, normals, K, extrinsic))
print(f"table: incidence angle {np.nanmin(angle[table]):.1f} .. {np.nanmax(angle[table]):.1f} deg"
      f"   (expected ~4 .. ~60)")
H, W = depth.shape
r, c = H // 2, W // 2                                   # optical axis hits the table centre
what = "the table" if table[r, c] else "a part"
print(f"image centre ({r}, {c}) is on {what}: angle {angle[r, c]:.1f} deg"
      f"   (expected ~32.5 if it is the table)")

parts = label >= 2
a = angle[parts]
bins = [0, 15, 30, 45, 60, 75, 90.01]
counts, _ = np.histogram(a, bins=bins)
print("aluminium pixels by incidence angle:")
for lo, hi, n in zip(bins[:-1], bins[1:], counts):
    print(f"  {lo:2.0f}-{min(hi, 90):2.0f} deg: {100 * n / len(a):5.1f} %")

# ---------- Figure: incidence angle of every pixel ----------
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(angle, cmap="magma", vmin=0, vmax=90, interpolation="nearest")
fig.colorbar(im, ax=ax, label="incidence angle [deg]")
ax.set_title("Incidence angle (0 = seen face-on, 90 = seen edge-on)")
ax.axis("off")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/check_normals.png", dpi=110)
