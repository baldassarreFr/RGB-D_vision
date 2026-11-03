"""
Drop a few parts on the table with PyBullet, then look at them with the camera.

Usage: python check_drop.py [seed]     (different seed -> different scene)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from camera import W, H, intrinsics, look_at
from parts import make_profile, make_bracket
from physics import drop_parts
from render import render_ideal

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Choose how many parts to drop
rng = np.random.default_rng(seed)
n_profiles = int(rng.integers(1, 3))      # 1 or 2 profiles  (upper bound excluded)
n_brackets = int(rng.integers(1, 5))      # 1 to 4 brackets
names = ["profile"] * n_profiles + ["bracket"] * n_brackets
meshes = [make_profile() if n == "profile" else make_bracket() for n in names]
poses, info = drop_parts(names, meshes, seed=seed)
print(f"seed {seed}: settled = {info['settled']} after {info['simulated_seconds']:.1f} s, "
      f"deepest overlap = {info['deepest_overlap_mm']:.2f} mm")

placed = []
for name, mesh, pose in zip(names, meshes, poses):
    m = mesh.copy()
    m.apply_transform(pose)
    placed.append(m)
    x, y, z = pose[:3, 3] * 1000
    print(f"  {name:8s} centre = ({x:+5.0f}, {y:+5.0f}, {z:+5.1f}) mm   "
          f"lowest point z = {m.bounds[0, 2] * 1000:+.2f} mm")

# Render: 1 = table, 2.. = the parts in the order above
K = intrinsics()
extrinsic = look_at(eye=(0.0, -0.35, 0.55), target=(0, 0, 0))
table = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
table.apply_translation([0, 0, -0.01])
depth, label = render_ideal([table] + placed, K, extrinsic, W, H)
for i, name in enumerate(names):
    print(f"  label {i + 2} ({name}): {(label == i + 2).sum()} visible pixels")

# Figure: close-up of the area with the parts
rows, cols = np.where(label >= 2)
r = slice(max(rows.min() - 20, 0), rows.max() + 20)
c = slice(max(cols.min() - 20, 0), cols.max() + 20)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
im = ax[0].imshow(np.where(depth > 0, depth, np.nan)[r, c], cmap="viridis", interpolation="nearest")
fig.colorbar(im, ax=ax[0])
ax[0].set_title(f"Ideal depth [m], seed {seed}")
ax[1].imshow(label[r, c], cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
ax[1].set_title(f"Labels: 1 table, then {n_profiles} profile(s), {n_brackets} bracket(s)")
for a in ax:
    a.axis("off")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig(f"outputs/drop_seed{seed}.png", dpi=120)
