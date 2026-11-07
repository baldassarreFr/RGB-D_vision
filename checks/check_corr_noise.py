"""
Step 2 check: correlated noise keeps the noise size of Step 1 and has the requested patch size.

Usage: python checks/check_corr_noise.py [scene folder]   (default: outputs/dataset/scenes/000000)

Figure: outputs/check_corr_noise.png (the same close-up of the error, one panel per corr_px).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from rgbd_sim.camera_model import add_noise, noise_sigma
from rgbd_sim.scene import load_scene

folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset/scenes/000000"
scene = load_scene(folder)
depth = scene["depth"]
sigma = noise_sigma(scene, material=True)
valid = depth > 0
print(folder)

errors = {}
for corr in (0, 2, 5, 10):
    noisy = add_noise(depth, sigma, np.random.default_rng(0), corr_px=corr)
    z = np.where(valid, (noisy - depth) / np.where(valid, sigma, 1), np.nan)
    errors[corr] = z
    r = max(corr, 1)
    both_h = valid[:, :-r] & valid[:, r:]
    both_v = valid[:-r, :] & valid[r:, :]
    c_h = np.corrcoef(z[:, :-r][both_h], z[:, r:][both_h])[0, 1]
    c_v = np.corrcoef(z[:-r, :][both_v], z[r:, :][both_v])[0, 1]
    print(f"corr_px {corr:2d}: std of noise/sigma {np.nanstd(z):.3f} (expected ~1);  correlation at "
          f"{r} px: left-right {c_h:.2f}, up-down {c_v:.2f} (expected {'~0' if corr == 0 else '~0.5'})")

# 3. corr_px = 0 is Step 1's noise: same draws as add_noise without corr_px
step1 = add_noise(depth, sigma, np.random.default_rng(0))
again = add_noise(depth, sigma, np.random.default_rng(0), corr_px=0)
print(f"corr_px 0 identical to Step 1 noise: {np.array_equal(step1, again)}   (expected True)")

# Figure: the error on a close-up around the parts
rows, cols = np.where(scene["label"] >= 2)
r = slice(max(rows.min() - 40, 0), rows.max() + 40)
c = slice(max(cols.min() - 40, 0), cols.max() + 40)
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
for a, (corr, z) in zip(ax, errors.items()):
    a.imshow(z[r, c], cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    a.set_title("independent (Step 1)" if corr == 0 else f"correlated, {corr} px")
    a.axis("off")
fig.suptitle("Depth error divided by sigma (red = farther, blue = closer): same size per pixel, "
             "different patch size")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/check_corr_noise.png", dpi=120, bbox_inches="tight")
print("figure: outputs/check_corr_noise.png")
