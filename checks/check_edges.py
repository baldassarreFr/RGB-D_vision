"""Check the edge effect on one scene.

Usage: python checks/check_edges.py [scene folder]   (default: outputs/dataset/scenes/000000)
Checks:
1. about half of the pixels on depth edges change (EDGE_SHIFT_PROBABILITY = 0.5);
2. every changed pixel takes exactly the depth of a direct neighbour (edge moves by <= 1 px,
   no new depth values are invented);
3. objects grow and shrink about equally often.
Figure: outputs/check_edges.png (close-up; blue = object grew by a pixel, orange = shrank).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from rgbd_sim.camera_model import shift_edges
from rgbd_sim.scene import load_scene

folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset/scenes/000000"
scene = load_scene(folder)
depth, label = scene["depth"], scene["label"]
shifted, moved = shift_edges(depth, label, np.random.default_rng(0))
print(f"{folder}")

# 1. How many pixels moved
changed = shifted != depth
print(f"pixels changed: {changed.sum()} ({100 * changed.sum() / (depth > 0).sum():.2f} % of the image)")

# 2. Each new value must be the depth of one of the 4 direct neighbours
neighbours = [np.roll(depth, shift=s, axis=a) for s, a in ((1, 0), (-1, 0), (1, 1), (-1, 1))]
from_neighbour = np.zeros_like(changed)
for n in neighbours:
    from_neighbour |= changed & (shifted == n)
print(f"changed pixels whose new depth is a direct neighbour's: "
      f"{100 * from_neighbour[changed].mean():.1f} %   (expected 100 %)")

# 3. Grow vs shrink: a pixel getting closer = the nearer object grew over it
grew = changed & (shifted < depth)
shrank = changed & (shifted > depth)
print(f"object grew: {grew.sum()} px, object shrank: {shrank.sum()} px   (expected about equal)")
print(f"changed pixels: {(changed & (label >= 2)).sum()} on the parts, "
      f"{(changed & (label == 1)).sum()} on the table")

# Figure
valid = depth > 0
grey = 0.4 + 0.6 * (depth - depth[valid].min()) / np.ptp(depth[valid])
rgb = np.dstack([np.clip(np.where(valid, grey, 1.0), 0, 1)] * 3)
rgb[grew] = [0.1, 0.3, 1.0]
rgb[shrank] = [1.0, 0.55, 0.0]
rows, cols = np.where(label >= 2)
r = slice(max(rows.min() - 15, 0), rows.max() + 15)
c = slice(max(cols.min() - 15, 0), cols.max() + 15)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(rgb[r, c], interpolation="nearest")
ax.set_title("Edges moved by one pixel: blue = object grew, orange = object shrank")
ax.axis("off")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/check_edges.png", dpi=120)
