"""Check the depth -> 3D conversion: rebuild the 3D points from the ideal depth image
and measure sizes we already know.

Run depth_first.py first: it writes depth_ideal.npy and label.npy.
Ground truth: table top at z = 0; bar 200 x 40 x 40 mm, centred at the origin.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from rgbd_sim.camera import intrinsics, look_at, backproject, camera_to_world

# Same camera as depth_first.py
K = intrinsics()
extrinsic = look_at(eye=(0.0, -0.35, 0.55), target=(0, 0, 0))


# Load the images and convert every pixel to a 3D point
depth = np.load("depth_ideal.npy")
label = np.load("label.npy")
pts_cam = backproject(depth, K)                 # (H, W, 3), camera frame
pts_world = camera_to_world(pts_cam, extrinsic) # (H, W, 3), world frame
valid = depth > 0

table = pts_world[valid & (label == 1)]         # (N, 3) points on the table
bar = pts_world[valid & (label == 2)]           # (M, 3) points on the bar


#Measurements, in millimetres
mm = 1000.0
print(f"table: {len(table)} points, z = {table[:, 2].mean() * mm:+.3f} mm "
      f"(max |z| {np.abs(table[:, 2]).max() * mm:.3f} mm)            expected 0")
print(f"bar:   {len(bar)} points, top z = {bar[:, 2].max() * mm:.3f} mm"
      f"                         expected 40")
print(f"bar x extent: {bar[:, 0].min() * mm:+.1f} .. {bar[:, 0].max() * mm:+.1f} mm"
      f"                      expected -100 .. +100")
print(f"bar y extent: {bar[:, 1].min() * mm:+.1f} .. {bar[:, 1].max() * mm:+.1f} mm"
      f"                        expected -20 .. +20")


# Top-down view of the point cloud, coloured by height
os.makedirs("outputs", exist_ok=True)
pts = pts_world[valid][::4]                     # every 4th point is enough to see the shape
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(pts[:, 0] * mm, pts[:, 1] * mm, c=pts[:, 2] * mm, s=0.5, cmap="viridis")
fig.colorbar(sc, ax=ax, label="height z [mm]")
ax.set_xlabel("world x [mm]")
ax.set_ylabel("world y [mm]")
ax.set_title("Point cloud from the ideal depth image (top view)")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("outputs/backprojection_topdown.png", dpi=120)
