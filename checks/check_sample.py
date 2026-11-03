"""
Generate one complete scene, save it, load it back, and check it end to end.

Usage: python check_sample.py [seed]
Checks:
1. saving and loading change nothing (depth only by the 0.1 mm rounding);
2. the saved poses are right: every pixel of a part, taken back into that part's
   own model frame, must fall inside the part's bounding box.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from rgbd_sim.camera import backproject
from rgbd_sim.parts import make_profile, make_bracket
from rgbd_sim.scene import generate_scene, save_scene, load_scene

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
folder = f"outputs/check_sample/{seed:06d}"

# ---------- 1. Generate, save, load ----------
scene = generate_scene(seed)
save_scene(scene, folder)
loaded = load_scene(folder)
print(f"saved to {folder}: {sorted(os.listdir(folder))}")

depth_error_mm = np.abs(loaded["depth"] - scene["depth"]).max() * 1000
print(f"depth after save/load: max change {depth_error_mm:.3f} mm   (expected <= 0.05)")
print(f"rgb identical: {np.array_equal(loaded['rgb'], scene['rgb'])},  "
      f"labels identical: {np.array_equal(loaded['label'], scene['label'])}")

# ---------- 2. Pose check, using only what was saved ----------
meta = loaded["meta"]
K = np.array(meta["camera"]["K"])
points_cam = backproject(loaded["depth"], K)                 # (H, W, 3), camera frame
half_size = {"profile": make_profile().extents / 2, "bracket": make_bracket().extents / 2}
for obj in meta["objects"]:
    pts = points_cam[loaded["label"] == obj["instance_id"]]  # this part's pixels, camera frame
    pose = np.array(obj["pose_camera"])                       # model -> camera
    R, t = pose[:3, :3], pose[:3, 3]
    pts_model = ((pts - t) @ R)                               # camera -> model: R^T (p - t), row form
    outside_mm = (np.abs(pts_model) - half_size[obj["class"]]).max() * 1000
    print(f"  instance {obj['instance_id']} {obj['class']:8s} {obj['visible_pixels']:5d} px   "
          f"farthest outside its box: {max(outside_mm, 0):.3f} mm   (expected ~0)")

# ---------- 3. Figure ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].imshow(loaded["rgb"])
ax[0].set_title("RGB (shaded)")
ax[1].imshow(np.where(loaded["depth"] > 0, loaded["depth"], np.nan), cmap="viridis")
ax[1].set_title("Depth [m] (loaded from 16-bit PNG)")
ax[2].imshow(loaded["label"], cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
ax[2].set_title("Instance labels")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(f"outputs/check_sample_seed{seed}.png", dpi=120)
