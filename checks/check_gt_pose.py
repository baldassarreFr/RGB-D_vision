"""
The true poses in meta.json put each part model exactly onto its pixels.

Usage: python checks/check_gt_pose.py [dataset folder]   (default: outputs/dataset)
For every part of every scene:
  1. take the part's 3D points from the IDEAL depth (pixels with its label), camera frame;
  2. put the part model in the camera frame with pose_camera (model -> camera);
  3. distance of every point to the model surface.
Expected: at most ~0.05 mm (the depth is stored in steps of 0.1 mm), for every part.
Control: the same with the model moved by 2 mm; the distances must become clearly larger,
otherwise the check could not detect a wrong pose.
"""
import os
import sys

import numpy as np
import open3d as o3d

from rgbd_sim.camera import backproject
from rgbd_sim.scene import MAKE_PART, load_scene


def distance_to_model(points, mesh, pose):
    """Distance (m) of each point (N, 3) to the surface of `mesh` placed with the 4x4 `pose`."""
    placed = mesh.copy()
    placed.apply_transform(pose)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(placed.vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(placed.faces, dtype=np.uint32)))
    return scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()


dataset = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset"
scene_dirs = sorted(os.path.join(dataset, "scenes", d) for d in os.listdir(os.path.join(dataset, "scenes")))
models = {name: make() for name, make in MAKE_PART.items()}     # model frame, metres

shift = np.eye(4)
shift[2, 3] = 0.002     # control: 2 mm along the viewing axis (a sideways shift can slide along a flat face unnoticed)
worst, worst_control = 0.0, np.inf
n_parts = 0
for folder in scene_dirs:
    scene = load_scene(folder)
    meta = scene["meta"]
    points_all = backproject(scene["depth"], np.array(meta["camera"]["K"]))   # (H, W, 3)
    for obj in meta["objects"]:
        mask = scene["label"] == obj["instance_id"]
        if not mask.any():
            continue                                            # part completely hidden
        points = points_all[mask]
        pose = np.array(obj["pose_camera"])
        d = distance_to_model(points, models[obj["class"]], pose) * 1000           # mm
        d_control = distance_to_model(points, models[obj["class"]], shift @ pose) * 1000
        worst = max(worst, d.max())
        worst_control = min(worst_control, np.median(d_control))
        n_parts += 1
        if d.max() > 0.1:
            print(f"  ! {folder} instance {obj['instance_id']} ({obj['class']}): "
                  f"max distance {d.max():.3f} mm")
    print(f"{folder}: {len(meta['objects'])} parts checked")

print(f"\nparts checked: {n_parts}")
print(f"true pose:   largest point-to-model distance {worst:.3f} mm   (expected <= ~0.05 mm)")
print(f"control (model moved by 2 mm): smallest median distance {worst_control:.3f} mm   "
      f"(expected clearly > 0.05 mm)")
