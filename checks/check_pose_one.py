"""
Check the pose estimation of every part on the ideal depth of one scene.

Usage: python checks/check_pose_one.py [scene folder] [voxel mm] [camera variant] [RANSAC factor] [RANSAC confidence]
       (defaults: outputs/dataset/scenes/000000, 3 mm, ideal, and the values in pose.py)

For every part it prints:
  pts        points of the part seen by the camera (before / after downsampling)
  on model   median and 90th percentile distance of those points to the model placed in the
             estimated pose: about 0 if the model lies on the points
  centre     distance between estimated and true centre of the part [mm]: about 0 if right
             (the symmetries of the parts do not move the centre)
  angle      rotation between estimated and true pose [deg]: 0, or a symmetry (90, 180) if right
  time       seconds for this part (all runs)
  score      depth check of the chosen pose and of every run (1 = explains the depth perfectly)

Then runs the first part again with the same seed: same result expected (reproducible).
"""
import sys

import numpy as np
import open3d as o3d

from rgbd_sim.metrics import add_s, metric_points
from rgbd_sim.pose import (RANSAC_CONFIDENCE, RANSAC_DISTANCE_FACTOR, estimate_pose_verified,
                          model_point_cloud, part_points)
from rgbd_sim.scene import MAKE_PART, load_depth, load_scene


def distance_to_model(points, mesh, pose):
    """Distance (m) of each point (N, 3) to the surface of `mesh` placed with the 4x4 `pose`."""
    placed = mesh.copy()
    placed.apply_transform(pose)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.core.Tensor(np.asarray(placed.vertices, dtype=np.float32)),
                     o3d.core.Tensor(np.asarray(placed.faces, dtype=np.uint32)))
    return rc.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()


def rotation_angle_deg(R1, R2):
    """Angle of the rotation that takes R1 to R2."""
    cos = (np.trace(R1.T @ R2) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset/scenes/000000"
voxel = (float(sys.argv[2]) if len(sys.argv) > 2 else 3.0) / 1000
variant = sys.argv[3] if len(sys.argv) > 3 else "ideal"
ransac = {"ransac_factor": float(sys.argv[4]) if len(sys.argv) > 4 else RANSAC_DISTANCE_FACTOR,
          "confidence": float(sys.argv[5]) if len(sys.argv) > 5 else RANSAC_CONFIDENCE}
scene = load_scene(folder)
scene["depth"] = load_depth(folder, variant)     # the labels stay the ideal ones
meta = scene["meta"]
K = np.array(meta["camera"]["K"])
meshes = {name: make() for name, make in MAKE_PART.items()}
models = {name: model_point_cloud(mesh) for name, mesh in meshes.items()}

print(f"{folder}, {variant} depth, voxel {voxel * 1000:.1f} mm, RANSAC distance "
      f"{ransac['ransac_factor']}v, confidence {ransac['confidence']}")
print(f"{'id':>3s} {'class':8s} {'pts':>11s} {'on model [mm]':>15s} {'centre [mm]':>12s} "
      f"{'angle':>7s} {'time':>6s} | depth-check score: chosen, all runs")
first = None
for obj in meta["objects"]:
    points = part_points(scene["depth"], scene["label"], obj["instance_id"], K)
    if len(points) == 0:
        print(f"{obj['instance_id']:3d} {obj['class']:8s}  not visible")
        continue
    pose, info = estimate_pose_verified(scene["depth"], scene["label"], obj["instance_id"], K,
                                        meshes[obj["class"]], models[obj["class"]], voxel, seed=0, **ransac)
    true = np.array(obj["pose_camera"])
    d = distance_to_model(points, meshes[obj["class"]], pose) * 1000
    centre = np.linalg.norm(pose[:3, 3] - true[:3, 3]) * 1000
    angle = rotation_angle_deg(true[:3, :3], pose[:3, :3])
    print(f"{obj['instance_id']:3d} {obj['class']:8s} {len(points):5d}/{info['scene_points_down']:<5d}"
          f" {np.median(d):6.2f} / {np.percentile(d, 90):5.2f} {centre:12.2f} {angle:6.1f}° "
          f"{info['seconds']:5.2f}s | {info['score']:.3f}  "
          + " ".join(f"{x:.3f}" for x in info["all_scores"]))
    if first is None:
        first = (obj, pose)

obj, pose = first
pose_again, _ = estimate_pose_verified(scene["depth"], scene["label"], obj["instance_id"], K,
                                       meshes[obj["class"]], models[obj["class"]], voxel, seed=0, **ransac)
# Compared with ADD-S: a different but symmetric pose counts as the same result
print(f"\nsame seed again, first part: ADD-S between the two results "
      f"{add_s(pose_again, pose, meshes[obj['class']], metric_points(meshes[obj['class']])) * 1000:.3f} mm"
      f"   (expected ~0; Open3D's RANSAC is not bit-reproducible)")
