"""
Diagnosis of the coarse step (FPFH + RANSAC) on one scene.

Usage: python checks/check_ransac.py [scene folder] [voxel mm] [camera variant]
       (defaults: outputs/dataset/scenes/000000, 3 mm, ideal)

If wrong poses get as many inliers as the true pose, the criterion cannot tell them apart.
"""
import sys

import numpy as np
import open3d as o3d

from rgbd_sim.pose import (HYPOTHESES, RANSAC_DISTANCE_FACTOR, _downsample_with_features,
                          estimate_pose, model_point_cloud, part_points)
from rgbd_sim.scene import MAKE_PART, load_depth, load_scene


def inlier_share(scene_down, model_down, pose, distance):
    """Share of scene points that land within `distance` of the model under pose (model->camera)."""
    moved = o3d.geometry.PointCloud(scene_down)
    moved.transform(np.linalg.inv(pose))                      # scene -> model frame
    d = np.asarray(moved.compute_point_cloud_distance(model_down))
    return np.mean(d < distance)


folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/dataset/scenes/000000"
voxel = (float(sys.argv[2]) if len(sys.argv) > 2 else 3.0) / 1000
variant = sys.argv[3] if len(sys.argv) > 3 else "ideal"
scene = load_scene(folder)
depth = load_depth(folder, variant)
meta = scene["meta"]
K = np.array(meta["camera"]["K"])
models = {name: model_point_cloud(make()) for name, make in MAKE_PART.items()}
dist = RANSAC_DISTANCE_FACTOR * voxel

print(f"{folder}, {variant} depth, voxel {voxel * 1000:.1f} mm, RANSAC inlier distance {dist * 1000:.1f} mm")
print(f"{'id':>3s} {'class':8s} {'pairs':>6s} {'right':>6s} | {'inliers true pose':>17s} | "
      f"RANSAC runs: inliers (centre error mm)")
for obj in meta["objects"]:
    points = part_points(depth, scene["label"], obj["instance_id"], K)
    if len(points) == 0:
        continue
    true = np.array(obj["pose_camera"])
    scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    model_down, model_fpfh = _downsample_with_features(models[obj["class"]], voxel, estimate_normals=False)
    scene_down, scene_fpfh = _downsample_with_features(scene_pcd, voxel, estimate_normals=True)

    # 1. FPFH pairs: most similar model histogram for every scene point
    tree = o3d.geometry.KDTreeFlann(model_fpfh)
    scene_in_model = np.asarray(scene_down.points) @ true[:3, :3] - true[:3, :3].T @ true[:3, 3]
    model_pts = np.asarray(model_down.points)
    right = 0
    for i in range(len(scene_down.points)):
        _, idx, _ = tree.search_knn_vector_xd(scene_fpfh.data[:, i], 1)
        right += np.linalg.norm(model_pts[idx[0]] - scene_in_model[i]) < 2 * voxel
    n_pairs = len(scene_down.points)

    # 2. Inliers of the true pose vs of the poses RANSAC returned
    runs = []
    for i in range(HYPOTHESES):
        _, info = estimate_pose(points, models[obj["class"]], voxel, seed=i)
        coarse = info["coarse_pose"]
        runs.append(f"{inlier_share(scene_down, model_down, coarse, dist):.2f} "
                    f"({np.linalg.norm(coarse[:3, 3] - true[:3, 3]) * 1000:4.1f})")
    print(f"{obj['instance_id']:3d} {obj['class']:8s} {n_pairs:6d} {100 * right / n_pairs:5.1f}% | "
          f"{inlier_share(scene_down, model_down, true, dist):17.2f} | " + "  ".join(runs))
