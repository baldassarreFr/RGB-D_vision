"""
Pose estimation of one part performed in three steps:
    - FPFH is used as a feature descriptor to find matching points between the image and the CAD model's
    - RANSAC is used to find a coarse pose that aligns the part's points with the model's points
    - ICP is used to refine the pose by minimizing the distance between the part's points and the model's points

  These steps are repeated with different random seeds (HYPOTHESES).
  Each resulting pose is checked by rendering the model in that pose and comparing its depth with the measured depth (tolerance d).
  The pose that explains it best is kept.

Two scales, tuned in the experiment:
  v  voxel size = size of shape details.
  d  tolerance = how far a measured point may lie from the true surface: tight ICP pass and
     depth check.


Poses are 4x4 matrices model -> camera.
"""
import time

import numpy as np
import open3d as o3d
import trimesh

from rgbd_sim.camera import backproject
from rgbd_sim.render import render_ideal


NORMAL_RADIUS_FACTOR = 2.0      # neighbourhood for estimating normals
FEATURE_RADIUS_FACTOR = 5.0     # neighbourhood for FPFH
RANSAC_DISTANCE_FACTOR = 1.0    # a scene point counts as matched if this close to the model

ICP_DISTANCE_FACTOR = 0.4       # final ICP pairs only points closer than this
ICP_WIDE_FACTOR = 1.5           # a first ICP pass
ICP_MAX_ITERATIONS = 50         # Open3D default is 30
RANSAC_MAX_ITERATIONS = 100000  
RANSAC_CONFIDENCE = 0.999       
EDGE_LENGTH_CHECK = 0.9         # reject 3-point samples whose shape differs (>10 %)
HYPOTHESES = 5                  # RANSAC+ICP runs per part (different seeds)

MODEL_POINTS = 50000


def model_point_cloud(mesh, seed=0):
    """Points on the whole surface of the CAD model (model frame), with exact outward normals."""
    points, faces = trimesh.sample.sample_surface(mesh, MODEL_POINTS, seed=seed)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(mesh.face_normals[faces])
    return pcd


def part_points(depth, label, instance_id, K):
    """(N, 3) camera-frame points of one part: pixels with its label that still have depth."""
    mask = (label == instance_id) & (depth > 0)
    return backproject(depth, K)[mask]


def _downsample_with_features(pcd, voxel, estimate_normals):
    """Voxel-downsample, (re)compute normals if needed, compute FPFH features."""
    down = pcd.voxel_down_sample(voxel)
    if estimate_normals:
        # scene: normals from the neighbours, turned towards the camera (at the origin),
        # i.e. outwards for every surface the camera can see
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS_FACTOR * voxel, max_nn=30))
        down.orient_normals_towards_camera_location(np.zeros(3))
    else:
        down.normalize_normals()      # model: exact normals, averaged inside each voxel
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=FEATURE_RADIUS_FACTOR * voxel, max_nn=100))
    return down, fpfh


_model_cache = {}


def _model_features(model_pcd, voxel):
    """Downsampled model + FPFH, computed once per (model, voxel) and then reused."""
    key = (id(model_pcd), voxel)
    if key not in _model_cache:
        _model_cache[key] = _downsample_with_features(model_pcd, voxel, estimate_normals=False)
    return _model_cache[key]


def estimate_pose(points, model_pcd, voxel, seed=0, tolerance=None,
                  ransac_factor=RANSAC_DISTANCE_FACTOR, confidence=RANSAC_CONFIDENCE):
    """Pose (4x4, model -> camera) of a part, from its camera-frame points (N, 3).

    Two scales:  voxel      = size of shape details (voxel, normals, FPFH, RANSAC, wide ICP)
                 tolerance  = how far a measured point may lie from the true surface (final
                              ICP); default 0.4 * voxel as in the Open3D tutorial

    Every step moves the scene points onto the model (scene -> model): every scene point has a
    true partner on the model, while most of the model (its back) has none in the scene. The
    result is inverted at the end.
    returns pose, info  (info: time, fitness, coarse pose, number of points)
    """
    t0 = time.perf_counter()
    o3d.utility.random.seed(seed)
    scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    reg = o3d.pipelines.registration

    # FPFH + RANSAC
    model_down, model_fpfh = _model_features(model_pcd, voxel)
    scene_down, scene_fpfh = _downsample_with_features(scene_pcd, voxel, estimate_normals=True)
    dist = ransac_factor * voxel
    coarse = reg.registration_ransac_based_on_feature_matching(
        scene_down, model_down, scene_fpfh, model_fpfh,
        False,          # no mutual filter: a partial view and a full model rarely choose each other
        dist,
        reg.TransformationEstimationPointToPoint(False),
        3,              # 3 pairs per attempt
        [reg.CorrespondenceCheckerBasedOnEdgeLength(EDGE_LENGTH_CHECK),
         reg.CorrespondenceCheckerBasedOnDistance(dist)],
        reg.RANSACConvergenceCriteria(RANSAC_MAX_ITERATIONS, confidence))

    # ICP refinement
    wide = reg.registration_icp(scene_pcd, model_pcd, ICP_WIDE_FACTOR * voxel, coarse.transformation,
                                reg.TransformationEstimationPointToPlane(),
                                reg.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERATIONS))
    tolerance = ICP_DISTANCE_FACTOR * voxel if tolerance is None else tolerance
    fine = reg.registration_icp(scene_pcd, model_pcd, tolerance, wide.transformation,
                                reg.TransformationEstimationPointToPlane(),
                                reg.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERATIONS))
    pose = np.linalg.inv(fine.transformation)

    info = {"seconds": time.perf_counter() - t0,
            "coarse_pose": np.linalg.inv(coarse.transformation),  # after RANSAC only
            "ransac_fitness": coarse.fitness,   # share of scene points (downsampled) on the model
            "icp_fitness": fine.fitness,        # share of scene points within tolerance of the model
            "scene_points": len(points),
            "scene_points_down": len(scene_down.points),
            "model_points_down": len(model_down.points)}
    return pose, info


def depth_check_score(pose, mesh, depth, mask, K, tolerance):
    """How well a pose explains the measured depth of a part (0-1, 1 = perfectly).

    Renders the model alone in `pose` (camera frame) and compares, pixel by pixel:
      explained:    pixel of the part where the model is within `tolerance` of the measured depth
      contradicted: pixel outside the part where the model would be in front of the measured
                    surface (then the camera would have seen the model there, and it did not)
    Pixels where the model would be hidden behind something else are not counted (occlusion).
    score = explained / (pixels of the part + contradicted)
    """
    placed = mesh.copy()
    placed.apply_transform(pose)
    h, w = depth.shape
    rendered, _ = render_ideal([placed], K, np.eye(4), w, h)   # camera frame = world frame here
    model_there = rendered > 0
    explained = mask & model_there & (np.abs(rendered - depth) < tolerance)
    contradicted = ~mask & model_there & (depth > 0) & (rendered < depth - tolerance)
    return explained.sum() / (mask.sum() + contradicted.sum())


def estimate_pose_verified(depth, label, instance_id, K, mesh, model_pcd, voxel, tolerance=None,
                           seed=0, **ransac):
    """Pose of one part (4x4, model -> camera): best of HYPOTHESES runs, by depth_check_score.

    tolerance: used by the final ICP and by the depth check (default 0.4 * voxel)

    ransac: optional ransac_factor=..., confidence=... passed to estimate_pose (for tests)
    returns pose, info  (info of the chosen run, plus all scores and the total time)
    """
    t0 = time.perf_counter()
    tolerance = ICP_DISTANCE_FACTOR * voxel if tolerance is None else tolerance
    mask = (label == instance_id) & (depth > 0)
    points = backproject(depth, K)[mask]
    best = None
    scores = []
    for i in range(HYPOTHESES):
        pose, info = estimate_pose(points, model_pcd, voxel, seed=seed * HYPOTHESES + i,
                                   tolerance=tolerance, **ransac)
        score = depth_check_score(pose, mesh, depth, mask, K, tolerance)
        scores.append(score)
        if best is None or score > best[2]:
            best = (pose, info, score)
    pose, info, score = best
    info = dict(info, score=score, all_scores=scores, seconds=time.perf_counter() - t0)
    return pose, info
