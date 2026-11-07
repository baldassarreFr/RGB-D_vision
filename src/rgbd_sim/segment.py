"""
Part/table segmentation: which pixels show a part and which the table?

Method:
  1. fit the table plane with RANSAC (Open3D segment_plane) on the measured points;
  2. every point higher than t above that plane is "part", everything else "table".
One parameter, t (metres): used both as RANSAC's inlier distance and as the height threshold.

Touching parts are not separated: this is part/table segmentation, not instance segmentation.
"""
import numpy as np
import open3d as o3d

from rgbd_sim.camera import backproject

PLANE_SAMPLE = 20000        # points used to fit the plane (random subset: faster), our choice
PLANE_ITERATIONS = 1000     # RANSAC iterations (Open3D tutorial value)


def fit_table_plane(points, t, seed=0):
    """Plane (unit normal n, offset d) with n.p + d = height above the table, n towards the camera."""
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(PLANE_SAMPLE, len(points)), replace=False)]
    o3d.utility.random.seed(seed)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample))
    (a, b, c, d), _ = pcd.segment_plane(distance_threshold=t, ransac_n=3, num_iterations=PLANE_ITERATIONS)
    n = np.array([a, b, c])
    norm = np.linalg.norm(n)
    n, d = n / norm, d / norm
    if d < 0:                      # camera (origin) must be on the positive side: height > 0 above
        n, d = -n, -d
    return n, d


def segment_parts(depth, K, t, seed=0):
    """(H, W) bool: pixels whose point is more than t above the fitted table plane.

    Pixels without depth are never "part". returns mask, (n, d)
    """
    valid = depth > 0
    points = backproject(depth, K)                 # (H, W, 3), camera frame, NaN where no depth
    n, d = fit_table_plane(points[valid], t, seed)
    height = np.where(valid, points @ n + d, -np.inf)
    return height > t, (n, d)


def true_table_plane(meta):
    """The real table plane (top face at world z = 0) in the camera frame, as (n, d)."""
    E = np.array(meta["camera"]["extrinsic_world_to_camera"])
    R, tr = E[:3, :3], E[:3, 3]
    n = R[:, 2]                    # world z axis (up) in camera coordinates
    return n, -n @ tr


def evaluate(mask, label, meta, found_share=0.5):
    """Compare a part mask with the true labels.

    Per part: share of its pixels marked "part" (recall); found if >= found_share.
    Pixel precision: share of the pixels marked "part" that really belong to a part.
    returns {"parts": [(instance_id, class, recall, found), ...], "precision": p,
             "found_rate": share of parts found, "f1": harmonic mean of found_rate and precision}
    """
    is_part = label >= 2
    parts = []
    for obj in meta["objects"]:
        gt = label == obj["instance_id"]
        if not gt.any():
            continue                                  # completely hidden part
        recall = float((mask & gt).sum() / gt.sum())
        parts.append((obj["instance_id"], obj["class"], recall, recall >= found_share))
    precision = float((mask & is_part).sum() / mask.sum()) if mask.any() else 0.0
    found_rate = float(np.mean([p[3] for p in parts])) if parts else float("nan")
    f1 = 0.0 if found_rate + precision == 0 else 2 * found_rate * precision / (found_rate + precision)
    return {"parts": parts, "precision": precision, "found_rate": found_rate, "f1": f1}
