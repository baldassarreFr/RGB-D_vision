"""
How wrong is an estimated pose?
Error measures between an estimated and a true pose.

Poses are 4x4 matrices model -> camera.
  add_s           decides success: mean distance from the model in the estimated pose to the
                  closest point of the model in the true pose (blind to symmetries).
  success         ADD-S below 10 % of the part's diameter.
  centre_error    distance between estimated and true centre
  rotation_error  rotation angle to the closest symmetric copy of the true pose (diagnosis)

The symmetries are known exactly because the parts are designed in parts.py
"""
import numpy as np
import open3d as o3d
import trimesh

ADD_S_THRESHOLD = 0.10      # share of the diameter (standard)
METRIC_POINTS = 2000        # model points for ADD-S, sampled once per class (our choice)


def _rot(axis, degrees):
    """3x3 rotation about a unit axis (Rodrigues)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(degrees)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


# Rotations (model frame) that map each part onto itself
SYMMETRIES = {
    # 4 turns of 90 deg about the long axis z, each also turned end over end (180 deg about x)
    "profile": [flip @ _rot([0, 0, 1], k * 90) for k in range(4)
                for flip in (np.eye(3), _rot([1, 0, 0], 180))],
    # identity, and 180 deg about the diagonal of the L: swaps the two legs, reverses the width
    "bracket": [np.eye(3), _rot([1, 1, 0], 180)],
}


def diameter(mesh):
    """
    Largest distance between two points of the part (m): between two of its vertices.

    All vertex pairs are compared.
    """
    v = np.asarray(mesh.vertices)
    return np.max(np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1))


def metric_points(mesh, seed=0):
    """Points spread over the model surface (model frame), used by add_s."""
    points, _ = trimesh.sample.sample_surface(mesh, METRIC_POINTS, seed=seed)
    return points


def _transform(pose, points):
    return points @ pose[:3, :3].T + pose[:3, 3]


def distance_to_surface(points, mesh, pose):
    """Distance (m) of each point (N, 3) to the surface of `mesh` placed with `pose`."""
    placed = mesh.copy()
    placed.apply_transform(pose)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.core.Tensor(np.asarray(placed.vertices, dtype=np.float32)),
                     o3d.core.Tensor(np.asarray(placed.faces, dtype=np.uint32)))
    return rc.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()


def add_s(pose_est, pose_true, mesh, points):
    """ADD-S (m): model points in the estimated pose -> closest point of the model in the true pose."""
    return float(np.mean(distance_to_surface(_transform(pose_est, points), mesh, pose_true)))


def success(add_s_value, part_diameter):
    return add_s_value < ADD_S_THRESHOLD * part_diameter


def centre_error(pose_est, pose_true):
    """Distance (m) between the estimated and the true centre of the part."""
    return float(np.linalg.norm(pose_est[:3, 3] - pose_true[:3, 3]))


def rotation_angle(R1, R2):
    """Angle (deg) of the rotation that takes R1 to R2."""
    cos = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def rotation_error(pose_est, pose_true, part_class):
    """Angle (deg) between the estimated rotation and the closest symmetric copy of the true one."""
    return min(rotation_angle(pose_true[:3, :3] @ S, pose_est[:3, :3])
               for S in SYMMETRIES[part_class])
