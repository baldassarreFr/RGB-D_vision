"""Camera model shared by all scripts: D435-like depth intrinsics, camera pose,
and conversion between depth pixels and 3D points.

Conventions:
- World frame: table top is the plane z = 0, z points up. Units: metres.
- Camera frame (OpenCV): x right, y down, z forward (out of the lens).
- Each pixel (col, row) has its centre at (col + 0.5, row + 0.5), as in Open3D's ray caster.
- Depth images store z-depth in metres; 0 means "no data".
"""

import numpy as np

W, H = 848, 480          # D435 depth stream resolution used in this project

# Horizontal depth field of view of the D435i (https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html)
HFOV_DEG = 87.0


def intrinsics(width=W, height=H, hfov_deg=HFOV_DEG):
    """3x3 pinhole matrix K: square pixels, principal point at the image centre."""
    fx = (width / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    return np.array([[fx, 0.0, width / 2],
                     [0.0, fx, height / 2],
                     [0.0, 0.0, 1.0]])


def look_at(eye, target, up=(0, 0, 1)):
    """World->camera 4x4 matrix ("extrinsic") for a camera at `eye` looking at `target`."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)          # forward
    x = np.cross(z, up)
    x /= np.linalg.norm(x)          # right
    y = np.cross(z, x)              # down
    R = np.stack([x, y, z])         # rows = camera axes in world coordinates
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return T


def backproject(depth, K):
    """Depth image (H, W) -> 3D points (H, W, 3) in CAMERA coordinates.

    Inverts the pinhole projection u = fx * X/Z + cx, v = fy * Y/Z + cy.
    Pixels without data (depth 0) become NaN.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)   # pixel centres
    z = np.where(depth > 0, depth, np.nan)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


def camera_to_world(points_cam, extrinsic):
    """Points (..., 3) in camera coordinates -> world coordinates.

    The extrinsic maps world to camera: p_cam = R @ p_world + t,
    so p_world = R^T @ (p_cam - t).
    """
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    p = (points_cam - t).reshape(-1, 3).T        # (3, N): one point per column
    p_world = R.T @ p                            # the textbook formula
    return p_world.T.reshape(points_cam.shape)   # back to the input shape, e.g. (H, W, 3)
