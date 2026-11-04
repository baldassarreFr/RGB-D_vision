"""D435i camera model: turns the ideal depth of a scene into what the camera would measure.

Each effect is one function. Every number comes from d435.py (see docs/camera_parameters.md);
nothing numeric is written here.
Effects so far:
  - noise: per-pixel random error along the viewing axis, whose size depends on distance
    and on the material (switch `material`).
  - stereo shadows: pixels whose 3D point the right imager cannot see get no depth
    (this also produces the invalid band on the left of the image).
  - edges: object outlines move by up to 1 pixel, randomly towards either side.
(An incidence-angle effect was tried and left out: see docs/camera_parameters.md, section 2.)
"""
import numpy as np

from rgbd_sim import d435
from rgbd_sim.camera import backproject, camera_to_world
from rgbd_sim.render import first_hit_distance
from rgbd_sim.scene import scene_meshes

# Numerical tolerance, not a camera parameter: a point counts as seen by the right imager if
# nothing lies more than 1 mm in front of it along the ray (the stored depth is rounded to 0.1 mm).
VISIBILITY_TOLERANCE_M = 0.001



def material_factor_image(label, meta):
    """(H, W) noise factor of each pixel's material: 1 for the matte table, 2.9 for aluminium."""
    factor = np.ones(label.shape)
    factor[label == meta["table_instance_id"]] = d435.MATERIAL_NOISE_FACTOR[meta["table_material"]]
    for obj in meta["objects"]:
        factor[label == obj["instance_id"]] = d435.MATERIAL_NOISE_FACTOR[obj["material"]]
    return factor


def noise_sigma(scene, material=True):
    """(H, W) standard deviation of the depth noise of every pixel, in metres (0 = no depth).

    sigma = base(z)               Intel formula, matte surface seen face-on
            x material factor     if material: 2.9 on aluminium (Halmetschlager-Funek)
    """
    depth, meta = scene["depth"], scene["meta"]
    K = np.array(meta["camera"]["K"])
    valid = depth > 0

    sigma = d435.base_noise_m(depth, K[0, 0])
    if material:
        sigma = sigma * material_factor_image(scene["label"], meta)
    return np.where(valid, sigma, 0.0)


def add_noise(depth, sigma, rng):
    """Move every valid pixel along the viewing axis by a random amount of size sigma.

    Independent Gaussian noise per pixel (spatial correlation is left out: its size is not
    published for this camera). Pixels without depth stay 0.
    """
    noisy = depth + rng.standard_normal(depth.shape) * sigma
    return np.where(depth > 0, noisy, 0.0).astype(np.float32)


def stereo_shadow_mask(scene):
    """(H, W) bool: True where the left imager sees a surface but the right imager does not.

    The depth image belongs to the left imager; the right imager sits BASELINE_M to its right
    (camera +x). A pixel can only be matched if its 3D point is also visible from the right:
    1. inside the right imager's field of view (otherwise: invalid band on the left of the image);
    2. not hidden behind something closer (otherwise: stereo shadow beside an object).
    """
    depth, meta = scene["depth"], scene["meta"]
    K = np.array(meta["camera"]["K"])
    extrinsic = np.array(meta["camera"]["extrinsic_world_to_camera"])
    h, w = depth.shape
    valid = depth > 0

    # 3D points of the valid pixels, in the left camera frame and in the right camera frame
    pts_left = backproject(depth, K)[valid]                 # (N, 3)
    pts_right = pts_left - np.array([d435.BASELINE_M, 0.0, 0.0])

    # 1. Field of view: project into the right image (same intrinsics: identical imagers)
    u = K[0, 0] * pts_right[:, 0] / pts_right[:, 2] + K[0, 2]
    v = K[1, 1] * pts_right[:, 1] / pts_right[:, 2] + K[1, 2]
    in_view = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # 2. Occlusion: cast a ray from the right imager's centre towards each point
    world_pts = camera_to_world(pts_left, extrinsic)
    right_centre = camera_to_world(np.array([d435.BASELINE_M, 0.0, 0.0]), extrinsic)
    to_point = world_pts - right_centre
    distance = np.linalg.norm(to_point, axis=1)
    directions = to_point / distance[:, None]
    origins = np.broadcast_to(right_centre, directions.shape)
    hit = first_hit_distance(scene_meshes(meta), origins, directions)
    not_hidden = hit >= distance - VISIBILITY_TOLERANCE_M

    shadow = np.zeros((h, w), dtype=bool)
    shadow[valid] = ~(in_view & not_hidden)
    return shadow


def shift_edges(depth, label, rng):
    """Move the outline of every object by up to EDGE_SHIFT_PX pixel, randomly in or out.

    A pixel is "on an edge" if a neighbour EDGE_SHIFT_PX away (left, right, up or down) belongs
    to another object (labels: table, or another part). No numerical threshold is needed.
    With probability EDGE_SHIFT_PROBABILITY the pixel takes that neighbour's depth, so the
    outline moves by one pixel: outwards or inwards. No new depths are created (no blending),
    pixels without depth are left alone, labels are not changed.
    Simplification: edges inside one object (rim of a slot, one leg of a bracket in front of the
    other) are not moved.
    """
    k = d435.EDGE_SHIFT_PX
    valid = depth > 0
    across = np.zeros_like(depth)                 # depth found across an edge (0 = no edge)
    for dy, dx in ((0, k), (0, -k), (k, 0), (-k, 0)):
        # value of the neighbour at (row + dy, col + dx); np.roll wraps around the image
        # border, so the outermost rows and columns are excluded below
        n_depth = np.roll(depth, shift=(-dy, -dx), axis=(0, 1))
        n_label = np.roll(label, shift=(-dy, -dx), axis=(0, 1))
        is_edge = valid & (n_depth > 0) & (n_label != label)
        across = np.where(is_edge & (across == 0), n_depth, across)
    across[:k, :] = across[-k:, :] = across[:, :k] = across[:, -k:] = 0
    move = (across > 0) & (rng.random(depth.shape) < d435.EDGE_SHIFT_PROBABILITY)
    return np.where(move, across, depth).astype(np.float32), move
