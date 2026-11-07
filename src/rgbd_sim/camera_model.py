"""
Turns the ideal depth of a scene into what a D435i camera would measure.

Each effect is one function. Every camera parameter comes from d435.py.

Step 1 (VARIANTS): the effects that have a published value for this camera, each on or off:
  - noise: per-pixel random error along the viewing axis, whose size depends on distance
    and on the material.
  - stereo shadows: pixels whose 3D point the right imager cannot see get no depth
    (this also produces the invalid band on the left of the image).
  - edges: object outlines move by up to 1 pixel, randomly towards either side.

Step 2 (sensitivity): two effects without a published value for this camera, each tried at several strengths:
    - holes on aluminium seen obliquely: a shiny surface reflects the projector's dots
        like a mirror. The angle at which this happens is not published for the D435, so we study this effect with several values.
    - spatially correlated noise: the random error is not independent per pixel, but neighbouring
        pixels share their error (correlation 0.5 at corr_px pixels). Its size is not published for this camera, so we study this effect with several values.

"""
import numpy as np

from rgbd_sim import d435
from rgbd_sim.camera import backproject, camera_to_world, incidence_angle
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


def noise_sigma(scene, material=True, depth=None):
    """
    (H, W) standard deviation of the depth noise of every pixel, in metres (0 = no depth).

    sigma = base(z)               Intel formula, matte surface seen face-on
            * material factor     if material: 2.9 on aluminium (Halmetschlager-Funek)
    depth: the depth to compute it for (default: the scene's ideal depth).
    """
    meta = scene["meta"]
    depth = scene["depth"] if depth is None else depth
    K = np.array(meta["camera"]["K"])
    valid = depth > 0

    sigma = d435.base_noise_m(depth, K[0, 0])
    if material:
        sigma = sigma * material_factor_image(scene["label"], meta)
    return np.where(valid, sigma, 0.0)


def correlated_field(shape, corr_px, rng):
    """(H, W) Gaussian noise with standard deviation 1 in every pixel, spatially correlated.

    corr_px = distance (pixels) at which the correlation between two pixels' values falls to 0.5.
    Made by smoothing independent noise with a Gaussian kernel and rescaling to std 1.
    For a Gaussian kernel of std k, the correlation at distance r is exp(-r^2 / (4 k^2)),
    which is 0.5 at r = 2 k sqrt(ln 2); so k = corr_px / (2 sqrt(ln 2)).
    """
    k = corr_px / (2 * np.sqrt(np.log(2)))
    radius = int(np.ceil(4 * k))
    x = np.arange(-radius, radius + 1)
    g = np.exp(-x**2 / (2 * k**2))
    g /= np.sqrt(np.sum(g**2))          # keeps std 1 after smoothing (separable: rows, then columns)
    h, w = shape
    white = rng.standard_normal((h + 2 * radius, w + 2 * radius))
    rows = np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 1, white)
    return np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 0, rows)


def add_noise(depth, sigma, rng, corr_px=0):
    """
    Move every valid pixel along the viewing axis by a random amount of size sigma.

    corr_px = 0 (Step 1): independent Gaussian noise per pixel.
    corr_px > 0 (Step 2, sensitivity): the same sigma per pixel, but neighbouring pixels share
    their error (correlation 0.5 at corr_px pixels). Pixels without depth stay 0.
    """
    if corr_px > 0:
        field = correlated_field(depth.shape, corr_px, rng)
    else:
        field = rng.standard_normal(depth.shape)      # exactly as in Step 1
    noisy = depth + field * sigma
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


def aluminium_holes_mask(scene, critical_deg):
    """
    (H, W) bool, Step 2 (sensitivity): aluminium pixels seen more obliquely than critical_deg.
    """
    meta = scene["meta"]
    aluminium = np.zeros(scene["label"].shape, dtype=bool)
    for obj in meta["objects"]:
        if obj["material"] == "aluminium":
            aluminium |= scene["label"] == obj["instance_id"]
    angle = np.degrees(incidence_angle(scene["depth"], scene["normals"],
                                       np.array(meta["camera"]["K"]),
                                       np.array(meta["camera"]["extrinsic_world_to_camera"])))
    with np.errstate(invalid="ignore"):
        return aluminium & (scene["depth"] > 0) & (angle > critical_deg)


def shift_edges(depth, label, rng):
    """
    Move the outline of every object by up to EDGE_SHIFT_PX pixel, randomly in or out.
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



# The camera conditions of the experiment: which effects are switched on.
VARIANTS = {
    "ideal":     dict(edges=False, shadows=False, noise=False, aluminium=False),
    "noise":     dict(edges=False, shadows=False, noise=True,  aluminium=False),  # Intel base noise only
    "aluminium": dict(edges=False, shadows=False, noise=True,  aluminium=True),   # base noise, x2.9 on the parts
    "shadows":   dict(edges=False, shadows=True,  noise=False, aluminium=False),
    "edges":     dict(edges=True,  shadows=False, noise=False, aluminium=False),
    "realistic": dict(edges=True,  shadows=True,  noise=True,  aluminium=True),   # all effects: the D435i model
}

# Step 2 (sensitivity): realistic + one effect without a published value for this camera,
# at several strengths. Step 1 (VARIANTS above) is not changed by these.
STEP2_VARIANTS = {
    **{f"realistic_holes{a}": dict(VARIANTS["realistic"], holes_deg=a) for a in (75, 45, 30)},
    **{f"realistic_corr{s}": dict(VARIANTS["realistic"], corr_px=s) for s in (2, 5, 10)},
}


def apply_camera(scene, edges, shadows, noise, aluminium, rng, corr_px=0, holes_deg=None):
    """
    Ideal depth -> depth as the D435i would measure it, with the chosen effects.

    Order: geometry first, then the random error on what is left.
      1. edges
      2. stereo shadows
      3. noise
    """
    depth = scene["depth"].copy()
    if edges:
        depth, _ = shift_edges(depth, scene["label"], rng)
    if shadows:
        depth[stereo_shadow_mask(scene)] = 0.0
    if holes_deg is not None:
        depth[aluminium_holes_mask(scene, holes_deg)] = 0.0
    if noise:
        sigma = noise_sigma(scene, material=aluminium, depth=depth)
        depth = add_noise(depth, sigma, rng, corr_px=corr_px)
    return depth.astype(np.float32)
