"""A simple RGB image for each scene: base colours lit by one light (Blinn-Phong model).

Not photorealistic on purpose. The experiment uses depth; the RGB image makes each
sample a complete RGB-D image. Simplification: RGB is rendered from the depth
camera's viewpoint (as if already aligned to depth); on the real D435i the colour
camera sits a few centimetres to the side.
"""
import numpy as np
from rgbd_sim.camera import backproject, camera_to_world

# Direction from the surface TOWARDS the light (a lamp above, slightly front-left).
LIGHT = np.array([-0.3, -0.4, 1.0]) / np.linalg.norm([-0.3, -0.4, 1.0])
AMBIENT = 0.25          # light that reaches every surface, even facing away from the lamp
DIFFUSE = 0.75          # light that depends on how much the surface faces the lamp

MATERIALS = {
    # name:      (base colour RGB 0-1, highlight strength, highlight sharpness)
    "table":     ((0.35, 0.35, 0.37), 0.0, 1),
    "aluminium": ((0.78, 0.79, 0.81), 0.5, 40),
}


def shade(depth, label, normals, materials, K, extrinsic):
    """RGB image (H, W, 3) uint8.

    materials: list of material names; materials[i] is used for label i + 1.
    """
    camera_centre = camera_to_world(np.zeros(3), extrinsic)
    points = camera_to_world(backproject(depth, K), extrinsic)     # NaN where no depth
    with np.errstate(invalid="ignore"):
        view = camera_centre - points                               # surface -> camera
        view /= np.linalg.norm(view, axis=-1, keepdims=True)
        halfway = view + LIGHT                                      # halfway between view and light
        halfway /= np.linalg.norm(halfway, axis=-1, keepdims=True)

    n_dot_l = np.clip(np.sum(normals * LIGHT, axis=-1), 0, None)    # 1 = faces the lamp, 0 = sideways or away
    n_dot_h = np.clip(np.sum(normals * halfway, axis=-1), 0, None)  # 1 = mirror direction: highlight

    rgb = np.zeros(depth.shape + (3,))                              # black where nothing is hit
    for i, name in enumerate(materials):
        mask = label == i + 1
        base, highlight, sharpness = MATERIALS[name]
        brightness = AMBIENT + DIFFUSE * n_dot_l[mask]
        shine = highlight * n_dot_h[mask] ** sharpness
        rgb[mask] = np.asarray(base) * brightness[:, None] + shine[:, None]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
