"""D435i camera model: turns the ideal depth of a scene into what the camera would measure.

Each effect is one function. Every number comes from d435.py (see docs/camera_parameters.md);
nothing numeric is written here.
Effects so far:
  - noise: per-pixel random error along the viewing axis, whose size depends on distance
    and on the material (switch `material`).
(An incidence-angle effect was tried and left out: see docs/camera_parameters.md, section 2.)
"""
import numpy as np

from rgbd_sim import d435


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
