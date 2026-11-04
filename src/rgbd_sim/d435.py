"""Every number of the D435i camera model, in one place.

Each constant has its row number in docs/camera_parameters.md, where its source is given.
The camera-model code imports from here and never writes a number of its own.
Intrinsics (resolution 848 x 480, HFOV 87 deg: rows 1-2) live in camera.py.
Run `python -m rgbd_sim.d435` to print the derived values used in the report.
"""
import numpy as np

# Row 3 - stereo geometry
BASELINE_M = 0.050                 # left -> right imager; the right imager is on the camera's +x side

# Rows 4-5 - base noise of a matte surface seen face-on
SUBPIXEL_RMS_PX = 0.08             # sigma_z = z^2 * SUBPIXEL_RMS_PX / (fx * BASELINE_M)

# Rows 6-7 - incidence-angle factor (Ahn et al. 2019, Eq. 2)
AHN_C0, AHN_C1, AHN_C2, AHN_C_ANGLE = 0.001063, 0.0007278, 0.003949, 0.022
ANGLE_CAP_DEG = 75.0               # largest angle Ahn et al. measured; no extrapolation beyond it

# Rows 8-9 - material factors, by the material names used in scene.py
MATERIAL_NOISE_FACTOR = {
    "aluminium": 2.9,              # Halmetschlager-Funek et al. 2019, Fig. 6 (D435, 1 m)
    "table": 1.0,                  # matte work surface: scene choice
}

# Row 10 - edge artifacts
EDGE_SHIFT_PX = 1                  # Halmetschlager-Funek et al. 2019, Table 2 (D435, 0-3 m)


def base_noise_m(z, fx):
    """Row 5: depth noise (standard deviation, metres) of a matte surface seen face-on."""
    return z**2 * SUBPIXEL_RMS_PX / (fx * BASELINE_M)


def angle_factor(z, theta):
    """Rows 6-7: how much noise grows with the incidence angle theta (radians), capped at 75 deg.

    Ratio of Ahn et al.'s Eq. 2 at angle theta to the same equation at 0 deg.
    """
    theta = np.minimum(theta, np.deg2rad(ANGLE_CAP_DEG))
    at_zero = AHN_C0 + AHN_C1 * z + AHN_C2 * z**2
    angle_term = AHN_C_ANGLE * z**1.5 * theta / (np.pi / 2 - theta) ** 2
    return (at_zero + angle_term) / at_zero


if __name__ == "__main__":
    from rgbd_sim.camera import intrinsics
    fx = intrinsics()[0, 0]
    print(f"fx = {fx:.1f} px, baseline = {BASELINE_M * 1000:.0f} mm")
    for z in (0.49, 0.65, 0.92):
        print(f"base noise at {z:.2f} m: {base_noise_m(z, fx) * 1000:.2f} mm")
    z = 0.65
    for deg in (0, 15, 30, 45, 60, 75, 85):
        print(f"angle factor at {z} m, {deg:2d} deg: {angle_factor(z, np.deg2rad(deg)):6.2f}")
