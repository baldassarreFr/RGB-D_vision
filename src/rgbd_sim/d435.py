"""Every number of the D435i camera model, in one place.

Each constant has its row number in docs/camera_parameters.md, where its source is given.
The camera-model code imports from here and never writes a number of its own.
Intrinsics (resolution 848 x 480, HFOV 87 deg: rows 1-2) live in camera.py.
Run `python -m rgbd_sim.d435` to print the derived values used in the report.
"""

# Row 3 - stereo geometry
BASELINE_M = 0.050                 # left -> right imager; the right imager is on the camera's +x side

# Rows 4-5 - base noise of a matte surface seen face-on
SUBPIXEL_RMS_PX = 0.08             # sigma_z = z^2 * SUBPIXEL_RMS_PX / (fx * BASELINE_M)

# Rows 6-7 - material factors, by the material names used in scene.py
MATERIAL_NOISE_FACTOR = {
    "aluminium": 2.9,              # Halmetschlager-Funek et al. 2019, Fig. 6 (D435, 1 m)
    "table": 1.0,                  # matte work surface: scene choice
}

# Rows 8-9 - edge artifacts
EDGE_SHIFT_PX = 1                  # Halmetschlager-Funek et al. 2019, Table 2 (D435, 0-3 m)
EDGE_SHIFT_PROBABILITY = 0.5       # OUR CHOICE: the source gives the maximum displacement, not
                                   # how often it happens; 1/2 = no preferred side


def base_noise_m(z, fx):
    """Row 5: depth noise (standard deviation, metres) of a matte surface seen face-on."""
    return z**2 * SUBPIXEL_RMS_PX / (fx * BASELINE_M)


if __name__ == "__main__":
    from rgbd_sim.camera import intrinsics
    fx = intrinsics()[0, 0]
    print(f"fx = {fx:.1f} px, baseline = {BASELINE_M * 1000:.0f} mm")
    for z in (0.49, 0.65, 0.92):
        base = base_noise_m(z, fx) * 1000
        print(f"noise at {z:.2f} m: {base:.2f} mm (matte), "
              f"{base * MATERIAL_NOISE_FACTOR['aluminium']:.2f} mm (aluminium)")
