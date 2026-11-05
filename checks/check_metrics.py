"""M3 check: the error measures give the expected answer on poses whose error we know.

Usage: python checks/check_metrics.py
For each part, starting from an arbitrary "true" pose:
1. the true pose itself                      -> every error 0
2. every symmetry: first checks that the rotated model really lies on the original one
   (largest distance ~0), then the errors of the symmetric pose -> ADD-S ~0, centre 0, rotation 0
   (the plain angle between the two poses is printed too: it is NOT 0)
3. poses that are wrong by a known amount    -> centre error = that amount; ADD-S as printed
Also prints the diameters and the success thresholds (10 % of the diameter).
"""
import numpy as np

from rgbd_sim.metrics import (SYMMETRIES, _rot, add_s, centre_error, diameter,
                              distance_to_surface, metric_points, rotation_angle,
                              rotation_error, success)
from rgbd_sim.scene import MAKE_PART


def pose(R=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


true = pose(_rot([1, 2, 3], 40), (0.02, -0.01, 0.65))       # arbitrary pose in front of the camera
mm = 1000
wrong = {
    "profile": {"slid 10 mm along its axis": pose(t=(0, 0, 0.010)),
                "slid 50 mm along its axis": pose(t=(0, 0, 0.050)),
                "turned 90 deg about x (not a symmetry)": pose(_rot([1, 0, 0], 90))},
    "bracket": {"moved 4 mm (one wall thickness) along x": pose(t=(0.004, 0, 0)),
                "turned 90 deg about z (not a symmetry)": pose(_rot([0, 0, 1], 90)),
                "turned 10 deg about z": pose(_rot([0, 0, 1], 10))},
}

for cls, make in MAKE_PART.items():
    mesh = make()
    points = metric_points(mesh)
    d = diameter(mesh)
    print(f"\n{cls}: diameter {d * mm:.1f} mm, success if ADD-S < {0.1 * d * mm:.1f} mm")
    print(f"  {'case':42s} {'ADD-S':>7s} {'ok':>3s} {'centre':>7s} {'rot.err':>7s} {'plain angle':>11s}")

    def row(name, est):
        a = add_s(est, true, mesh, points)
        print(f"  {name:42s} {a * mm:6.2f}  {'yes' if success(a, d) else 'no':>3s} "
              f"{centre_error(est, true) * mm:6.2f}  {rotation_error(est, true, cls):6.1f}° "
              f"{rotation_angle(true[:3, :3], est[:3, :3]):10.1f}°")

    row("true pose", true)
    for i, S in enumerate(SYMMETRIES[cls]):
        on_itself = distance_to_surface(points @ S.T, mesh, np.eye(4)).max() * mm
        row(f"symmetry {i} (rotated model off itself: {on_itself:.3f} mm)", true @ pose(S))
    for name, error in wrong[cls].items():
        row(name, true @ error)                 # error applied in the model frame
