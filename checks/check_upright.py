"""
How often do profiles land standing on one end? Compare bounce (restitution) values.

Usage: python checks/check_upright.py [n_seeds]     (default 100 seeds, physics only)
A profile counts as "upright" when its long axis (model z) is within ~25 deg of vertical.
"""
import sys
import time

import numpy as np

from rgbd_sim.parts import make_profile, make_bracket
from rgbd_sim.physics import drop_parts
from rgbd_sim.scene import choose_parts

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MAKE = {"profile": make_profile, "bracket": make_bracket}

for restitution in (0.0, 0.2):
    t0 = time.time()
    profiles = upright = parts = far = not_settled = 0
    seconds = []
    for seed in range(n_seeds):
        names = choose_parts(seed)
        poses, info = drop_parts(names, [MAKE[n]() for n in names], seed=seed,
                                 restitution=restitution)
        seconds.append(info["simulated_seconds"])
        not_settled += not info["settled"]
        for name, pose in zip(names, poses):
            parts += 1
            x, y = pose[0, 3], pose[1, 3]
            far += not (-0.30 < x < 0.30 and -0.25 < y < 0.30)    # wandered out of the area
            if name == "profile":
                profiles += 1
                upright += abs(pose[2, 2]) > 0.9                  # model z almost vertical
    print(f"restitution {restitution:.1f}: upright profiles {upright}/{profiles} "
          f"({100 * upright / profiles:.0f}%), parts far from the drop area {far}/{parts}, "
          f"not settled {not_settled}/{n_seeds}, settle time {np.mean(seconds):.1f} s avg "
          f"/ {max(seconds):.1f} s max   [{time.time() - t0:.0f} s]")
