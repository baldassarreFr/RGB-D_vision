# Documentation

Details of the synthetic scene pipeline.

## Physics-based placement

Parts are dropped onto the table with PyBullet [1] so that they come to rest in
physically plausible poses. Each scene contains 1–2 profiles and 1–4 brackets, chosen by the
seed. Every part starts from a uniformly random orientation above its own drop point; drop
points are drawn by rejection sampling in a 0.30 × 0.27 m area with a minimum spacing of
90 mm, so that parts do not land on top of each other.

The simulator uses simple collision shapes: one box for a profile, and two boxes, one per leg,
for a bracket. A single box or a convex hull would fill the inside of the L and allow resting
poses that are physically impossible. The mass follows from the mesh volume and the density of
aluminium, and each body rotates about the true centre of mass computed from the mesh; for the
bracket this is 8.5 mm away from the bounding-box centre towards the corner of the L, which
decides how it tips over. Bullet multiplies the coefficients of the two bodies in contact; with
a part friction of 0.6 and the default table friction of 0.5, the effective friction is 0.30
between part and table and 0.36 between parts. The simulation runs at 240 Hz until all parts
have been still for 0.5 s.

The resulting scenes show the three stable poses of the bracket (standing as an L, lying on its
side, and resting on both leg tips like a tent) and, in about 6 % of the cases, a profile
standing on one end. I tested whether a small restitution removes these upright profiles: it
does not, because a vertical bounce creates no rotation. A restitution of 0.2 left the fraction
unchanged (10 of 161 profiles), while 0.5 reduced it only from 6 % to 5 % and pushed 6 % of the
parts (25 of 428) out of the working area. I therefore kept no restitution and accept upright
profiles as a rare but possible pose.



## Pose estimation: pilot runs

The report describes four changes to the Open3D global-registration tutorial:
a voxel size per class, a RANSAC tolerance of 1.0 v instead of 1.5 v, a wide ICP pass before
the tight one, and five hypotheses per part chosen by a depth check. This section documents the
pilot runs behind them. All pilot runs use the tuning scenes (never the evaluation scenes) and
ideal depth unless stated otherwise. The numbers were recorded during development; the
commands repeat the runs with the current code. Code: `src/rgbd_sim/pose.py`.

### Voxel size per class

On pilot scenes 0–9, the method solves 43 of 46 parts (centre error below 1 mm) with
v = 8 mm for profiles and v = 3 mm for brackets:

| voxel v | profiles solved | brackets solved |
|---|---|---|
| 3 mm | 11 / 17 | **26 / 29** |
| 8 mm | **17 / 17** | 25 / 29 |
| 10 mm | 15 / 17 | 22 / 29 |

The depth-check score separates the outcomes: correct poses scored at least 0.983, wrong
ones at most 0.883.

Reproduce, for each scene and voxel size:
```
for s in 0 1 2 3 4 5 6 7 8 9; do
    python checks/check_pose_one.py outputs/dataset/scenes/00000$s 3
done
```
(and the same with 8 and 10 instead of 3).

### Why one voxel size does not fit both classes

With v = 3 mm, FPFH paired fewer than 1 % of the profile points correctly, and RANSAC never
found the true pose, although that pose had the most inliers. On scene 1 (v = 3 mm):

| | profiles | brackets |
|---|---|---|
| FPFH pairs pointing to the right model point | 0.2–0.6 % | 4–13 % |

Inlier share of the true pose: 1.00; of the poses returned by RANSAC: 0.38–0.77. The search
fails, not the criterion. With v = 8 mm the profiles' RANSAC poses reach an inlier share of
0.95–1.00.

Reproduce: `python checks/check_ransac.py outputs/dataset/scenes/000001 3` (and with 8).

### RANSAC tolerance

Tutorial value: 1.5 v. On tuning scene 0, 1.5 v exceeds the 4 mm wall of the brackets for
v ≥ 3 mm, and RANSAC confused the two faces of the wall; 0.5 v lost correct poses to the
rounding of the voxel grid; 1.0 v solved all 5 parts at v = 3 mm.

Reproduce: `python checks/check_pose_one.py outputs/dataset/scenes/000000 3 ideal 1.0`
(and with 0.5 and 1.5 as the fourth argument).

## Pose estimation results

| Parts | Condition | Success | ADD-S AUC | Median centre error [mm] | Centre error below 1 mm | Centre error below 2 mm |
|---|---|---|---|---|---|---|
| All (420) | A | 0.993 | 0.969 | 0.01 | 0.87 | 0.90 |
| | B | 0.990 | 0.848 | 1.90 | 0.05 | 0.56 |
| | C | 0.981 | 0.845 | 1.83 | 0.09 | 0.63 |
| Brackets (267) | A | 0.989 | 0.954 | 0.01 | 0.81 | 0.86 |
| | B | 0.985 | 0.800 | 1.98 | 0.03 | 0.52 |
| | C | 0.970 | 0.785 | 1.98 | 0.01 | 0.52 |
| Profiles (153) | A | 1.000 | 0.995 | 0.01 | 0.98 | 0.98 |
| | B | 1.000 | 0.932 | 1.84 | 0.08 | 0.64 |
| | C | 1.000 | 0.950 | 1.45 | 0.21 | 0.81 |

*Pose estimation on the 100 test scenes. For all parts, the 95 % confidence intervals are at
most ±0.02 for success and AUC and ±0.1 mm for the median centre error.*

## References

1. E. Coumans, Y. Bai, *PyBullet, a Python module for physics simulation for games, robotics
   and machine learning*, 2016–2021. https://pybullet.org
2. Q.-Y. Zhou, J. Park, V. Koltun, "Open3D: A modern library for 3D data processing,"
   arXiv:1801.09847, 2018.
3. J. F. Blinn, "Models of light reflection for computer synthesized pictures," *SIGGRAPH*, 1977.
