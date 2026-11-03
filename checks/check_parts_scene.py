"""
Puts a bracket and a profile on the table and look at them with the D435 camera (ideal depth).
Then, checks that both parts are visible, and their top surfaces come out at the right height.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix

from rgbd_sim.camera import W, H, intrinsics, look_at, backproject, camera_to_world
from rgbd_sim.parts import make_profile, make_bracket
from rgbd_sim.render import render_ideal

# Camera
K = intrinsics()
extrinsic = look_at(eye=(0.0, -0.35, 0.55), target=(0, 0, 0))

# Scene
table = trimesh.creation.box(extents=[1.0, 1.0, 0.02])   # created centred at the origin
table.apply_translation([0, 0, -0.01])                    # top face at z = 0

# Pose = 4x4 matrix model -> world. First rotate, then move
# Profile: lay it down (model z = long axis -> world x), lift its centre 20 mm (half of 40)
pose_profile = translation_matrix([0.0, -0.05, 0.02]) @ rotation_matrix(np.pi / 2, [0, 1, 0])
# Bracket: stand it up (model y = upright leg -> world z), then turn it 90 deg about
# the vertical so the upright leg is at the back and the L opens towards the camera
pose_bracket = (translation_matrix([0.06, 0.08, 0.02])
                @ rotation_matrix(-np.pi / 2, [0, 0, 1])
                @ rotation_matrix(np.pi / 2, [1, 0, 0]))

profile = make_profile()
profile.apply_transform(pose_profile)
bracket = make_bracket()
bracket.apply_transform(pose_bracket)

# Render: 1 = table, 2 = profile, 3 = bracket
depth, label = render_ideal([table, profile, bracket], K, extrinsic, W, H)

# Checks: pixel count and highest point of each object
pts = camera_to_world(backproject(depth, K), extrinsic)
for lab, name, expected_top in ((1, "table", 0.0), (2, "profile", 40.0), (3, "bracket", 40.0)):
    z_mm = pts[label == lab][:, 2] * 1000
    print(f"{name:8s} {len(z_mm):7d} pixels   top = {z_mm.max():6.2f} mm   (expected {expected_top:.0f})")

# Figure: full depth, labels, and a close-up of each part
def crop_box(mask, pad=15):
    """Row/column range around the pixels where mask is True, with some margin."""
    rows, cols = np.where(mask)
    return (slice(max(rows.min() - pad, 0), rows.max() + pad),
            slice(max(cols.min() - pad, 0), cols.max() + pad))

depth_show = np.where(depth > 0, depth, np.nan)       # empty pixels drawn white
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
ax[0, 0].imshow(depth_show, cmap="viridis")
ax[0, 0].set_title("Ideal depth [m]")
ax[0, 1].imshow(label, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
ax[0, 1].set_title("Labels: 1 table, 2 profile, 3 bracket")
for a, lab, name in ((ax[1, 0], 2, "profile"), (ax[1, 1], 3, "bracket")):
    rows, cols = crop_box(label == lab)
    im = a.imshow(depth_show[rows, cols], cmap="viridis", interpolation="nearest")
    a.set_title(f"Close-up: {name} (depth [m])")
    fig.colorbar(im, ax=a)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/parts_scene.png", dpi=120)
