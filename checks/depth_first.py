"""Step 1: render an IDEAL depth image of a simple scene, seen by a D435-like camera.

Ideal = no noise, no holes, perfect edges. The camera imperfections come later,
as separate functions applied on top of this image.
"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from rgbd_sim.camera import W, H, intrinsics, look_at


# ---------- 1. Camera intrinsics (D435 depth stream at 848x480) ----------
K = intrinsics()
fx = K[0, 0]                        # focal length in pixels (~447 px)


# ---------- 2. Camera pose: where the camera is and where it looks ----------
eye = np.array([0.0, -0.35, 0.55])  # 35 cm in front of the origin, 55 cm above the table
extrinsic = look_at(eye, target=(0, 0, 0))

# ---------- 3. Scene: a table and a bar (placeholder for an aluminium profile) ----------
table = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 0.02)     # 1 m x 1 m, 2 cm thick
table.translate((-0.5, -0.5, -0.02))                              # top surface at z = 0
bar = o3d.geometry.TriangleMesh.create_box(0.20, 0.04, 0.04)     # 200 x 40 x 40 mm
bar.translate((-0.10, -0.02, 0.0))                                # centred, lying on the table

scene = o3d.t.geometry.RaycastingScene()
table_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(table))
bar_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(bar))

# ---------- 4. Cast one ray per pixel ----------
rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    intrinsic_matrix=o3d.core.Tensor(K),
    extrinsic_matrix=o3d.core.Tensor(extrinsic),
    width_px=W,
    height_px=H,
)
hits = scene.cast_rays(rays)

t_hit = hits["t_hit"].numpy()          # (H, W) distance along each ray; inf = ray hit nothing
geom = hits["geometry_ids"].numpy()    # (H, W) id of the mesh each ray hit
dirs = rays.numpy()[..., 3:]           # (H, W, 3) ray directions in world coordinates

# ---------- 5. Ray length -> z-depth (what a RealSense actually reports) ----------
z_axis = extrinsic[2, :3]              # camera viewing direction, in world coordinates
hit = np.isfinite(t_hit)
depth = np.zeros((H, W), dtype=np.float32)          # 0 = "no data", same convention as RealSense
depth[hit] = t_hit[hit] * (dirs[hit] @ z_axis)      # project the hit onto the optical axis

label = np.zeros((H, W), dtype=np.uint8)            # 0 background, 1 table, 2 bar
label[hit & (geom == table_id)] = 1
label[hit & (geom == bar_id)] = 2

# ---------- 6. Sanity checks ----------
print(f"fx = {fx:.1f} px")
print(f"pixels with depth: {hit.mean():.1%}")
print(f"depth range: {depth[hit].min():.3f} .. {depth[hit].max():.3f} m")
print(f"bar pixels: {(label == 2).sum()}")

np.save("depth_ideal.npy", depth)
np.save("label.npy", label)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
im = ax[0].imshow(np.where(hit, depth, np.nan), cmap="viridis")
ax[0].set_title("Ideal depth [m]")
fig.colorbar(im, ax=ax[0])
ax[1].imshow(label, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
ax[1].set_title("Labels: 0 background, 1 table, 2 bar")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig("first_depth.png", dpi=120)
plt.show()
