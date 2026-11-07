"""
Ideal depth rendering using Open3D ray casting: one ray per pixel into a scene of meshes.
"""
import numpy as np
import open3d as o3d


def render_ideal(meshes, K, extrinsic, width, height, with_normals=False):
    """Ideal depth and label images of a scene.

    meshes:    list of trimesh.Trimesh, already placed in world coordinates (metres)
    K:         3x3 intrinsic matrix
    extrinsic: 4x4 world->camera matrix
    returns:   depth (H, W) float32, z-depth in metres, 0 = no hit
               label (H, W) uint8, 0 = nothing hit, i + 1 = meshes[i]
               normals (H, W, 3), only if with_normals=True: unit vector perpendicular
               to the surface seen by each pixel, in world coordinates (0 = no hit)
    """
    # 1. Put every mesh in the ray-casting scene; remember the id Open3D gives it.
    scene = o3d.t.geometry.RaycastingScene()
    ids = []
    for mesh in meshes:
        vertices = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
        triangles = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32))
        ids.append(scene.add_triangles(vertices, triangles))

    # 2. One ray per pixel, through the pixel centre.
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(K),
        extrinsic_matrix=o3d.core.Tensor(extrinsic),
        width_px=width,
        height_px=height,
    )
    hits = scene.cast_rays(rays)
    t_hit = hits["t_hit"].numpy()
    geom = hits["geometry_ids"].numpy()
    dirs = rays.numpy()[..., 3:]

    # 3. Ray parameter -> z-depth (distance along the camera's forward axis).
    hit = np.isfinite(t_hit)
    z_axis = extrinsic[2, :3]
    depth = np.zeros((height, width), dtype=np.float32)
    depth[hit] = t_hit[hit] * (dirs[hit] @ z_axis)

    # 4. Label image: which mesh each pixel sees.
    label = np.zeros((height, width), dtype=np.uint8)
    for i, mesh_id in enumerate(ids):
        label[hit & (geom == mesh_id)] = i + 1

    # 5. Optional: surface normals, used for shading the RGB image and for the effects
    #    that depend on the viewing angle (Step 2 holes).
    if with_normals:
        normals = hits["primitive_normals"].numpy().astype(np.float64)
        normals[~hit] = 0.0
        return depth, label, normals
    return depth, label


def first_hit_distance(meshes, origins, directions):
    """Distance along each ray to the first surface it hits (inf if none).

    origins, directions: (N, 3) arrays in world coordinates; directions must have length 1,
    so that the returned ray parameter is a distance in metres.
    """
    scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                            o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
