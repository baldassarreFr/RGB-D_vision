"""
Ideal depth rendering: one ray per pixel into a scene of meshes (Open3D ray casting).
"""
import numpy as np
import open3d as o3d


def render_ideal(meshes, K, extrinsic, width, height):
    """Ideal depth and label images of a scene.

    meshes:    list of trimesh.Trimesh, already placed in world coordinates (metres)
    K:         3x3 intrinsic matrix
    extrinsic: 4x4 world->camera matrix
    returns:   depth (H, W) float32, z-depth in metres, 0 = no hit
               label (H, W) uint8, 0 = nothing hit, i + 1 = meshes[i]
    """
    # Put every mesh in the ray-casting scene
    scene = o3d.t.geometry.RaycastingScene()
    ids = []
    for mesh in meshes:
        vertices = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
        triangles = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32))
        ids.append(scene.add_triangles(vertices, triangles))

    # One ray per pixel, through the pixel centre
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

    # Ray parameter -> z-depth
    hit = np.isfinite(t_hit)
    z_axis = extrinsic[2, :3]
    depth = np.zeros((height, width), dtype=np.float32)
    depth[hit] = t_hit[hit] * (dirs[hit] @ z_axis)

    # Label image
    label = np.zeros((height, width), dtype=np.uint8)
    for i, mesh_id in enumerate(ids):
        label[hit & (geom == mesh_id)] = i + 1
    return depth, label




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
    # Put every mesh in the ray-casting scene
    scene = o3d.t.geometry.RaycastingScene()
    ids = []
    for mesh in meshes:
        vertices = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
        triangles = o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32))
        ids.append(scene.add_triangles(vertices, triangles))
 
    # One ray per pixel, through the pixel centre
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
 
    # Ray parameter -> z-depth
    hit = np.isfinite(t_hit)
    z_axis = extrinsic[2, :3]
    depth = np.zeros((height, width), dtype=np.float32)
    depth[hit] = t_hit[hit] * (dirs[hit] @ z_axis)
 
    # Label image
    label = np.zeros((height, width), dtype=np.uint8)
    for i, mesh_id in enumerate(ids):
        label[hit & (geom == mesh_id)] = i + 1
 
    # surface normals
    # used for shading the RGB image and for the camera effects that depend on the viewing angle
    if with_normals:
        normals = hits["primitive_normals"].numpy().astype(np.float64)
        normals[~hit] = 0.0
        return depth, label, normals
    return depth, label
 