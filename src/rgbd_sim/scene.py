"""One synthetic scene: drop parts, render ideal RGB-D and labels, save to / load from disk.

Folder of a saved scene:
  rgb.png    8-bit colour image (H, W, 3)
  depth.png  16-bit depth; value * depth_scale_m (in meta.json) = metres; 0 = no data
  label.png  8-bit instance ids: 0 = nothing, 1 = table, 2.. = parts (see meta.json)
  meta.json  camera parameters, and for every part: its class, material and true pose
  normals.npz  surface normal seen by each pixel, world frame, (H, W, 3) float16; 0 = no hit
"""
import json
import os

import numpy as np
import trimesh
from PIL import Image

from rgbd_sim.camera import W, H, HFOV_DEG, intrinsics, look_at
from rgbd_sim.parts import make_profile, make_bracket
from rgbd_sim.physics import drop_parts
from rgbd_sim.render import render_ideal
from rgbd_sim.shading import shade

DEPTH_SCALE = 0.0001                   # metres per depth unit (0.1 mm), stored in meta.json
CAMERA_EYE = (0.0, -0.35, 0.55)        # same camera pose as all the checks so far
CAMERA_TARGET = (0.0, 0.0, 0.0)
MAKE_PART = {"profile": make_profile, "bracket": make_bracket}
MATERIAL = {"table": "table", "profile": "aluminium", "bracket": "aluminium"}


def make_table():
    table = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
    table.apply_translation([0, 0, -0.01])          # top face at z = 0
    return table


def choose_parts(seed):
    """Which parts are in the scene with this seed: 1-2 profiles and 1-4 brackets."""
    rng = np.random.default_rng(seed)
    n_profiles = int(rng.integers(1, 3))            # 1 or 2
    n_brackets = int(rng.integers(1, 5))            # 1 to 4
    return ["profile"] * n_profiles + ["bracket"] * n_brackets


def generate_scene(seed):
    """Build one scene. The seed decides everything: part count, drop poses, result."""
    # 1. Which parts, and where they come to rest.
    names = choose_parts(seed)
    meshes = [MAKE_PART[n]() for n in names]
    poses, physics_info = drop_parts(names, meshes, seed=seed)
    placed = []
    for mesh, pose in zip(meshes, poses):
        m = mesh.copy()
        m.apply_transform(pose)
        placed.append(m)

    # 2. Render: depth, labels (1 = table, 2.. = parts), normals; then the RGB image.
    K = intrinsics()
    extrinsic = look_at(CAMERA_EYE, CAMERA_TARGET)
    depth, label, normals = render_ideal([make_table()] + placed, K, extrinsic, W, H,
                                         with_normals=True)
    materials = [MATERIAL["table"]] + [MATERIAL[n] for n in names]
    rgb = shade(depth, label, normals, materials, K, extrinsic)

    # 3. Everything needed to use and to check this scene later.
    objects = []
    for i, (name, pose) in enumerate(zip(names, poses)):
        instance = i + 2
        objects.append({
            "instance_id": instance,
            "class": name,
            "material": MATERIAL[name],
            "pose_world": pose.tolist(),                 # model -> world
            "pose_camera": (extrinsic @ pose).tolist(),  # model -> camera: what pose estimation must find
            "visible_pixels": int((label == instance).sum()),
        })
    meta = {
        "seed": seed,
        "camera": {
            "reference": "Intel RealSense D435i, depth stream (simulated)",
            "width": W, "height": H, "hfov_deg": HFOV_DEG,
            "K": K.tolist(),
            "extrinsic_world_to_camera": extrinsic.tolist(),
            "depth_scale_m": DEPTH_SCALE,
        },
        "table_instance_id": 1,
        "table_material": MATERIAL["table"],
        "objects": objects,
        "physics": physics_info,
    }
    return {"rgb": rgb, "depth": depth, "label": label, "normals": normals, "meta": meta}


def scene_meshes(meta):
    """Rebuild the scene's meshes (table first, then the parts) from meta.json alone."""
    meshes = [make_table()]
    for obj in meta["objects"]:
        mesh = MAKE_PART[obj["class"]]()
        mesh.apply_transform(np.array(obj["pose_world"]))
        meshes.append(mesh)
    return meshes


def save_scene(scene, folder):
    os.makedirs(folder, exist_ok=True)
    depth_units = np.round(scene["depth"] / DEPTH_SCALE)
    assert depth_units.max() < 2**16, "depth too large for 16 bits at this scale"
    Image.fromarray(scene["rgb"]).save(os.path.join(folder, "rgb.png"))
    Image.fromarray(depth_units.astype(np.uint16)).save(os.path.join(folder, "depth.png"))
    Image.fromarray(scene["label"]).save(os.path.join(folder, "label.png"))
    np.savez_compressed(os.path.join(folder, "normals.npz"),
                        normals=scene["normals"].astype(np.float16))   # float16: ~0.001 precision
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(scene["meta"], f, indent=2)


def load_scene(folder):
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    depth_units = np.array(Image.open(os.path.join(folder, "depth.png")))
    return {
        "rgb": np.array(Image.open(os.path.join(folder, "rgb.png"))),
        "depth": depth_units.astype(np.float32) * meta["camera"]["depth_scale_m"],
        "label": np.array(Image.open(os.path.join(folder, "label.png"))),
        "normals": np.load(os.path.join(folder, "normals.npz"))["normals"].astype(np.float32),
        "meta": meta,
    }


def save_models(folder):
    """The part models in their own frame (PLY), needed later for pose estimation."""
    os.makedirs(folder, exist_ok=True)
    for name, make in MAKE_PART.items():
        make().export(os.path.join(folder, f"{name}.ply"))
