"""Physics drop: let the parts fall on the table and settle, to get realistic resting poses.

The physics engine uses simple collision shapes, not the detailed meshes:
- for the profile: one box
- for the bracket: two boxes, one per leg of the L.
"""
import numpy as np
import pybullet as p
from trimesh.transformations import random_rotation_matrix, quaternion_from_matrix

from rgbd_sim.parts import (MM, PROFILE_SIZE, PROFILE_LENGTH,
                            BRACKET_LEG, BRACKET_THICKNESS, BRACKET_WIDTH)

ALUMINIUM_DENSITY = 2700.0      # kg/m^3
TIME_STEP = 1.0 / 240.0         # PyBullet's default simulation step, in seconds
RESTITUTION = 0.0               # bounciness of the parts: 0 = no bounce, 1 = perfectly elastic

# Bullet multiplies the values of the two bodies in contact:
PART_FRICTION = 0.6             # our choice, not measured: part-table 0.6 x 0.5 = 0.30, part-part 0.36
TABLE_FRICTION = 0.5            # PyBullet's default
TABLE_RESTITUTION = 1.0         # so that part-table restitution = RESTITUTION of the parts (x 1)

# Where parts are dropped from (world x, y, metres): the central part of the camera view.
DROP_X = (-0.15, 0.15)
DROP_Y = (-0.12, 0.15)
MIN_SPACING = 0.09              # minimum distance between two drop points, metres


def drop_points(n, rng, tries=1000):
    """n random (x, y) drop points, at least MIN_SPACING apart from each other.

    Rejection sampling: draw a random point, keep it only if it is far enough
    from the points already kept. Gives up (and accepts closer points) after
    `tries` attempts, so it never loops forever.
    """
    points = []
    for _ in range(tries):
        if len(points) == n:
            break
        candidate = np.array([rng.uniform(*DROP_X), rng.uniform(*DROP_Y)])
        if all(np.linalg.norm(candidate - q) >= MIN_SPACING for q in points):
            points.append(candidate)
    while len(points) < n:                       # area too crowded: fill without the spacing rule
        points.append(np.array([rng.uniform(*DROP_X), rng.uniform(*DROP_Y)]))
    return points


def collision_boxes(name):
    """Collision boxes of a part: list of (centre, half_size), in the model frame, metres."""
    if name == "profile":
        h = PROFILE_SIZE / 2 * MM
        return [((0, 0, 0), (h, h, PROFILE_LENGTH / 2 * MM))]
    if name == "bracket":
        a, t, w = BRACKET_LEG * MM, BRACKET_THICKNESS * MM, BRACKET_WIDTH * MM
        return [((0, -a / 2 + t / 2, 0), (a / 2, t / 2, w / 2)),    # bottom leg, along x
                ((-a / 2 + t / 2, 0, 0), (t / 2, a / 2, w / 2))]    # upright leg, along y
    raise ValueError(f"unknown part: {name}")


def drop_parts(names, meshes, seed, max_seconds=10.0, restitution=RESTITUTION):
    """Drop the parts from random poses above the table and simulate until they stop.

    names:   e.g. ["profile", "bracket", "bracket"]
    meshes:  the matching trimesh meshes, in their model frame
    returns: list of 4x4 poses (model -> world), one per part, and a dict of info
    """
    rng = np.random.default_rng(seed)               # same seed -> same scene
    client = p.connect(p.DIRECT)                    # physics only, no window
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
    table = p.createMultiBody(0, plane, physicsClientId=client)   # mass 0 = fixed: table top at z = 0

    p.changeDynamics(table, -1, lateralFriction=TABLE_FRICTION, restitution=TABLE_RESTITUTION,
                     physicsClientId=client)

    xy = drop_points(len(names), rng)               # one separate drop point per part
    bodies, coms = [], []
    for i, (name, mesh) in enumerate(zip(names, meshes)):
        # PyBullet moves each body around its centre of mass, so we place the body's
        # frame there and shift the collision boxes by the same amount.
        com = np.asarray(mesh.center_mass)          # centre of mass, in the model frame
        boxes = collision_boxes(name)
        shape = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_BOX] * len(boxes),
            halfExtents=[list(half) for _, half in boxes],
            collisionFramePositions=[list(np.asarray(centre) - com) for centre, _ in boxes],
            physicsClientId=client)

        # Random starting pose: random rotation, its own drop point, and each part a bit
        # higher than the previous one, as extra safety against starting inside each other.
        R0 = random_rotation_matrix(rng.random(3))
        q = quaternion_from_matrix(R0)              # trimesh order:  w, x, y, z
        start = [xy[i][0], xy[i][1], 0.06 + 0.06 * i]
        body = p.createMultiBody(
            baseMass=mesh.volume * ALUMINIUM_DENSITY,
            baseCollisionShapeIndex=shape,
            basePosition=start,
            baseOrientation=[q[1], q[2], q[3], q[0]],    # PyBullet order: x, y, z, w
            physicsClientId=client)
        p.changeDynamics(body, -1, lateralFriction=PART_FRICTION, restitution=restitution,
                         physicsClientId=client)
        bodies.append(body)
        coms.append(com)

    # Simulate until every part has been still for half a second (or max_seconds).
    steps, still_checks = 0, 0
    while steps * TIME_STEP < max_seconds and still_checks < 5:
        p.stepSimulation(physicsClientId=client)
        steps += 1
        if steps % 24 == 0:                          # check 10 times per simulated second
            moving = False
            for body in bodies:
                linear, angular = p.getBaseVelocity(body, physicsClientId=client)
                if np.linalg.norm(linear) > 1e-3 or np.linalg.norm(angular) > 1e-2:
                    moving = True
            still_checks = 0 if moving else still_checks + 1

    # Overlap check: a negative contact distance means two shapes are inside each other.
    contacts = p.getContactPoints(physicsClientId=client)
    deepest = min([c[8] for c in contacts], default=0.0)

    # Read the final poses and convert them back to the model frame of each part.
    poses = []
    for body, com in zip(bodies, coms):
        position, orientation = p.getBasePositionAndOrientation(body, physicsClientId=client)
        R = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = np.asarray(position) - R @ com     # body frame is at the centre of mass
        poses.append(pose)
    p.disconnect(client)

    info = {"simulated_seconds": steps * TIME_STEP,
            "settled": still_checks >= 5,
            "deepest_overlap_mm": -deepest * 1000}
    return poses, info