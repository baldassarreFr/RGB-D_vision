"""
3D models of the parts: a 40x40 T-slot profile and an angle bracket.
The different sizes are not taken from a specific manufacturer, but plausible measures are used

Each part is a 2D cross-section (shapely) extruded into 3D (trimesh).
"""

import trimesh
from shapely import affinity
from shapely.geometry import Point, box
from shapely.ops import unary_union

MM = 0.001               # shapes drawn in millimetres

# T-slot profile 40x40 (all values in mm)
PROFILE_SIZE = 40.0      # outer square
SLOT_OPENING = 8.0       # width of the slot at the surface
LIP = 3.0                # thickness of the lips on each side of the opening
GROOVE_WIDTH = 14.0      # width of the wider groove under the lips
SLOT_DEPTH = 11.0        # from the outer face to the bottom of the groove
BORE_DIAMETER = 6.8      # centre hole
PROFILE_LENGTH = 150.0   # length of the piece

# Angle bracket (all values in mm)
BRACKET_LEG = 40.0       # length of each leg of the L
BRACKET_THICKNESS = 4.0  # wall thickness
BRACKET_WIDTH = 36.0     # how wide the L is extruded


def profile_cross_section():
    """2D shape of the profile seen end-on, in mm, centred at (0, 0)."""
    half = PROFILE_SIZE / 2
    square = box(-half, -half, half, half)
    # One T-shaped slot on the top side is made by a narrow opening through the lip, and a wider groove underneath it.
    opening = box(-SLOT_OPENING / 2, half - LIP, SLOT_OPENING / 2, half)
    groove = box(-GROOVE_WIDTH / 2, half - SLOT_DEPTH, GROOVE_WIDTH / 2, half - LIP)
    slot = unary_union([opening, groove])
    # The same slot on all four sides: rotate it around the centre.
    slots = [affinity.rotate(slot, angle, origin=(0, 0)) for angle in (0, 90, 180, 270)]
    bore = Point(0, 0).buffer(BORE_DIAMETER / 2, quad_segs=16)   # circle as a 64-sided polygon
    return square.difference(unary_union(slots + [bore]))


def bracket_cross_section():
    """2D L shape of the bracket seen from the side, in mm, centred on its bounding box."""
    a, t = BRACKET_LEG, BRACKET_THICKNESS
    l_shape = unary_union([box(0, 0, a, t),     # bottom leg, along x
                           box(0, 0, t, a)])    # upright leg, along y
    return affinity.translate(l_shape, -a / 2, -a / 2)


def extrude(shape, length_mm):
    """Push a 2D shape (mm) along the z axis by length_mm; centre it; convert to metres."""
    mesh = trimesh.creation.extrude_polygon(shape, height=length_mm)   # from z = 0 to z = length
    mesh.apply_translation([0, 0, -length_mm / 2])                      # centre along z
    mesh.apply_scale(MM)                                                 # mm -> m
    return mesh


def make_profile():
    """T-slot profile; its long axis is the model z axis."""
    return extrude(profile_cross_section(), PROFILE_LENGTH)


def make_bracket():
    """Angle bracket; the L lies in the model x-y plane, its width is along z."""
    return extrude(bracket_cross_section(), BRACKET_WIDTH)


if __name__ == "__main__":
    for name, mesh in (("profile", make_profile()), ("bracket", make_bracket())):
        size = mesh.extents / MM                     # bounding-box size, in mm
        volume_cm3 = mesh.volume / MM**3 / 1000      # m^3 -> mm^3 -> cm^3
        print(f"{name:8s} watertight={mesh.is_watertight}  "
              f"size = {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm  "
              f"volume = {volume_cm3:.2f} cm^3  triangles = {len(mesh.faces)}")
