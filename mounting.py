"""Per-segment IMU mounting corrections.

Project segment frame convention:
- +X: forward
- +Y: up
- +Z: left

All rotations here are defined as R_seg_from_sensor:
they rotate vectors expressed in the SENSOR frame into the SEGMENT frame:
    v_seg = R_seg_from_sensor.apply(v_sensor)

Given a world orientation of the sensor (R_world_from_sensor), we compute the
world orientation of the segment as:
    R_world_from_segment = R_world_from_sensor * (R_seg_from_sensor)^-1
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length axis provided to mounting definition.")
    return v / n


def _R_from_cols(col_x: np.ndarray,
                 col_y: np.ndarray,
                 col_z: Optional[np.ndarray] = None) -> R:
    """
    Build Rotation from the SENSOR basis vectors expressed in SEGMENT coordinates.

    IMPORTANT:
    - This function enforces a *proper* right-handed rotation matrix.
    - If det(M) < 0 (left-handed), it flips the 3rd column.
    - If col_z is None, it is computed as cross(col_x, col_y).
    """
    cx = _unit(col_x)
    cy = _unit(col_y)

    if col_z is None:
        cz = np.cross(cx, cy)
        cz = _unit(cz)
    else:
        cz = _unit(col_z)

    M = np.column_stack([cx, cy, cz]).astype(float)

    # Enforce a proper rotation (determinant +1). If it's a reflection, flip Z.
    if np.linalg.det(M) < 0:
        M[:, 2] *= -1.0

    return R.from_matrix(M)


# Helper basis vectors in SEGMENT coordinates (X fwd, Y up, Z right)
X = np.array([1.0, 0.0, 0.0])   # forward
Y = np.array([0.0, 1.0, 0.0])   # up
Z = np.array([0.0, 0.0, 1.0])   # left

# Convenience aliases
FWD = X
BWD = -X
UP  = Y
LFT = -Z
RGT = Z  # "right" in this segment convention


# --- Mounting definitions from user (N-pose) ---
# NOTE:
# These are "sensor axis -> segment axis" mappings:
#   col1 = where sensor +X points in segment coords
#   col2 = where sensor +Y points in segment coords
#   col3 = where sensor +Z points in segment coords
#
# We keep your stated directions as-is, but _R_from_cols enforces right-handedness
# so SciPy won't throw determinant errors.

# Left thigh: +x up, +y forward, +z left
THIGH_L = _R_from_cols(UP, FWD)

# Right thigh mirrored: +x up, +y forward, +z right
THIGH_R = _R_from_cols(UP, BWD)

# Left lower leg: +x up, +y forward, +z out (left => left)
SHANK_L = _R_from_cols(UP, FWD)

# Right lower leg mirrored: out => right
SHANK_R = _R_from_cols(UP, BWD)

# Upper arm Left: +x up, +y forward, +z out (left => left)
UARM_L = _R_from_cols(UP, FWD)
UARM_R = _R_from_cols(UP, BWD)

# Forearm Left: same convention
FARM_L = _R_from_cols(UP, FWD)
FARM_R = _R_from_cols(UP, BWD)

# Hand Left: same convention
HAND_L = _R_from_cols(UP, FWD)
HAND_R = _R_from_cols(UP, BWD)

# Torso (mounted on front of torso): +z forwards, +x up, +y right
# sensor x -> up (Y), sensor y -> right (-Z), sensor z -> forward (X)
TORSO = _R_from_cols(UP, RGT)

# Head (mounted on back of head): +z backwards, +x up, +y left
# sensor x -> up (Y), sensor y -> left (Z), sensor z -> backward (-X)
HEAD = _R_from_cols(UP, LFT)

FOOT = _R_from_cols(BWD, RGT)

PELVIS = _R_from_cols(UP, LFT)

MOUNTING_SEG_FROM_SENSOR: Dict[str, R] = {
    "pelvis": PELVIS, 

    "trunk": TORSO,
    "head": HEAD,

    "upper_leg_left": THIGH_L,
    "lower_leg_left": SHANK_L,
    "upper_leg_right": THIGH_R,
    "lower_leg_right": SHANK_R,

    "upper_arm_left": UARM_L,
    "forearm_left": FARM_L,
    "hand_left": HAND_L,

    "upper_arm_right": UARM_R,
    "forearm_right": FARM_R,
    "hand_right": HAND_R,

    "foot_left": FOOT,
    "foot_right": FOOT,
}


def apply_mounting(world_rots: Dict[str, R]) -> Dict[str, R]:
    """Apply mounting corrections to a dict of world-from-sensor rotations."""
    corrected: Dict[str, R] = {}
    for seg, R_w_s in world_rots.items():
        mount = MOUNTING_SEG_FROM_SENSOR.get(seg, R.identity())
        # world-from-seg = world-from-sensor * (seg-from-sensor)^-1
        corrected[seg] = R_w_s * mount.inv()
    return corrected
