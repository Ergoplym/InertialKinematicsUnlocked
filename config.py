"""Configuration and body model definitions"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class JointType(Enum):
    """Joint types with expected degrees of freedom"""
    HINGE = "hinge"      # 1 DOF (knee, elbow)
    BALL = "ball"        # 3 DOF (hip, shoulder)
    SADDLE = "saddle"    # 2 DOF primary (wrist, ankle)


@dataclass
class SegmentDef:
    """Definition of a body segment"""
    name: str
    parent: Optional[str]  # None for root (pelvis)
    joint_type: Optional[JointType]
    length: float  # meters
    
    # ISB-recommended Euler sequence for this joint (for reference/debugging)
    euler_seq: str = "ZXY"
    
    # Anatomical axis labels (what each Euler angle represents)
    axis_labels: Tuple[str, str, str] = ("Flex/Ext", "Add/Abd", "Rot")


# Body model - hierarchical segment tree
BODY_MODEL = {
    # Trunk
    "pelvis": SegmentDef("pelvis", None, None, 0.0),
    "trunk": SegmentDef("trunk", "pelvis", JointType.BALL, 0.50, "ZXY", 
                        ("Flex/Ext", "Lat Bend", "Axial Rot")),
    "head": SegmentDef("head", "trunk", JointType.BALL, 0.25, "ZXY",
                       ("Flex/Ext", "Lat Bend", "Axial Rot")),
    
    # Right arm
    "upper_arm_right": SegmentDef("upper_arm_right", "trunk", JointType.BALL, 0.30, "ZXY",
                                  ("Plane Elev", "Elevation", "Axial Rot")),
    "forearm_right": SegmentDef("forearm_right", "upper_arm_right", JointType.HINGE, 0.25, "ZXY",
                                ("Flex/Ext", "Carry", "Pron/Sup")),
    "hand_right": SegmentDef("hand_right", "forearm_right", JointType.SADDLE, 0.18, "ZXY",
                             ("Flex/Ext", "Rad/Uln", "Rot")),
    
    # Left arm
    "upper_arm_left": SegmentDef("upper_arm_left", "trunk", JointType.BALL, 0.30, "ZXZ",
                                 ("Plane Elev", "Elevation", "Axial Rot")),
    "forearm_left": SegmentDef("forearm_left", "upper_arm_left", JointType.HINGE, 0.25, "ZXY",
                               ("Flex/Ext", "Carry", "Pron/Sup")),
    "hand_left": SegmentDef("hand_left", "forearm_left", JointType.SADDLE, 0.18, "ZXY",
                            ("Flex/Ext", "Rad/Uln", "Rot")),
    
    # Right leg
    "upper_leg_right": SegmentDef("upper_leg_right", "pelvis", JointType.BALL, 0.40, "ZXY",
                                  ("Flex/Ext", "Add/Abd", "Int/Ext Rot")),
    "lower_leg_right": SegmentDef("lower_leg_right", "upper_leg_right", JointType.HINGE, 0.40, "ZXY",
                                  ("Flex/Ext", "Var/Valg", "Tib Rot")),
    "foot_right": SegmentDef("foot_right", "lower_leg_right", JointType.SADDLE, 0.25, "ZXY",
                             ("Dorsi/Plant", "Inv/Ev", "Abd/Add")),
    
    # Left leg
    "upper_leg_left": SegmentDef("upper_leg_left", "pelvis", JointType.BALL, 0.40, "ZXY",
                                 ("Flex/Ext", "Add/Abd", "Int/Ext Rot")),
    "lower_leg_left": SegmentDef("lower_leg_left", "upper_leg_left", JointType.HINGE, 0.40, "ZXY",
                                 ("Flex/Ext", "Var/Valg", "Tib Rot")),
    "foot_left": SegmentDef("foot_left", "lower_leg_left", JointType.SADDLE, 0.25, "ZXY",
                            ("Dorsi/Plant", "Inv/Ev", "Abd/Add")),
}


SEGMENT_NAMES = list(BODY_MODEL.keys())

# Display settings
UPDATE_RATE_HZ = 60.0     # IMU data processing rate
DISPLAY_RATE_HZ = 30.0    # Visualization update rate
ANGLE_BUFFER_LEN = 2000   # Number of data points to keep in CSV recording
STATIC_CALIB_DELAY = 5.0  # seconds for static calibration countdown
MOVEMENT_PAUSE = 5.0      # seconds pause between movements

# Visualization settings (OPTIMIZED)
# Using table instead of real-time plots dramatically improves performance
# - Table updates are 10-20x faster than plot updates
# - Matplotlib 3D uses persistent artists for efficient updates
# - All data still recorded to CSV for post-hoc plotting

# Rotation representation
USE_QUATERNIONS = False  # True = use quaternions, False = use Euler angles
QUATERNION_FORMAT = "WXYZ"  # Scalar-first convention [w, x, y, z]

# Notes on quaternion usage:
# - Quaternions are stored in [w, x, y, z] format (scalar-first)
# - This avoids gimbal lock issues inherent in Euler angles
# - Euler angles are still computed for debugging/reference
# - CSV exports include both quaternions and Euler angles for compatibility
