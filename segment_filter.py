"""
Segment filtering and neutral pose handling for partial body configurations.
Allows the system to work with subset of sensors while keeping full skeleton visible.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Set, Optional
from config import BODY_MODEL, SEGMENT_NAMES


def get_neutral_orientation() -> R:
    """Get identity rotation for neutral pose."""
    return R.identity()


def get_neutral_quaternion() -> np.ndarray:
    """Get quaternion [w, x, y, z] for neutral orientation."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def fill_missing_segments_with_neutral(
    segment_orientations: Dict[str, R],
    active_segments: Set[str]
) -> Dict[str, R]:
    """
    Fill in missing segments with neutral (identity) orientations.
    
    This allows visualization and kinematics to work with partial sensor setups
    while keeping the full skeleton visible. Inactive segments appear in their
    neutral (T-pose or N-pose) configuration.
    
    Args:
        segment_orientations: Dictionary of available segment orientations
        active_segments: Set of segments that have active sensors
        
    Returns:
        Complete dictionary with all segments (missing ones set to neutral)
    """
    filled = {}
    
    for seg_name in SEGMENT_NAMES:
        if seg_name in segment_orientations:
            # Segment has sensor data - use it
            filled[seg_name] = segment_orientations[seg_name]
        elif seg_name in active_segments:
            # Segment should be active but is missing - use neutral
            # This handles cases where sensor might have dropped out
            filled[seg_name] = get_neutral_orientation()
            print(f"[SegmentFilter] Warning: Active segment '{seg_name}' missing, using neutral pose")
        else:
            # Segment is intentionally inactive (no sensor in this mode) - use neutral
            filled[seg_name] = get_neutral_orientation()
    
    return filled


def filter_joint_angles_by_active_segments(
    joint_data: Dict[str, tuple],
    active_segments: Set[str]
) -> Dict[str, tuple]:
    """
    Filter joint angles to only include joints where both segments are active.
    
    Joints connecting active to inactive segments will show neutral (zero) angles.
    
    Args:
        joint_data: Dictionary of {joint_name: (quaternion, euler_angles)}
        active_segments: Set of segments with active sensors
        
    Returns:
        Filtered dictionary with same structure
    """
    filtered = {}
    
    for joint_name, data in joint_data.items():
        # Parse joint name: "parent_child"
        parts = joint_name.split('_')
        
        # Handle multi-word segment names (e.g., "upper_arm_left")
        # Find the split point by checking against known segments
        parent = None
        child = None
        
        for i in range(1, len(parts)):
            potential_parent = '_'.join(parts[:i])
            potential_child = '_'.join(parts[i:])
            
            if potential_parent in SEGMENT_NAMES and potential_child in SEGMENT_NAMES:
                parent = potential_parent
                child = potential_child
                break
        
        if parent is None or child is None:
            # Couldn't parse - include by default
            filtered[joint_name] = data
            continue
        
        # Check if both segments are active
        if parent in active_segments and child in active_segments:
            # Both active - use real data
            filtered[joint_name] = data
        else:
            # At least one inactive - set to neutral
            neutral_quat = get_neutral_quaternion()
            neutral_euler = np.array([0.0, 0.0, 0.0])
            filtered[joint_name] = (neutral_quat, neutral_euler)
    
    return filtered


def validate_minimum_sensors(
    available_segments: list,
    required_segments: list
) -> tuple[bool, list]:
    """
    Validate that minimum required sensors are present.
    
    Args:
        available_segments: List of segments with sensor data
        required_segments: List of required segments for current body config
        
    Returns:
        (is_valid, missing_segments) tuple
    """
    available = set(available_segments)
    required = set(required_segments)
    missing = required - available
    
    return (len(missing) == 0, sorted(list(missing)))


def get_segment_status_summary(
    available_segments: list,
    active_segments: Set[str]
) -> Dict[str, str]:
    """
    Get human-readable status for each segment.
    
    Args:
        available_segments: Segments with sensor data
        active_segments: Segments that should be active
        
    Returns:
        Dictionary of {segment_name: status_string}
    """
    status = {}
    available = set(available_segments)
    
    for seg_name in SEGMENT_NAMES:
        if seg_name in available and seg_name in active_segments:
            status[seg_name] = "Active (sensor data)"
        elif seg_name in active_segments:
            status[seg_name] = "Missing (sensor required)"
        else:
            status[seg_name] = "Neutral (no sensor in this mode)"
    
    return status


# Example usage in main application:
"""
from body_config import get_body_config
from segment_filter import fill_missing_segments_with_neutral

# Get active segments for current configuration
config = get_body_config(self.recording_ctrl.body_config)
active_segments = config.get_active_segments()

# Fill missing segments before kinematics
world_rots = apply_mounting(raw_sample)  # All available sensors
complete_rots = fill_missing_segments_with_neutral(world_rots, active_segments)

# Now kinematics and visualization get complete skeleton
# - Active segments: use real sensor data
# - Inactive segments: appear in neutral pose
joint_data = self.kinematics.compute_joint_angles(complete_rots)
segment_positions = self.kinematics.forward_kinematics(complete_rots)
"""
