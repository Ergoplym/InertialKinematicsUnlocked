"""
Body configuration system for different sensor setups.
Supports upper body, lower body, and full body tracking modes.
"""

from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class BodyConfiguration:
    """Configuration for a specific body tracking mode."""
    name: str
    display_name: str
    description: str
    required_segments: List[str]  # Segments that MUST have sensors
    num_sensors: int
    
    def get_active_segments(self) -> Set[str]:
        """Get set of segments that are actively tracked in this mode."""
        return set(self.required_segments)
    
    def is_segment_active(self, segment_name: str) -> bool:
        """Check if a segment is active in this configuration."""
        return segment_name in self.required_segments


# Define the three body configurations
BODY_CONFIGS = {
    "whole_body": BodyConfiguration(
        name="whole_body",
        display_name="Whole Body",
        description="Full body tracking - 15 sensors",
        required_segments=[
            "pelvis",           # Root (REQUIRED)
            "trunk",            # Torso
            "head",             # Head
            "upper_arm_right",  # Right upper arm
            "forearm_right",    # Right forearm
            "hand_right",       # Right hand
            "upper_arm_left",   # Left upper arm
            "forearm_left",     # Left forearm
            "hand_left",        # Left hand
            "upper_leg_right",  # Right thigh
            "lower_leg_right",  # Right shin
            "foot_right",       # Right foot
            "upper_leg_left",   # Left thigh
            "lower_leg_left",   # Left shin
            "foot_left",        # Left foot
        ],
        num_sensors=15
    ),
    
    "upper_body": BodyConfiguration(
        name="upper_body",
        display_name="Upper Body Only",
        description="Upper body tracking - 9 sensors (includes pelvis for reference)",
        required_segments=[
            "pelvis",           # Root (REQUIRED - pelvis-centric model)
            "trunk",            # Torso
            "head",             # Head
            "upper_arm_right",  # Right upper arm
            "forearm_right",    # Right forearm
            "hand_right",       # Right hand
            "upper_arm_left",   # Left upper arm
            "forearm_left",     # Left forearm
            "hand_left",        # Left hand
        ],
        num_sensors=9
    ),
    
    "lower_body": BodyConfiguration(
        name="lower_body", 
        display_name="Lower Body Only",
        description="Lower body tracking - 7 sensors",
        required_segments=[
            "pelvis",           # Root (REQUIRED - pelvis-centric model)
            "upper_leg_right",  # Right thigh
            "lower_leg_right",  # Right shin
            "foot_right",       # Right foot
            "upper_leg_left",   # Left thigh
            "lower_leg_left",   # Left shin
            "foot_left",        # Left foot
        ],
        num_sensors=7
    ),
}


# Default configuration
DEFAULT_BODY_CONFIG = "whole_body"


def get_body_config(config_name: str) -> BodyConfiguration:
    """
    Get body configuration by name.
    
    Args:
        config_name: Name of configuration ('whole_body', 'upper_body', 'lower_body')
        
    Returns:
        BodyConfiguration object
        
    Raises:
        ValueError: If config_name is invalid
    """
    if config_name not in BODY_CONFIGS:
        raise ValueError(f"Unknown body configuration: {config_name}. "
                        f"Valid options: {list(BODY_CONFIGS.keys())}")
    return BODY_CONFIGS[config_name]


def get_inactive_segments(config_name: str, all_segments: List[str]) -> Set[str]:
    """
    Get segments that are NOT active in the given configuration.
    
    Args:
        config_name: Name of configuration
        all_segments: List of all possible segment names
        
    Returns:
        Set of inactive segment names
    """
    config = get_body_config(config_name)
    active = config.get_active_segments()
    return set(all_segments) - active


def validate_sensor_data(config_name: str, available_segments: List[str]) -> tuple[bool, List[str]]:
    """
    Validate that sensor data contains all required segments for configuration.
    
    Args:
        config_name: Name of configuration
        available_segments: List of segment names in sensor data
        
    Returns:
        (is_valid, missing_segments) tuple
    """
    config = get_body_config(config_name)
    required = set(config.required_segments)
    available = set(available_segments)
    missing = required - available
    
    return (len(missing) == 0, sorted(list(missing)))


def get_sensor_count_status(detected: int, expected: int) -> tuple[str, str]:
    """
    Get status message and color for sensor count validation.
    
    Args:
        detected: Number of detected sensors
        expected: Number of expected sensors
        
    Returns:
        (status_message, color_code) tuple
    """
    if detected == expected:
        return (f"✓ {detected} / {expected}", "#27ae60")  # Green
    elif detected > expected:
        return (f"✓ {detected} / {expected} (extras)", "#3498db")  # Blue
    else:
        return (f"⚠️ {detected} / {expected}", "#e67e22")  # Orange


# Sensor mapping notes for future virtual scapula implementation
"""
FUTURE: Virtual Scapula Motion

When implementing virtual scapula tracking, the changes should be applied to:
- upper_body configuration (affects shoulder kinematics)
- whole_body configuration (affects shoulder kinematics)
- lower_body configuration (NOT affected - no shoulder tracking)

Implementation approach:
1. Add 'scapula_right' and 'scapula_left' to BODY_MODEL in config.py
2. Update kinematics.py to compute virtual scapula motion from trunk + upper_arm
3. Modify shoulder joint angles to be scapula-relative instead of trunk-relative
4. Update visualization to show virtual scapula (optional, can be hidden)

This will automatically apply to upper_body and whole_body modes since they
share the same upper body segment hierarchy.
"""
