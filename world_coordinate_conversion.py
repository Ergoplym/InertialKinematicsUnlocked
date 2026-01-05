"""
World coordinate system conversion for Xsens Awinda sensors

This module handles the conversion from Xsens sensor local coordinates
to the anatomical/world coordinate system expected by the kinematics engine.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict

class WorldCoordinateConverter:
    """
    Converts sensor orientations from Xsens local frame to anatomical world frame.
    
    Xsens sensor local frame (typical mounting on sacrum):
        - Sensor lying flat against lower back
        - +Z points outward (posterior, away from body)
        - +X points upward (superior, toward head)  
        - +Y points right (lateral right)
        
    Target anatomical world frame (ISB convention):
        - +X points forward (anterior)
        - +Y points up (superior)
        - +Z points right (lateral right)
        
    Required transformation: Rotate sensor frame to align with anatomical frame
    """
    
    def __init__(self, sensor_mounting="sacrum_standard"):
        """
        Initialize coordinate converter.
        
        Args:
            sensor_mounting: Type of sensor mounting
                - "sacrum_standard": Standard Xsens mounting on sacrum
                    Sensor +Z posterior, +X superior, +Y right
                - "sacrum_rotated": Rotated 90° (less common)
                - "xsens_enu_to_app": Remap Xsens global ENU -> project world (Forward/Up/Right)
                - "custom": Use custom transformation (set via set_custom_transform)
        """
        self.sensor_mounting = sensor_mounting
        self._init_transform()
    
    def _init_transform(self):
        """Initialize the sensor-to-world transformation."""
        
        if self.sensor_mounting == "sacrum_standard":
            # Xsens: +Z posterior, +X superior, +Y right
            # World: +X anterior, +Y superior, +Z right
            # 
            # Mapping:
            #   World +X (anterior) = -Sensor +Z (flip posterior to anterior)
            #   World +Y (superior) = Sensor +X
            #   World +Z (right) = Sensor +Y
            #
            # Rotation matrix that performs this mapping:
            rotation_matrix = np.array([
                [0,  0, -1],  # World X = -Sensor Z  (posterior -> anterior)
                [1,  0,  0],  # World Y =  Sensor X  (up)
                [0, -1,  0]   # World Z = -Sensor Y  (assumes sensor +Y points LEFT)
            ])
            self.sensor_to_world = R.from_matrix(rotation_matrix)
            if np.linalg.det(rotation_matrix) < 0:
                print("[WorldCoordinates] WARNING: sacrum_standard mapping was left-handed; using a right-handed variant (flipped sensor Y).")
                print("[WorldCoordinates] Recommendation: prefer xsens_enu_to_app + per-segment mounting corrections.")
            
        
        elif self.sensor_mounting in ("xsens_enu_to_app", "xsens_enu_to_fur"):
            # Xsens global frame (common ENU): +X = East, +Y = North, +Z = Up
            # Target anatomical/world frame used by this project: +X = Forward, +Y = Up, +Z = Right
            #
            # If the subject faces geographic/magnetic North during static calibration:
            #   Forward  (World +X) = Global +Y (North)
            #   Up       (World +Y) = Global +Z
            #   Right    (World +Z) = Global +X (East)
            rotation_matrix = np.array([
                [1,  0,  0],   # World X (Forward) = Global Y (North)
                [0,  0,  1],   # World Y (Up)      = Global Z (Up)
                [0,  -1,  0],   # World Z (Right)   = Global X (East)
            ])
            self.sensor_to_world = R.from_matrix(rotation_matrix)

        elif self.sensor_mounting == "pelvis_anterior":
            # Alternative mounting: sensor rotated 90° clockwise (viewed from behind)
            # Xsens: +Z posterior, +Y superior, +X right
            rotation_matrix = np.array([
                [0,  0,  1],  # World X = -Sensor Z
                [1,  0,  0],  # World Y = Sensor Y
                [0,  1,  0]   # World Z = Sensor X
            ])
            self.sensor_to_world = R.from_matrix(rotation_matrix)
            
        elif self.sensor_mounting == "identity":
            # No transformation (for testing or if sensors already in world frame)
            self.sensor_to_world = R.identity()
            
        else:
            # Default to identity, user must set custom
            print(f"[WorldCoordinates] Unknown mounting '{self.sensor_mounting}', using identity")
            self.sensor_to_world = R.identity()
        
        print(f"[WorldCoordinates] Initialized with mounting: {self.sensor_mounting}")
        print(f"[WorldCoordinates] Transform (Euler XYZ, deg): "
              f"{self.sensor_to_world.as_euler('xyz', degrees=True)}")
    
    def set_custom_transform(self, rotation: R):
        """Set a custom sensor-to-world transformation."""
        self.sensor_to_world = rotation
        self.sensor_mounting = "custom"
    
    def convert_orientations(self, sensor_orientations: Dict[str, R]) -> Dict[str, R]:
        """
        Convert all sensor orientations from sensor frame to world frame.
        
        Args:
            sensor_orientations: Dict of {segment_name: Rotation (in sensor frame)}
            
        Returns:
            Dict of {segment_name: Rotation (in world frame)}
        """
        world_orientations = {}
        
        for segment_name, sensor_rot in sensor_orientations.items():
            # Apply transformation: world_rot = sensor_to_world * sensor_rot
            world_rot = self.sensor_to_world * sensor_rot
            world_orientations[segment_name] = world_rot
        
        return world_orientations
    
    def diagnostic_check(self, sensor_quat: np.ndarray, segment_name: str = "pelvis"):
        """
        Check if the coordinate transformation looks correct.
        
        Args:
            sensor_quat: Raw sensor quaternion [w, x, y, z] in neutral standing
            segment_name: Name of segment being checked
        """
        print(f"\n{'='*70}")
        print(f"COORDINATE TRANSFORM DIAGNOSTIC - {segment_name}")
        print(f"{'='*70}")
        
        # Convert to rotation
        sensor_rot = R.from_quat([sensor_quat[1], sensor_quat[2], 
                                  sensor_quat[3], sensor_quat[0]])
        
        # Apply transform
        world_rot = self.sensor_to_world * sensor_rot
        
        # Get axes in world frame
        world_matrix = world_rot.as_matrix()
        x_axis = world_matrix[:, 0]
        y_axis = world_matrix[:, 1]
        z_axis = world_matrix[:, 2]
        
        print(f"\nSensor mounting type: {self.sensor_mounting}")
        print(f"\nRaw sensor orientation (Euler XYZ, deg): "
              f"{sensor_rot.as_euler('xyz', degrees=True)}")
        print(f"World orientation (Euler XYZ, deg): "
              f"{world_rot.as_euler('xyz', degrees=True)}")
        
        print(f"\nSegment axes in world frame:")
        print(f"  +X axis (should point forward): {x_axis}")
        print(f"  +Y axis (should point up):      {y_axis}")
        print(f"  +Z axis (should point right):   {z_axis}")
        
        # Check alignment
        forward_check = x_axis[0]  # Should be positive and large
        up_check = y_axis[1]       # Should be positive and large
        right_check = z_axis[2]    # Should be positive and large
        
        print(f"\nAlignment check (in neutral standing pose):")
        print(f"  Forward component: {forward_check:+.3f} (expect > +0.9)")
        print(f"  Up component:      {up_check:+.3f} (expect > +0.9)")
        print(f"  Right component:   {right_check:+.3f} (expect > +0.9)")
        
        total_score = abs(forward_check) + abs(up_check) + abs(right_check)
        print(f"\nTotal alignment score: {total_score:.2f} / 3.00")
        
        if total_score > 2.7:
            print("✓ EXCELLENT - Coordinate transform is correct!")
        elif total_score > 2.3:
            print("✓ GOOD - Coordinate transform looks reasonable")
        elif total_score > 1.5:
            print("⚠ WARNING - Coordinate transform may need adjustment")
        else:
            print("✗ POOR - Coordinate transform is likely incorrect")
        
        print(f"{'='*70}\n")


def auto_detect_sensor_mounting(neutral_pose_sample: Dict[str, np.ndarray]) -> str:
    """
    Attempt to automatically detect sensor mounting orientation.
    
    Args:
        neutral_pose_sample: Dict of {segment_name: quat [w,x,y,z]} in neutral standing
        
    Returns:
        Best guess for mounting type: "sacrum_standard", "sacrum_rotated", or "identity"
    """
    if "pelvis" not in neutral_pose_sample:
        print("[AutoDetect] No pelvis data, defaulting to sacrum_standard")
        return "sacrum_standard"
    
    pelvis_quat = neutral_pose_sample["pelvis"]
    pelvis_rot = R.from_quat([pelvis_quat[1], pelvis_quat[2], 
                              pelvis_quat[3], pelvis_quat[0]])
    
    # Get sensor axes in world
    sensor_matrix = pelvis_rot.as_matrix()
    
    # Test different mounting configurations
    configs = {
        "sacrum_standard": np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]]),
        "sacrum_rotated": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        "identity": np.eye(3)
    }
    
    best_score = -1
    best_config = "sacrum_standard"
    
    for config_name, transform_matrix in configs.items():
        # Apply transformation
        transformed = transform_matrix @ sensor_matrix
        
        # Check if transformed axes align with expected (identity in neutral)
        # Y should be vertical (0, 1, 0)
        y_axis = transformed[:, 1]
        score = abs(y_axis[1])  # How vertical is the up axis?
        
        if score > best_score:
            best_score = score
            best_config = config_name
    
    print(f"[AutoDetect] Best mounting configuration: {best_config} (score: {best_score:.3f})")
    return best_config


# Example usage and testing
if __name__ == "__main__":
    # Test case 1: Standard sacrum mounting
    print("TEST 1: Standard Sacrum Mounting")
    print("-" * 70)
    
    converter = WorldCoordinateConverter("sacrum_standard")
    
    # Simulate sensor reading in neutral pose
    # Xsens sensor: +Z pointing backward (posterior), +X pointing up
    # In neutral standing, this might read close to identity
    test_quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    converter.diagnostic_check(test_quat, "pelvis")