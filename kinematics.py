"""Kinematics engine for joint angle computation using quaternions"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Tuple
from config import BODY_MODEL


class KinematicsEngine:
    """Computes joint rotations (as quaternions) from calibrated segment orientations"""
    
    def __init__(self, body_model: Dict):
        self.body_model = body_model
        # Store as quaternions [w, x, y, z]
        self.joint_quaternions: Dict[str, np.ndarray] = {}
        # Also provide Euler angles for backward compatibility/debugging
        self.joint_angles_euler: Dict[str, np.ndarray] = {}
    
    def _extract_angles_from_matrix(self, rel_rot: R, euler_seq: str = "ZXY") -> np.ndarray:
        """
        Extract joint angles from relative rotation in a rotation-invariant way.
        
        This avoids gimbal lock issues by using direct rotation matrix decomposition
        instead of standard Euler angle extraction.
        
        Args:
            rel_rot: Relative rotation from parent to child segment
            euler_seq: Euler sequence (determines order of rotations)
            
        Returns:
            [angle0, angle1, angle2] in degrees, corresponding to the Euler sequence
        """
        # Get rotation matrix
        R_mat = rel_rot.as_matrix()
        
        # Extract angles based on Euler sequence
        # Using robust matrix decomposition to avoid gimbal lock
        
        if euler_seq == "ZXY":
            # Rotation order: Z (first) -> X (second) -> Y (third)
            # Common for most joints (hip, knee, ankle, trunk, etc.)
            angle0 = np.arctan2(R_mat[0, 1], R_mat[1, 1]) * 180.0 / np.pi  # Z rotation
            angle1 = np.arcsin(-R_mat[2, 1]) * 180.0 / np.pi                # X rotation
            angle2 = np.arctan2(R_mat[2, 0], R_mat[2, 2]) * 180.0 / np.pi  # Y rotation
            
        elif euler_seq == "YXY":
            # Rotation order: Y -> X -> Y
            # Used for shoulder joints
            angle0 = np.arctan2(R_mat[0, 1], -R_mat[2, 1]) * 180.0 / np.pi  # Y rotation (plane of elevation)
            angle1 = np.arccos(R_mat[1, 1]) * 180.0 / np.pi                  # X rotation (elevation)
            angle2 = np.arctan2(R_mat[1, 0], R_mat[1, 2]) * 180.0 / np.pi   # Y rotation (axial rotation)
            
        elif euler_seq == "XYZ":
            # Rotation order: X -> Y -> Z
            angle0 = np.arctan2(-R_mat[1, 2], R_mat[2, 2]) * 180.0 / np.pi  # X rotation
            angle1 = np.arcsin(R_mat[0, 2]) * 180.0 / np.pi                  # Y rotation
            angle2 = np.arctan2(-R_mat[0, 1], R_mat[0, 0]) * 180.0 / np.pi  # Z rotation
            
        elif euler_seq == "ZYX":
            # Rotation order: Z -> Y -> X
            angle0 = np.arctan2(R_mat[1, 0], R_mat[0, 0]) * 180.0 / np.pi   # Z rotation
            angle1 = np.arcsin(-R_mat[2, 0]) * 180.0 / np.pi                 # Y rotation
            angle2 = np.arctan2(R_mat[2, 1], R_mat[2, 2]) * 180.0 / np.pi   # X rotation
            
        elif euler_seq == "XZY":
            # Rotation order: X -> Z -> Y
            angle0 = np.arctan2(R_mat[2, 1], R_mat[1, 1]) * 180.0 / np.pi   # X rotation
            angle1 = np.arcsin(-R_mat[0, 1]) * 180.0 / np.pi                 # Z rotation
            angle2 = np.arctan2(R_mat[0, 2], R_mat[0, 0]) * 180.0 / np.pi   # Y rotation
            
        elif euler_seq == "YZX":
            # Rotation order: Y -> Z -> X
            angle0 = np.arctan2(-R_mat[2, 0], R_mat[0, 0]) * 180.0 / np.pi  # Y rotation
            angle1 = np.arcsin(R_mat[1, 0]) * 180.0 / np.pi                  # Z rotation
            angle2 = np.arctan2(-R_mat[1, 2], R_mat[1, 1]) * 180.0 / np.pi  # X rotation
            
        else:
            # Fallback to scipy's method for unknown sequences
            # This may still have gimbal lock issues but maintains compatibility
            return rel_rot.as_euler(euler_seq, degrees=True)
        
        return np.array([angle0, angle1, angle2])
    
    def compute_joint_rotations(self, segment_orientations: Dict[str, R]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute all joint rotations from segment orientations.
        
        Returns: {joint_name: (quaternion [w,x,y,z], euler_angles [deg])}
        """
        self.joint_quaternions = {}
        self.joint_angles_euler = {}
        
        for seg_name, seg_def in self.body_model.items():
            if seg_def.parent is None:
                continue  # Skip root
            
            parent_name = seg_def.parent
            child_name = seg_name
            
            # Relative rotation: parent -> child
            parent_rot = segment_orientations[parent_name]
            child_rot = segment_orientations[child_name]
            rel_rot = parent_rot.inv() * child_rot
            
            # Store as quaternion [w, x, y, z] (scalar-first convention)
            quat = rel_rot.as_quat()  # Returns [x, y, z, w]
            quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # Reorder to [w, x, y, z]
            
            # Compute Euler angles using rotation-invariant matrix decomposition
            # This avoids gimbal lock for ALL joints
            angles = self._extract_angles_from_matrix(rel_rot, seg_def.euler_seq)
            
            # Store both representations
            joint_name = f"{parent_name}_{child_name}"
            self.joint_quaternions[joint_name] = quat_wxyz
            self.joint_angles_euler[joint_name] = np.array(angles)
        
        return {k: (self.joint_quaternions[k], self.joint_angles_euler[k]) 
                for k in self.joint_quaternions.keys()}
    
    def get_joint_quaternions(self) -> Dict[str, np.ndarray]:
        """Get joint rotations as quaternions [w, x, y, z]"""
        return self.joint_quaternions.copy()
    
    def get_joint_euler_angles(self) -> Dict[str, np.ndarray]:
        """Get joint rotations as Euler angles (for backward compatibility)"""
        return self.joint_angles_euler.copy()
    
    def quaternion_to_euler(self, quat_wxyz: np.ndarray, euler_seq: str = 'ZXY') -> np.ndarray:
        """
        Convert quaternion [w, x, y, z] to Euler angles.
        
        Args:
            quat_wxyz: Quaternion in [w, x, y, z] format
            euler_seq: Euler sequence (default 'ZXY')
        
        Returns:
            Euler angles in degrees
        """
        # Convert back to scipy format [x, y, z, w]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot = R.from_quat(quat_xyzw)
        return rot.as_euler(euler_seq, degrees=True)
    
    def euler_to_quaternion(self, angles: np.ndarray, euler_seq: str = 'ZXY') -> np.ndarray:
        """
        Convert Euler angles to quaternion [w, x, y, z].
        
        Args:
            angles: Euler angles in degrees
            euler_seq: Euler sequence (default 'ZXY')
        
        Returns:
            Quaternion in [w, x, y, z] format
        """
        rot = R.from_euler(euler_seq, angles, degrees=True)
        quat_xyzw = rot.as_quat()  # [x, y, z, w]
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    def compute_segment_positions(self, segment_orientations: Dict[str, R]) -> Dict[str, np.ndarray]:
        """Compute 3D positions of segments using forward kinematics."""
        positions = {}
        
        # Pelvis at origin (standing height)
        pelvis_pos = np.array([0.0, 1.0, 0.0])
        positions["pelvis"] = pelvis_pos
        
        # Get rotations
        pelvis_rot = segment_orientations["pelvis"]
        trunk_rot = segment_orientations["trunk"]
        head_rot = segment_orientations["head"]
        
        # Trunk and head (up the spine)
        trunk_length = self.body_model["trunk"].length
        head_length = self.body_model["head"].length
        
        trunk_pos = pelvis_pos + trunk_rot.apply([0, trunk_length, 0])
        head_pos = trunk_pos + head_rot.apply([0, head_length, 0])
        
        positions["trunk"] = trunk_pos
        positions["head"] = head_pos
        
        # Head tip (visual cue for head rotation)
        head_tip_len = 0.15
        head_tip_pos = head_pos + head_rot.apply([head_tip_len, 0, 0])
        positions["head_tip"] = head_tip_pos

        # Shoulder bar (transverse) at top of trunk
        shoulder_width = 0.20
        shoulder_r_pos = trunk_pos + trunk_rot.apply([0, 0, shoulder_width])
        shoulder_l_pos = trunk_pos + trunk_rot.apply([0, 0, -shoulder_width])

        positions["trunk_tip_right"] = shoulder_r_pos
        positions["trunk_tip_left"] = shoulder_l_pos
        
        # Right arm
        upper_arm_r_rot = segment_orientations["upper_arm_right"]
        forearm_r_rot = segment_orientations["forearm_right"]
        hand_r_rot = segment_orientations["hand_right"]
        
        upper_arm_r_len = self.body_model["upper_arm_right"].length
        forearm_r_len = self.body_model["forearm_right"].length
        hand_r_len = self.body_model["hand_right"].length
        
        elbow_r_pos = shoulder_r_pos + upper_arm_r_rot.apply([0, -upper_arm_r_len, 0])
        wrist_r_pos = elbow_r_pos + forearm_r_rot.apply([0, -forearm_r_len, 0])
        hand_r_pos = wrist_r_pos + hand_r_rot.apply([0, -hand_r_len, 0])
        
        positions["upper_arm_right"] = elbow_r_pos
        positions["forearm_right"] = wrist_r_pos
        positions["hand_right"] = hand_r_pos
        
        # Left arm
        upper_arm_l_rot = segment_orientations["upper_arm_left"]
        forearm_l_rot = segment_orientations["forearm_left"]
        hand_l_rot = segment_orientations["hand_left"]
        
        upper_arm_l_len = self.body_model["upper_arm_left"].length
        forearm_l_len = self.body_model["forearm_left"].length
        hand_l_len = self.body_model["hand_left"].length
        
        elbow_l_pos = shoulder_l_pos + upper_arm_l_rot.apply([0, -upper_arm_l_len, 0])
        wrist_l_pos = elbow_l_pos + forearm_l_rot.apply([0, -forearm_l_len, 0])
        hand_l_pos = wrist_l_pos + hand_l_rot.apply([0, -hand_l_len, 0])
        
        positions["upper_arm_left"] = elbow_l_pos
        positions["forearm_left"] = wrist_l_pos
        positions["hand_left"] = hand_l_pos
        
        # Hip offsets
        hip_width = 0.18
        hip_r_pos = pelvis_pos + pelvis_rot.apply([0, 0, hip_width])
        hip_l_pos = pelvis_pos + pelvis_rot.apply([0, 0, -hip_width])

        positions["hip_right"] = hip_r_pos
        positions["hip_left"] = hip_l_pos
        
        # Right leg (FIXED FK: chain translations in pelvis frame)
        upper_leg_r_rot = segment_orientations["upper_leg_right"]
        lower_leg_r_rot = segment_orientations["lower_leg_right"]
        foot_r_rot = segment_orientations["foot_right"]

        upper_leg_r_len = self.body_model["upper_leg_right"].length
        lower_leg_r_len = self.body_model["lower_leg_right"].length
        foot_r_len = self.body_model["foot_right"].length

        # Relative rotations
        femur_r_rel = pelvis_rot.inv() * upper_leg_r_rot
        shank_r_rel = upper_leg_r_rot.inv() * lower_leg_r_rot
        foot_r_rel = lower_leg_r_rot.inv() * foot_r_rot

        # Knee (hip -> knee in pelvis frame)
        knee_r_pos = hip_r_pos + pelvis_rot.apply(
            femur_r_rel.apply([0, -upper_leg_r_len, 0])
        )

        # Ankle (knee -> ankle, still chained via pelvis frame)
        ankle_r_pos = knee_r_pos + pelvis_rot.apply(
            femur_r_rel.apply(
                shank_r_rel.apply([0, -lower_leg_r_len, 0])
            )
        )

        # Foot tip (ankle -> foot tip; keep your forward tip along +X of foot)
        # FIXED: Use absolute foot orientation to prevent error accumulation during body rotation
        foot_r_pos = ankle_r_pos + foot_r_rot.apply([foot_r_len, 0, 0])

        positions["upper_leg_right"] = knee_r_pos
        positions["lower_leg_right"] = ankle_r_pos
        positions["foot_right"] = foot_r_pos
        
        # Left leg (FIXED FK: chain translations in pelvis frame)
        upper_leg_l_rot = segment_orientations["upper_leg_left"]
        lower_leg_l_rot = segment_orientations["lower_leg_left"]
        foot_l_rot = segment_orientations["foot_left"]

        upper_leg_l_len = self.body_model["upper_leg_left"].length
        lower_leg_l_len = self.body_model["lower_leg_left"].length
        foot_l_len = self.body_model["foot_left"].length

        # Relative rotations
        femur_l_rel = pelvis_rot.inv() * upper_leg_l_rot
        shank_l_rel = upper_leg_l_rot.inv() * lower_leg_l_rot
        foot_l_rel = lower_leg_l_rot.inv() * foot_l_rot

        knee_l_pos = hip_l_pos + pelvis_rot.apply(
            femur_l_rel.apply([0, -upper_leg_l_len, 0])
        )

        ankle_l_pos = knee_l_pos + pelvis_rot.apply(
            femur_l_rel.apply(
                shank_l_rel.apply([0, -lower_leg_l_len, 0])
            )
        )

        # FIXED: Use absolute foot orientation to prevent error accumulation during body rotation
        foot_l_pos = ankle_l_pos + foot_l_rot.apply([foot_l_len, 0, 0])

        positions["upper_leg_left"] = knee_l_pos
        positions["lower_leg_left"] = ankle_l_pos
        positions["foot_left"] = foot_l_pos
        
        return positions
