"""
Dual-Pose Static Calibration

Uses N-pose + T-pose to compute better sensor alignments.
Much simpler than functional calibration, better than single-pose.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Tuple, List


class DualPoseCalibration:
    """
    Calibrate using two static poses: N-pose and T-pose.
    
    Workflow:
    1. Capture N-pose (arms at sides)
    2. Capture T-pose (arms extended laterally)
    3. Compute alignments that satisfy BOTH constraints
    4. Compute offsets with alignments applied
    """
    
    def __init__(self):
        self.n_pose_sample = None
        self.t_pose_sample = None
        self.alignments = {}
        self.offsets = {}
        self.is_complete = False
        
        # Expected angles for each pose
        self.n_pose_expected = self._get_n_pose_expected()
        self.t_pose_expected = self._get_t_pose_expected()
    
    def _get_n_pose_expected(self) -> Dict[str, np.ndarray]:
        """Expected joint angles in N-pose (all zeros)."""
        return {
            # Upper limbs - all neutral
            'upper_arm_right': np.array([0, 0, 0]),
            'upper_arm_left': np.array([0, 0, 0]),
            'forearm_right': np.array([0, 0, 0]),
            'forearm_left': np.array([0, 0, 0]),
            'hand_right': np.array([0, 0, 0]),
            'hand_left': np.array([0, 0, 0]),
            
            # Lower limbs - all neutral
            'upper_leg_right': np.array([0, 0, 0]),
            'upper_leg_left': np.array([0, 0, 0]),
            'lower_leg_right': np.array([0, 0, 0]),
            'lower_leg_left': np.array([0, 0, 0]),
            'foot_right': np.array([0, 0, 0]),
            'foot_left': np.array([0, 0, 0]),
            
            # Trunk/head - all neutral
            'trunk': np.array([0, 0, 0]),
            'head': np.array([0, 0, 0]),
        }
    
    def _get_t_pose_expected(self) -> Dict[str, np.ndarray]:
        """Expected joint angles in T-pose."""
        return {
            # Upper limbs - arms extended laterally
            # Using ZXY for all joints: [Flex/Ext, Abd/Add, Axial Rot]
            # In T-pose: 0° flex/ext, 90° abduction, 0° axial
            'upper_arm_right': np.array([0, 90, 0]),   # ZXY: 0° flex, 90° abduction, 0° axial
            'upper_arm_left': np.array([0, 90, 0]),    # ZXY: 0° flex, 90° abduction, 0° axial
            
            # Elbows use ZXY: [Flex/Ext, Carry, Pron/Sup]
            'forearm_right': np.array([0, 0, 0]),      # Straight elbow
            'forearm_left': np.array([0, 0, 0]),       # Straight elbow
            
            # Wrists use ZXY: [Flex/Ext, Rad/Uln, Rot]
            'hand_right': np.array([0, 0, 0]),         # Neutral wrist
            'hand_left': np.array([0, 0, 0]),          # Neutral wrist
            
            # Lower limbs - all use ZXY
            'upper_leg_right': np.array([0, 0, 0]),
            'upper_leg_left': np.array([0, 0, 0]),
            'lower_leg_right': np.array([0, 0, 0]),
            'lower_leg_left': np.array([0, 0, 0]),
            'foot_right': np.array([0, 0, 0]),
            'foot_left': np.array([0, 0, 0]),
            
            # Trunk/head - use ZXY
            'trunk': np.array([0, 0, 0]),
            'head': np.array([0, 0, 0]),
        }
    
    def capture_n_pose(self, sensor_orientations: Dict[str, R]):
        """Capture N-pose sample."""
        # Convert to pelvis-relative frame
        self.n_pose_sample = self._to_pelvis_relative(sensor_orientations)
        print("[DualCalib] N-pose captured")
    
    def capture_t_pose(self, sensor_orientations: Dict[str, R]):
        """Capture T-pose sample."""
        # Convert to pelvis-relative frame
        self.t_pose_sample = self._to_pelvis_relative(sensor_orientations)
        print("[DualCalib] T-pose captured")
    
    def _to_pelvis_relative(self, sensor_orientations: Dict[str, R]) -> Dict[str, R]:
        """Convert world-frame sensors to pelvis-relative frame."""
        pelvis = sensor_orientations['pelvis']
        
        relative = {}
        for seg, orient in sensor_orientations.items():
            if seg == 'pelvis':
                relative[seg] = R.identity()
            else:
                relative[seg] = pelvis.inv() * orient
        
        return relative
    
    def compute_calibration(self) -> Tuple[Dict[str, R], Dict[str, R]]:
        """
        Compute alignments and offsets from both poses.
        
        Returns:
            (alignments, offsets) - both as Dict[str, R]
        """
        if self.n_pose_sample is None or self.t_pose_sample is None:
            raise ValueError("Must capture both N-pose and T-pose first")
        
        print("[DualCalib] Computing dual-pose calibration...")
        
        # Compute alignments for each segment
        self.alignments = self._compute_alignments()
        
        # Compute offsets using N-pose with alignments applied
        # (Use N-pose as the "neutral" reference)
        self.offsets = self._compute_offsets_from_n_pose()
        
        self.is_complete = True
        print("[DualCalib] Calibration complete!")
        
        return self.alignments, self.offsets
    
    def _compute_alignments(self) -> Dict[str, R]:
        """Compute alignment for each segment using dual constraints."""
        alignments = {}
        
        # Pelvis is always identity
        alignments['pelvis'] = R.identity()
        
        # Define parent relationships
        parents = self._get_parent_map()
        
        # Compute alignment for each segment
        for segment in self.n_pose_sample.keys():
            if segment == 'pelvis':
                continue
            
            parent = parents.get(segment, 'pelvis')
            
            # Get expected angles for this segment
            n_expected = self.n_pose_expected.get(segment, np.array([0, 0, 0]))
            t_expected = self.t_pose_expected.get(segment, np.array([0, 0, 0]))
            
            # Find best alignment
            best_alignment = self._find_best_alignment(
                segment, parent, n_expected, t_expected
            )
            
            alignments[segment] = best_alignment
        
        return alignments
    
    def _find_best_alignment(self, segment: str, parent: str,
                            n_expected: np.ndarray, t_expected: np.ndarray) -> R:
        """Find alignment that best satisfies both N-pose and T-pose constraints."""
        
        # Get Euler sequence for this joint
        euler_seq = self._get_euler_sequence(segment)
        
        # Step 1: Discrete search to find good starting point
        candidates = self._generate_alignment_candidates()
        
        best_alignment = R.identity()
        best_score = -np.inf
        
        for name, alignment in candidates:
            # === N-POSE ERROR ===
            parent_n = self.n_pose_sample[parent]
            child_n = self.n_pose_sample[segment]
            
            rel_n = parent_n.inv() * (child_n * alignment)
            angles_n = rel_n.as_euler(euler_seq, degrees=True)
            error_n = np.linalg.norm(angles_n - n_expected)
            
            # === T-POSE ERROR ===
            parent_t = self.t_pose_sample[parent]
            child_t = self.t_pose_sample[segment]
            
            rel_t = parent_t.inv() * (child_t * alignment)
            angles_t = rel_t.as_euler(euler_seq, degrees=True)
            error_t = np.linalg.norm(angles_t - t_expected)
            
            # === COMBINED SCORE ===
            total_error = error_n + error_t
            score = -total_error  # Negative because lower error is better
            
            if score > best_score:
                best_score = score
                best_alignment = alignment
        
        discrete_error = -best_score
        print(f"[DualCalib] {segment}: discrete error={discrete_error:.2f}°", end="")
        
        # Step 2: Continuous optimization around best discrete candidate
        optimized_alignment = self._optimize_alignment_continuous(
            best_alignment, discrete_error, segment, parent, n_expected, t_expected, euler_seq
        )
        
        return optimized_alignment
    
    def _optimize_alignment_continuous(self, initial_alignment: R, discrete_error: float,
                                      segment: str, parent: str,
                                      n_expected: np.ndarray, t_expected: np.ndarray,
                                      euler_seq: str) -> R:
        """
        Continuous optimization of alignment around initial guess.
        
        Uses scipy.optimize to find the optimal rotation.
        """
        from scipy.optimize import minimize
        
        # Get samples
        parent_n = self.n_pose_sample[parent]
        child_n = self.n_pose_sample[segment]
        parent_t = self.t_pose_sample[parent]
        child_t = self.t_pose_sample[segment]
        
        # Objective function: minimize total angle error
        def objective(rotvec):
            """Compute error for a rotation vector."""
            # Convert rotation vector to rotation
            alignment = R.from_rotvec(rotvec)
            
            # N-pose error
            rel_n = parent_n.inv() * (child_n * alignment)
            angles_n = rel_n.as_euler(euler_seq, degrees=True)
            error_n = np.linalg.norm(angles_n - n_expected)
            
            # T-pose error
            rel_t = parent_t.inv() * (child_t * alignment)
            angles_t = rel_t.as_euler(euler_seq, degrees=True)
            error_t = np.linalg.norm(angles_t - t_expected)
            
            return error_n + error_t
        
        # Initial guess: rotation vector of best discrete alignment
        x0 = initial_alignment.as_rotvec()
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='Powell',  # Powell works well for rotation optimization
            options={'maxiter': 100, 'ftol': 0.01}  # Stop at 0.01° tolerance
        )
        
        # Get optimized alignment
        optimized_alignment = R.from_rotvec(result.x)
        optimized_error = result.fun
        
        improvement = discrete_error - optimized_error
        print(f" → optimized error={optimized_error:.2f}° (improvement: {improvement:.2f}°)")
        
        return optimized_alignment
    
    def _get_euler_sequence(self, segment: str) -> str:
        """Get the Euler sequence for a segment."""
        # Use ZXY for all segments (works better than YXY for shoulders)
        return 'ZXY'
    
    def _compute_offsets_from_n_pose(self) -> Dict[str, R]:
        """Compute offsets from N-pose with alignments applied."""
        offsets = {}
        
        # Pelvis offset (world frame)
        pelvis_n = self.n_pose_sample['pelvis']  # This is identity in pelvis frame
        # Need actual world-frame pelvis for pelvis offset
        # For now, assume identity (will be set properly by main calibration)
        offsets['pelvis'] = R.identity()
        
        # Other segments: offset in pelvis-relative frame
        for segment, orient_rel in self.n_pose_sample.items():
            if segment == 'pelvis':
                continue
            
            # Apply alignment
            aligned_rel = orient_rel * self.alignments[segment]
            
            # Offset is inverse (to bring to neutral)
            offsets[segment] = aligned_rel.inv()
        
        return offsets
    
    def _get_parent_map(self) -> Dict[str, str]:
        """Get parent segment for each segment."""
        return {
            # Upper limbs
            'upper_arm_right': 'trunk',
            'upper_arm_left': 'trunk',
            'forearm_right': 'upper_arm_right',
            'forearm_left': 'upper_arm_left',
            'hand_right': 'forearm_right',
            'hand_left': 'forearm_left',
            
            # Lower limbs
            'upper_leg_right': 'pelvis',
            'upper_leg_left': 'pelvis',
            'lower_leg_right': 'upper_leg_right',
            'lower_leg_left': 'upper_leg_left',
            'foot_right': 'lower_leg_right',
            'foot_left': 'lower_leg_left',
            
            # Trunk/head
            'trunk': 'pelvis',
            'head': 'trunk',
        }
    
    def _generate_alignment_candidates(self) -> List[Tuple[str, R]]:
        """Generate candidate alignment rotations."""
        candidates = []
        
        # Identity
        candidates.append(('identity', R.identity()))
        
        # 180° flips around each axis
        for axis in ['x', 'y', 'z']:
            candidates.append((f'{axis}180', R.from_euler(axis, 180, degrees=True)))
        
        # 90° rotations around each axis
        for axis in ['x', 'y', 'z']:
            for angle in [90, -90]:
                candidates.append((f'{axis}{angle:+d}', R.from_euler(axis, angle, degrees=True)))
        
        # Common combinations
        candidates.append(('xy180', R.from_euler('xy', [180, 180], degrees=True)))
        candidates.append(('xz180', R.from_euler('xz', [180, 180], degrees=True)))
        candidates.append(('yz180', R.from_euler('yz', [180, 180], degrees=True)))
        
        # Fine adjustments (±15°, ±30° around each axis)
        for axis in ['x', 'y', 'z']:
            for angle in [-30, -15, 15, 30]:
                candidates.append((f'{axis}{angle:+d}', R.from_euler(axis, angle, degrees=True)))
        
        return candidates
    
    def reset(self):
        """Reset calibration state."""
        self.n_pose_sample = None
        self.t_pose_sample = None
        self.alignments = {}
        self.offsets = {}
        self.is_complete = False
        print("[DualCalib] Reset")
