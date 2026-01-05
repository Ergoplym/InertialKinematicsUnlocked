"""
Pelvis-Relative Functional Calibration System

Computes sensor-to-segment alignments that work with pelvis-relative calibration.
All samples and computations are done in pelvis-relative frame.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import SEGMENT_NAMES


@dataclass
class MovementDefinition:
    """Definition of a calibration movement."""
    name: str
    description: str
    instruction: str
    duration: float  # seconds
    pause_after: float  # seconds
    target_segments: List[str]
    method: str  # calibration method to use


class PelvisRelativeFunctionalCalibration:
    """
    Functional calibration designed for pelvis-relative calibration system.
    
    Key Differences from Standard Functional Calibration:
    1. All samples are stored in PELVIS-RELATIVE frame
    2. Alignments are computed for pelvis-relative rotations
    3. Parent segments are always considered in computations
    
    Movement Sequence:
    1. Squats - Lower limb (bilateral symmetry)
    2. Arms Forward + Sideways - Shoulders (sequential planar)
    3. Elbow Flexion - Elbows (bilateral hinge)
    4. Forearm Rotation - Forearms (bilateral axial)
    5. Trunk Rotation - Trunk (principal axis)
    6. Head Rotation - Head (principal axis)
    
    Total time: ~80 seconds
    """
    
    def __init__(self):
        self.movements = [
            MovementDefinition(
                name="squat",
                description="Deep Squats",
                instruction="Feet shoulder-width apart. Perform 5 slow, DEEP squats. Go down as far as comfortable.",
                duration=10.0,
                pause_after=4.0,
                target_segments=["pelvis", "upper_leg_right", "upper_leg_left",
                               "lower_leg_right", "lower_leg_left", "foot_right", "foot_left"],
                method="bilateral_symmetry"
            ),
            MovementDefinition(
                name="arm_planar",
                description="Arm Forward + Sideways",
                instruction="Arms at sides. First: Raise arms FORWARD to shoulder height 3 times. Then: Raise arms OUT TO SIDES to shoulder height 3 times.",
                duration=12.0,
                pause_after=4.0,
                target_segments=["upper_arm_right", "upper_arm_left"],
                method="shoulder_planar"
            ),
            MovementDefinition(
                name="elbow_flex",
                description="Elbow Flexion",
                instruction="Arms at sides. Bend and straighten both elbows fully 4 times (touch shoulders, then straighten).",
                duration=8.0,
                pause_after=3.0,
                target_segments=["forearm_right", "forearm_left"],
                method="bilateral_hinge"
            ),
            MovementDefinition(
                name="forearm_rotation",
                description="Forearm Rotation",
                instruction="Elbows bent 90°, upper arms at sides. Rotate both forearms 4 times: palms UP → palms DOWN → palms UP.",
                duration=8.0,
                pause_after=3.0,
                target_segments=["forearm_right", "forearm_left"],
                method="bilateral_axial"
            ),
            MovementDefinition(
                name="trunk_rotation",
                description="Trunk Rotation",
                instruction="Hands on hips. Rotate trunk left and right 3 times each direction. Keep pelvis still.",
                duration=8.0,
                pause_after=3.0,
                target_segments=["trunk"],
                method="principal_axis"
            ),
            MovementDefinition(
                name="head_rotation",
                description="Head Rotation",
                instruction="Shake head 'no' - turn left and right 3 times each direction.",
                duration=6.0,
                pause_after=3.0,
                target_segments=["head"],
                method="principal_axis"
            ),
        ]
        
        # Storage for computed alignments
        self.alignments: Dict[str, R] = {seg: R.identity() for seg in SEGMENT_NAMES}
        
        # Storage for recorded samples (in pelvis-relative frame!)
        self.recorded_samples: Dict[str, List[Dict[str, R]]] = {}
        
        # Current movement index
        self.current_movement_index = 0
        self.is_recording = False
        self.recording_start_time = None
    
    def get_current_movement(self) -> Optional[MovementDefinition]:
        """Get the current movement definition."""
        if self.current_movement_index < len(self.movements):
            return self.movements[self.current_movement_index]
        return None
    
    def start_recording(self):
        """Start recording samples for current movement."""
        self.is_recording = True
        self.recording_start_time = None
        movement = self.get_current_movement()
        if movement:
            self.recorded_samples[movement.name] = []
    
    def record_sample(self, sensor_orientations: Dict[str, R]):
        """
        Record a sample during movement.
        
        CRITICAL: Converts to pelvis-relative frame before storing!
        """
        if not self.is_recording:
            return
        
        movement = self.get_current_movement()
        if not movement:
            return
        
        # Get pelvis orientation
        pelvis_world = sensor_orientations['pelvis']
        
        # Convert ALL segments to pelvis-relative frame
        pelvis_relative_sample = {}
        
        for seg_name, sensor_rot in sensor_orientations.items():
            if seg_name == 'pelvis':
                # Pelvis is identity in its own frame
                pelvis_relative_sample[seg_name] = R.identity()
            else:
                # Express segment relative to pelvis
                pelvis_relative_sample[seg_name] = pelvis_world.inv() * sensor_rot
        
        # Store pelvis-relative sample
        self.recorded_samples[movement.name].append(pelvis_relative_sample)
    
    def stop_recording(self):
        """Stop recording and process the movement."""
        self.is_recording = False
        movement = self.get_current_movement()
        
        if not movement:
            return False, "No current movement"
        
        samples = self.recorded_samples.get(movement.name, [])
        
        if len(samples) < 10:
            return False, f"Insufficient samples: {len(samples)} (need at least 10)"
        
        # Check movement quality
        quality_ok, quality_msg = self._check_movement_quality(samples, movement)
        
        if not quality_ok:
            return False, f"Poor movement quality: {quality_msg}"
        
        # Compute alignments for this movement
        success, msg = self._compute_alignments(samples, movement)
        
        if not success:
            return False, msg
        
        # Move to next movement
        self.current_movement_index += 1
        
        return True, f"Movement '{movement.description}' complete. Quality: {quality_msg}"
    
    def is_complete(self) -> bool:
        """Check if all movements are complete."""
        return self.current_movement_index >= len(self.movements)
    
    def reset(self):
        """Reset calibration to start over."""
        self.current_movement_index = 0
        self.is_recording = False
        self.recorded_samples = {}
        self.alignments = {seg: R.identity() for seg in SEGMENT_NAMES}
    
    def _check_movement_quality(self, samples: List[Dict[str, R]], 
                                movement: MovementDefinition) -> Tuple[bool, str]:
        """
        Check if recorded movement has sufficient quality.
        
        Samples are already in pelvis-relative frame.
        """
        
        if len(samples) < 10:
            return False, "Too few samples"
        
        # For bilateral movements, check symmetry
        if "bilateral" in movement.method:
            if not self._check_bilateral_symmetry(samples, movement):
                return False, "Poor bilateral symmetry"
        
        # Check range of motion
        rom_ok, rom_msg = self._check_range_of_motion(samples, movement)
        if not rom_ok:
            return False, rom_msg
        
        return True, "Good quality"
    
    def _check_bilateral_symmetry(self, samples: List[Dict[str, R]], 
                                  movement: MovementDefinition) -> bool:
        """Check if left and right movements are symmetric (in pelvis-relative frame)."""
        
        # Find paired segments
        pairs = []
        for seg in movement.target_segments:
            if seg.endswith("_right"):
                left_seg = seg.replace("_right", "_left")
                if left_seg in movement.target_segments:
                    pairs.append((seg, left_seg))
        
        if not pairs:
            return True  # No bilateral pairs to check
        
        # Check correlation between left and right
        for right_seg, left_seg in pairs:
            right_data = [sample[right_seg] for sample in samples]
            left_data = [sample[left_seg] for sample in samples]
            
            # Extract rotation magnitudes (already in pelvis frame)
            right_mags = [rot.magnitude() for rot in right_data]
            left_mags = [rot.magnitude() for rot in left_data]
            
            # Check correlation
            if len(right_mags) > 1 and len(left_mags) > 1:
                correlation = np.corrcoef(right_mags, left_mags)[0, 1]
                
                if correlation < 0.5:  # Lower threshold (pelvis frame can be noisier)
                    return False
        
        return True
    
    def _check_range_of_motion(self, samples: List[Dict[str, R]], 
                               movement: MovementDefinition) -> Tuple[bool, str]:
        """Check if movement has sufficient range of motion (in pelvis-relative frame)."""
        
        # Get first target segment
        target_seg = movement.target_segments[0]
        
        # Skip pelvis itself
        if target_seg == 'pelvis':
            if len(movement.target_segments) > 1:
                target_seg = movement.target_segments[1]
            else:
                return True, "N/A"
        
        # Extract orientations (already pelvis-relative)
        orientations = [sample[target_seg] for sample in samples]
        
        # Compute relative rotations from first sample
        rel_rotations = [orientations[0].inv() * orient for orient in orientations]
        
        # Get magnitudes
        magnitudes = [rot.magnitude() * 180 / np.pi for rot in rel_rotations]
        
        # Check range
        rom = max(magnitudes) - min(magnitudes)
        
        if rom < 15:  # Lower threshold for pelvis-relative (pelvis movement affects apparent ROM)
            return False, f"Insufficient ROM: {rom:.1f}° (need >15°)"
        
        return True, f"ROM: {rom:.1f}°"
    
    def _compute_alignments(self, samples: List[Dict[str, R]], 
                           movement: MovementDefinition) -> Tuple[bool, str]:
        """
        Compute sensor alignments from recorded movement.
        
        CRITICAL: Samples are in pelvis-relative frame!
        """
        
        method = movement.method
        
        if method == "bilateral_symmetry":
            return self._compute_bilateral_symmetry_alignment(samples, movement)
        
        elif method == "shoulder_planar":
            return self._compute_shoulder_planar_alignment(samples, movement)
        
        elif method == "bilateral_hinge":
            return self._compute_bilateral_hinge_alignment(samples, movement)
        
        elif method == "bilateral_axial":
            return self._compute_bilateral_axial_alignment(samples, movement)
        
        elif method == "principal_axis":
            return self._compute_principal_axis_alignment(samples, movement)
        
        else:
            return False, f"Unknown method: {method}"
    
    def _compute_bilateral_symmetry_alignment(self, samples: List[Dict[str, R]], 
                                             movement: MovementDefinition) -> Tuple[bool, str]:
        """
        Compute alignment using bilateral symmetry constraint.
        Samples are in pelvis-relative frame.
        """
        
        print(f"[PelvisFuncCalib] Computing bilateral symmetry alignment for: {movement.target_segments}")
        
        # For legs: calibrate hip, knee, ankle joints
        # Parent for hips is pelvis (identity in pelvis frame)
        if "upper_leg_right" in movement.target_segments:
            success = self._calibrate_bilateral_joint_pelvis_relative(
                samples, "pelvis", "upper_leg_right", "upper_leg_left"
            )
            if not success:
                return False, "Failed to calibrate hips"
        
        # For knees: parent is upper_leg (thigh)
        if "lower_leg_right" in movement.target_segments:
            success = self._calibrate_bilateral_joint_pelvis_relative(
                samples, "upper_leg_right", "lower_leg_right", "lower_leg_left",
                parent_left="upper_leg_left"
            )
            if not success:
                return False, "Failed to calibrate knees"
        
        # For ankles: parent is lower_leg (shank)
        if "foot_right" in movement.target_segments:
            success = self._calibrate_bilateral_joint_pelvis_relative(
                samples, "lower_leg_right", "foot_right", "foot_left",
                parent_left="lower_leg_left"
            )
            if not success:
                return False, "Failed to calibrate ankles"
        
        return True, "Bilateral symmetry calibration complete"
    
    def _calibrate_bilateral_joint_pelvis_relative(self, samples: List[Dict[str, R]],
                                                   parent_right: str, child_right: str, child_left: str,
                                                   parent_left: Optional[str] = None) -> bool:
        """
        Calibrate a bilateral joint pair using symmetry.
        All samples are already in pelvis-relative frame.
        
        Args:
            samples: List of pelvis-relative orientation samples
            parent_right: Parent segment for right side
            child_right: Child segment for right side
            child_left: Child segment for left side
            parent_left: Parent segment for left side (if different from right)
        """
        
        if parent_left is None:
            parent_left = parent_right
        
        # Extract data (already pelvis-relative!)
        parent_r_data = [s[parent_right] for s in samples]
        parent_l_data = [s[parent_left] for s in samples]
        child_r_data = [s[child_right] for s in samples]
        child_l_data = [s[child_left] for s in samples]
        
        # Generate candidates
        candidates = self._generate_alignment_candidates()
        
        best_score = -np.inf
        best_alignment_r = R.identity()
        best_alignment_l = R.identity()
        
        # Test candidates
        for name_r, align_r in candidates:
            for name_l, align_l in candidates:
                
                # Compute relative rotations with alignment
                # Note: parent and child are already in pelvis frame
                rel_r = [(pr.inv() * (cr * align_r))
                        for pr, cr in zip(parent_r_data, child_r_data)]
                rel_l = [(pl.inv() * (cl * align_l))
                        for pl, cl in zip(parent_l_data, child_l_data)]
                
                # Extract angles (simple magnitude for now)
                angles_r = [rot.magnitude() * 180 / np.pi for rot in rel_r]
                angles_l = [rot.magnitude() * 180 / np.pi for rot in rel_l]
                
                # Score: correlation between left and right
                if len(angles_r) == len(angles_l) and len(angles_r) > 1:
                    correlation = np.corrcoef(angles_r, angles_l)[0, 1]
                    
                    # Also want good range of motion
                    rom_r = np.max(angles_r) - np.min(angles_r)
                    rom_l = np.max(angles_l) - np.min(angles_l)
                    avg_rom = (rom_r + rom_l) / 2
                    
                    # Score combines correlation and ROM
                    score = correlation * np.log(avg_rom + 1)
                    
                    if score > best_score:
                        best_score = score
                        best_alignment_r = align_r
                        best_alignment_l = align_l
        
        # Store best alignments
        self.alignments[child_right] = best_alignment_r
        self.alignments[child_left] = best_alignment_l
        
        print(f"[PelvisFuncCalib] {child_right}/{child_left}: score={best_score:.3f}")
        
        return best_score > 0
    
    def _compute_shoulder_planar_alignment(self, samples: List[Dict[str, R]], 
                                          movement: MovementDefinition) -> Tuple[bool, str]:
        """
        Compute shoulder alignment from sequential planar movements.
        Samples are in pelvis-relative frame.
        """
        
        print(f"[PelvisFuncCalib] Shoulder calibration: {len(samples)} samples")
        
        # For now, use principal axis on combined samples
        # TODO: Implement proper sequential planar method
        
        for seg in movement.target_segments:
            success = self._calibrate_principal_axis_pelvis_relative(
                samples, seg, parent="trunk"
            )
            if not success:
                return False, f"Failed to calibrate {seg}"
        
        return True, "Shoulder planar calibration complete"
    
    def _compute_bilateral_hinge_alignment(self, samples: List[Dict[str, R]], 
                                          movement: MovementDefinition) -> Tuple[bool, str]:
        """
        Compute alignment for bilateral hinge joints (elbows).
        Samples are in pelvis-relative frame.
        """
        
        print(f"[PelvisFuncCalib] Computing bilateral hinge alignment for: {movement.target_segments}")
        
        # Calibrate right and left elbows
        if "forearm_right" in movement.target_segments:
            success = self._calibrate_hinge_joint_pelvis_relative(
                samples, "upper_arm_right", "forearm_right"
            )
            if not success:
                return False, "Failed to calibrate right elbow"
        
        if "forearm_left" in movement.target_segments:
            success = self._calibrate_hinge_joint_pelvis_relative(
                samples, "upper_arm_left", "forearm_left"
            )
            if not success:
                return False, "Failed to calibrate left elbow"
        
        return True, "Bilateral hinge calibration complete"
    
    def _calibrate_hinge_joint_pelvis_relative(self, samples: List[Dict[str, R]],
                                               parent: str, child: str) -> bool:
        """
        Calibrate a single hinge joint.
        Samples are in pelvis-relative frame.
        """
        
        parent_data = [s[parent] for s in samples]
        child_data = [s[child] for s in samples]
        
        candidates = self._generate_alignment_candidates()
        
        best_score = -np.inf
        best_alignment = R.identity()
        
        for name, align in candidates:
            # Compute relative rotations (parent and child already pelvis-relative)
            rel_rots = [(p.inv() * (c * align))
                       for p, c in zip(parent_data, child_data)]
            
            # Convert to Euler angles (ZXY sequence)
            angles = np.array([rot.as_euler('ZXY', degrees=True) for rot in rel_rots])
            
            # For hinge: one axis should have high variance, others low
            variances = np.var(angles, axis=0)
            primary_var = np.max(variances)
            secondary_var = np.sum(variances) - primary_var
            
            # Score: high primary, low secondary
            score = primary_var / (secondary_var + 1)
            
            if score > best_score:
                best_score = score
                best_alignment = align
        
        self.alignments[child] = best_alignment
        print(f"[PelvisFuncCalib] {child}: hinge score={best_score:.3f}")
        
        return best_score > 1.0
    
    def _compute_bilateral_axial_alignment(self, samples: List[Dict[str, R]], 
                                          movement: MovementDefinition) -> Tuple[bool, str]:
        """
        Compute alignment for bilateral axial rotation (forearm pronation/supination).
        Samples are in pelvis-relative frame.
        """
        
        print(f"[PelvisFuncCalib] Computing bilateral axial alignment for: {movement.target_segments}")
        
        # Similar to hinge, but looking for rotation around long axis
        for seg in movement.target_segments:
            parent = "upper_arm_right" if "right" in seg else "upper_arm_left"
            success = self._calibrate_principal_axis_pelvis_relative(samples, seg, parent=parent)
            if not success:
                return False, f"Failed to calibrate {seg}"
        
        return True, "Bilateral axial calibration complete"
    
    def _compute_principal_axis_alignment(self, samples: List[Dict[str, R]], 
                                         movement: MovementDefinition) -> Tuple[bool, str]:
        """
        Compute alignment using principal axis method.
        Samples are in pelvis-relative frame.
        """
        
        print(f"[PelvisFuncCalib] Computing principal axis alignment for: {movement.target_segments}")
        
        for seg in movement.target_segments:
            if seg == "trunk":
                parent = "pelvis"
            elif seg == "head":
                parent = "trunk"
            else:
                parent = "pelvis"  # fallback
            
            success = self._calibrate_principal_axis_pelvis_relative(samples, seg, parent=parent)
            if not success:
                return False, f"Failed to calibrate {seg}"
        
        return True, "Principal axis calibration complete"
    
    def _calibrate_principal_axis_pelvis_relative(self, samples: List[Dict[str, R]],
                                                  segment: str, parent: str) -> bool:
        """
        Calibrate using principal axis method.
        Samples are in pelvis-relative frame.
        """
        
        parent_data = [s[parent] for s in samples]
        child_data = [s[segment] for s in samples]
        
        candidates = self._generate_alignment_candidates()
        
        best_score = -np.inf
        best_alignment = R.identity()
        
        for name, align in candidates:
            # Compute relative rotations (already pelvis-relative)
            rel_rots = [(p.inv() * (c * align))
                       for p, c in zip(parent_data, child_data)]
            
            # Get rotation vectors
            rotvecs = np.array([rot.as_rotvec() for rot in rel_rots])
            
            # SVD to find principal axis
            if len(rotvecs) > 3:
                U, S, Vt = np.linalg.svd(rotvecs)
                
                # Score: ratio of first to second singular value
                if S[1] > 1e-6:
                    score = S[0] / S[1]
                else:
                    score = S[0]
                
                if score > best_score:
                    best_score = score
                    best_alignment = align
        
        self.alignments[segment] = best_alignment
        print(f"[PelvisFuncCalib] {segment}: principal axis score={best_score:.3f}")
        
        return best_score > 1.0
    
    def _generate_alignment_candidates(self) -> List[Tuple[str, R]]:
        """Generate candidate sensor alignment rotations."""
        
        candidates = []
        
        # Identity
        candidates.append(('identity', R.identity()))
        
        # 180° flips
        for axis in ['x', 'y', 'z']:
            candidates.append((f'{axis}180', R.from_euler(axis, 180, degrees=True)))
        
        # 90° rotations
        for axis in ['x', 'y', 'z']:
            for angle in [90, -90]:
                candidates.append((f'{axis}{angle:+d}', R.from_euler(axis, angle, degrees=True)))
        
        # Double 180° rotations
        candidates.append(('xy180', R.from_euler('xy', [180, 180], degrees=True)))
        candidates.append(('xz180', R.from_euler('xz', [180, 180], degrees=True)))
        candidates.append(('yz180', R.from_euler('yz', [180, 180], degrees=True)))
        
        # Small adjustments (±15°, ±30°)
        for axis in ['x', 'y', 'z']:
            for angle in [-30, -15, 15, 30]:
                candidates.append((f'{axis}{angle:+d}_fine', R.from_euler(axis, angle, degrees=True)))
        
        return candidates
