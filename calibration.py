"""Enhanced calibration manager with post-functional static calibration"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Optional, Tuple, Any
from config import BODY_MODEL, SEGMENT_NAMES

# Maximum samples to record per movement (for performance)
MAX_SAMPLES_PER_MOVEMENT = 500


class CalibrationManager:
    """
    Manages orientation-agnostic calibration for all segments with full-body
    functional calibration sequence and post-functional static calibration.
    """
    
    def __init__(self, body_model: Dict[str, Any]):
        self.body_model = body_model
        
        # Sensor-to-segment alignment rotations (found via functional calib)
        self.alignments: Dict[str, R] = {seg: R.identity() for seg in SEGMENT_NAMES}
        
        # Neutral pose offsets (found via static calib)
        self.offsets: Dict[str, R] = {seg: R.identity() for seg in SEGMENT_NAMES}
        
        # Calibration state
        self.is_static_calibrated = False
        self.is_functional_calibrated = False
        self.needs_post_static_calibration = False
        self.post_static_ready = False
        self.calibration_pose = "n_pose"  # Track which pose was used for calibration
        
        # Recorded movements for functional calibration
        self.recorded_movements: Dict[str, List[Dict[str, R]]] = {}
        
        # Movement sequence configuration with pause intervals
        self.movement_sequence = [
            {
                "name": "squat",
                "description": "5 slow squats",
                "duration": 8.0,
                "pause": 5.0,
                "target_segments": ["pelvis", "upper_leg_right", "upper_leg_left",
                                   "lower_leg_right", "lower_leg_left", "foot_right", "foot_left"],
                "method": "bilateral_symmetry",
                "instruction": "Stand with feet shoulder-width apart and perform 5 slow squats"
            },
            {
                "name": "trunk_flexion",
                "description": "Trunk flexion/extension",
                "duration": 6.0,
                "pause": 5.0,
                "target_segments": ["trunk"],
                "method": "principal_axis_alignment",
                "instruction": "Hands on hips; bend forwards/backwards keeping pelvis still"
            },
            {
                "name": "trunk_lateral_bend",
                "description": "Side bends left and right",
                "duration": 6.0,
                "pause": 5.0,
                "target_segments": ["trunk"],
                "method": "principal_axis_alignment",
                "instruction": "Hands on hips; bend sideways left and right, keep legs straight and pelvis still"
            },
            {
                "name": "head_nod",
                "description": "Nod 'yes'",
                "duration": 5.0,
                "pause": 4.0,
                "target_segments": ["head"],
                "method": "principal_axis_alignment",
                "instruction": "Keep trunk still and nod 'yes' (flexion/extension)"
            },
            {
                "name": "head_rotate",
                "description": "Shake 'no'",
                "duration": 5.0,
                "pause": 4.0,
                "target_segments": ["head"],
                "method": "principal_axis_alignment",
                "instruction": "Keep trunk still and shake 'no' (axial rotation)"
            },
            {
                "name": "arm_flexion",
                "description": "Arm flexion/extension",
                "duration": 6.0,
                "pause": 5.0,
                "target_segments": ["upper_arm_right", "upper_arm_left"],
                "method": "shoulder_multi",
                "instruction": "Raise and lower arms forward (keep trunk still)"
            },
            {
                "name": "arm_abduction",
                "description": "Arm abduction/adduction",
                "duration": 6.0,
                "pause": 5.0,
                "target_segments": ["upper_arm_right", "upper_arm_left"],
                "method": "shoulder_multi",
                "instruction": "Raise and lower arms out to the side (keep trunk still)"
            },
            {
                "name": "elbow_flexion",
                "description": "Elbow flexion/extension",
                "duration": 6.0,
                "pause": 5.0,
                "target_segments": ["forearm_right", "forearm_left"],
                "method": "elbow_hinge",
                "instruction": "Bend and straighten elbows (keep upper arms still)"
            },
            {
                "name": "wrist_motion",
                "description": "Wrist flexion and deviation",
                "duration": 6.0,
                "pause": 5.0,
                "target_segments": ["hand_right", "hand_left"],
                "method": "wrist_alignment",
                "instruction": "Move wrists up/down (flexion) and side-to-side (deviation)"
            },
            {
                "name": "ankle_dorsiflexion",
                "description": "Ankle dorsi/plantar flexion",
                "duration": 6.0,
                "pause": 5.0,
                "target_segments": ["foot_right", "foot_left"],
                "method": "ankle_alignment",
                "instruction": "Point toes up and down (keep knees straight)"
            }
        ]
        
        self.current_movement_index = 0
        self.calibration_progress = 0.0  # 0.0 to 1.0
        self.in_pause_phase = False
        self.pause_start_time = None
        
    def reset_static_calibration(self):
        """Reset static calibration to allow re-calibration"""
        self.offsets = {seg: R.identity() for seg in SEGMENT_NAMES}
        self.is_static_calibrated = False
        self.needs_post_static_calibration = False
        self.post_static_ready = False
        print("[Calibration] Static calibration reset")
    
    def reset_full_calibration(self):
        """Reset both static and functional calibration"""
        self.reset_static_calibration()
        self.alignments = {seg: R.identity() for seg in SEGMENT_NAMES}
        self.is_functional_calibrated = False
        self.recorded_movements.clear()
        self.current_movement_index = 0
        self.calibration_progress = 0.0
        print("[Calibration] Full calibration reset")
    
    def static_calibration(self, sensor_orientations: Dict[str, R], is_post_calibration: bool = False, pose_type: str = "n_pose"):
        """
        Perform static calibration in neutral anatomical pose.
        
        Computes offsets that work with pelvis-relative calibration application.
        
        Args:
            sensor_orientations: Raw sensor orientations
            is_post_calibration: If True, this is post-functional calibration
            pose_type: "n_pose" (arms at sides) or "t_pose" (arms extended laterally)
        """
        if is_post_calibration:
            print(f"[Calibration] Starting POST-functional static calibration ({pose_type})...")
        else:
            print(f"[Calibration] Starting static calibration ({pose_type})...")
        
        # Get pelvis sensor orientation
        pelvis_sensor = sensor_orientations['pelvis']
        
        # For pelvis: compute offset as before (to world frame)
        aligned_pelvis = pelvis_sensor * self.alignments['pelvis']
        self.offsets['pelvis'] = aligned_pelvis.inv()
        
        # For all other segments: compute offset in pelvis-relative frame
        for seg_name, sensor_rot in sensor_orientations.items():
            if seg_name == 'pelvis':
                continue
            
            # Express sensor relative to pelvis sensor
            sensor_rel_pelvis = pelvis_sensor.inv() * sensor_rot
            
            # Apply alignment in pelvis-relative frame
            aligned_rel = sensor_rel_pelvis * self.alignments[seg_name]
            
            # Apply pose-specific adjustments for upper limbs
            if pose_type == "t_pose":
                # In T-pose, arms are extended laterally (90° abduction in frontal plane)
                # This is rotation around the X-axis (sagittal/forward-back axis)
                # 
                # IMPORTANT: We're using pelvis-relative calibration, so ALL segments
                # (upper arm, forearm, hand) are expressed relative to pelvis, not to each other.
                # Therefore, they all need the T-pose correction applied.
                
                if seg_name == 'upper_arm_right':
                    # Right arm: extended laterally (right) in T-pose → down in neutral
                    # Rotate 90° around X (forward axis) to bring arm down
                    t_pose_correction = R.from_euler('x', 90, degrees=True)
                    aligned_rel = aligned_rel * t_pose_correction
                    
                elif seg_name == 'upper_arm_left':
                    # Left arm: extended laterally (left) in T-pose → down in neutral
                    # Rotate -90° around X (forward axis) to bring arm down
                    t_pose_correction = R.from_euler('x', -90, degrees=True)
                    aligned_rel = aligned_rel * t_pose_correction
                    
                elif seg_name == 'forearm_right':
                    # Forearm also horizontal in T-pose, needs same correction as upper arm
                    t_pose_correction = R.from_euler('x', 90, degrees=True)
                    aligned_rel = aligned_rel * t_pose_correction
                    
                elif seg_name == 'forearm_left':
                    # Forearm also horizontal in T-pose, needs same correction as upper arm
                    t_pose_correction = R.from_euler('x', -90, degrees=True)
                    aligned_rel = aligned_rel * t_pose_correction
                    
                elif seg_name == 'hand_right':
                    # Hand also horizontal in T-pose, needs same correction
                    t_pose_correction = R.from_euler('x', 90, degrees=True)
                    aligned_rel = aligned_rel * t_pose_correction
                    
                elif seg_name == 'hand_left':
                    # Hand also horizontal in T-pose, needs same correction
                    t_pose_correction = R.from_euler('x', -90, degrees=True)
                    aligned_rel = aligned_rel * t_pose_correction
            
            # Compute offset in pelvis-relative frame
            # This offset will be applied as: offset * sensor_rel_pelvis * alignment
            # We want the result to be identity in neutral pose
            self.offsets[seg_name] = aligned_rel.inv()
        
        if is_post_calibration:
            self.needs_post_static_calibration = False
            self.post_static_ready = True
            print(f"[Calibration] POST-functional static calibration ({pose_type}) complete.")
        
        self.is_static_calibrated = True
        self.calibration_pose = pose_type
        print(f"[Calibration] Static calibration ({pose_type}) complete.")
        
        # Debug: Print pelvis orientation
        pelvis_euler = pelvis_sensor.as_euler('xyz', degrees=True)
        print(f"[Calibration] Pelvis raw orientation (XYZ Euler): {pelvis_euler}")
        pelvis_euler = pelvis_sensor.as_euler('xyz', degrees=True)
        print(f"[Calibration] Pelvis raw orientation (XYZ Euler): {pelvis_euler}")
    
    def start_functional_calibration_sequence(self):
        """Start the full functional calibration sequence."""
        self.current_movement_index = 0
        self.is_functional_calibrated = False
        self.needs_post_static_calibration = False
        self.post_static_ready = False
        self.calibration_progress = 0.0
        self.in_pause_phase = False
        self.recorded_movements.clear()
        print("[Calibration] Starting full functional calibration sequence...")
        return self._get_current_movement()
    
    def functional_calibration_complete(self):
        """Mark functional calibration as complete and request post-static calibration"""
        self.is_functional_calibrated = True
        self.needs_post_static_calibration = True
        self.post_static_ready = False
        print("[Calibration] Functional calibration complete - ready for post-static calibration")
    
    def start_movement_recording(self, movement_name: str):
        """Start recording sensor data for functional calibration."""
        self.recorded_movements[movement_name] = []
        print(f"[Calibration] Recording movement: {movement_name}")
    
    def record_movement_sample(self, movement_name: str, sensor_orientations: Dict[str, R]):
        """Record a sample during functional calibration movement."""
        if movement_name not in self.recorded_movements:
            return
        
        # Limit samples for performance
        if len(self.recorded_movements[movement_name]) >= MAX_SAMPLES_PER_MOVEMENT:
            return
        
        # Apply current calibration and store
        calibrated = {}
        for seg, sensor_rot in sensor_orientations.items():
            calibrated[seg] = self.offsets[seg] * sensor_rot * self.alignments[seg]
        
        self.recorded_movements[movement_name].append(calibrated)
    
    def start_pause_phase(self):
        """Start pause phase between movements."""
        self.in_pause_phase = True
        movement = self._get_current_movement()
        if movement:
            print(f"[Calibration] Pause before {movement['description']}")
    
    def check_pause_complete(self, elapsed_time: float) -> bool:
        """Check if pause phase is complete."""
        if not self.in_pause_phase:
            return True
        
        movement = self._get_current_movement()
        if movement and elapsed_time >= movement["pause"]:
            self.in_pause_phase = False
            return True
        
        return False
    
    def movement_completed(self, movement_name: str) -> bool:
        """
        Called when a movement recording is complete.
        Processes the movement and returns True if calibration should continue.
        """
        if movement_name not in self.recorded_movements:
            return False
        
        samples = self.recorded_movements[movement_name]
        movement_config = next(m for m in self.movement_sequence if m["name"] == movement_name)
        
        if len(samples) < 20:
            print(f"[Calibration] Insufficient samples for {movement_name} ({len(samples)} samples)")
            return False
        
        print(f"[Calibration] Processing {movement_name} ({len(samples)} samples)...")
        
        # Apply the appropriate calibration method
        success = False
        if movement_config["method"] == "bilateral_symmetry":
            success = self._calibrate_bilateral_symmetry(
                movement_name, 
                movement_config["target_segments"]
            )
        elif movement_config["method"] == "principal_axis_alignment":
            success = self._calibrate_principal_axis_alignment(
                movement_name,
                movement_config["target_segments"]
            )
        elif movement_config["method"] == "shoulder_multi":
            success = self._calibrate_shoulder_multi(
                movement_name,
                movement_config["target_segments"]
            )
        elif movement_config["method"] == "elbow_hinge":
            success = self._calibrate_elbow_hinge(
                movement_name,
                movement_config["target_segments"]
            )
        elif movement_config["method"] == "wrist_alignment":
            success = self._calibrate_wrist_alignment(
                movement_name,
                movement_config["target_segments"]
            )
        elif movement_config["method"] == "ankle_alignment":
            success = self._calibrate_ankle_alignment(
                movement_name,
                movement_config["target_segments"]
            )
        
        # Update progress
        self.current_movement_index += 1
        self.calibration_progress = self.current_movement_index / len(self.movement_sequence)
        
        print(f"[Calibration] Movement {movement_name} processed. Progress: {self.calibration_progress:.1%}")
        return success
    
    def _calibrate_bilateral_symmetry(self, movement_name: str, target_segments: List[str]) -> bool:
        """Calibrate leg segments using bilateral symmetry."""
        samples = self.recorded_movements[movement_name]
        
        # For legs, we assume right side is reference
        leg_pairs = [
            ("upper_leg_right", "upper_leg_left"),
            ("lower_leg_right", "lower_leg_left"),
            ("foot_right", "foot_left")
        ]
        
        for right_seg, left_seg in leg_pairs:
            if left_seg not in target_segments or right_seg not in target_segments:
                continue
            
            # Get parent segment
            parent_name = self.body_model[left_seg].parent
            if not parent_name:
                continue
            
            best_alignment = self._find_optimal_alignment_bilateral(
                samples,
                parent_name,
                right_seg,
                left_seg
            )
            
            if best_alignment is not None:
                self.alignments[left_seg] = best_alignment
                print(f"[Calibration] Aligned {left_seg} with {right_seg}")
            else:
                print(f"[Calibration] Failed to align {left_seg}")
        
        return True
    
    
    def _calibrate_principal_axis_alignment(self, movement_name: str, target_segments: List[str]) -> bool:
        """Calibrate segments by identifying principal axis of movement."""
        samples = self.recorded_movements[movement_name]
        samples = self._filter_samples_for_movement(movement_name, samples)

        for seg_name in target_segments:
            seg_def = self.body_model[seg_name]
            parent_name = seg_def.parent
            if not parent_name:
                continue

            # Expected primary axis depends on BOTH the segment and the movement recorded
            if seg_name == "trunk":
                # trunk_flexion: flex/ext; trunk_lateral_bend: lateral bend
                expected_axis = 0 if "flex" in movement_name else 1
            elif seg_name == "head":
                # head_nod: flex/ext; head_rotate: axial rotation
                expected_axis = 0 if "nod" in movement_name else 2
            else:
                expected_axis = 0

            success = self._align_to_expected_axis(seg_name, samples, expected_axis)
            if success:
                print(f"[Calibration] Successfully aligned {seg_name}")

        return True


    
    
    def _calibrate_shoulder_multi(self, movement_name: str, target_segments: List[str]) -> bool:
        """Calibrate upper arms using two movements (flexion + abduction) and contralateral symmetry.

        This keeps your original 'right is reference' idea (so we don't break the legs),
        but waits until both arm movements are recorded before solving.
        """
        # Only solve once we have BOTH movements
        flex = self.recorded_movements.get("arm_flexion", [])
        abd = self.recorded_movements.get("arm_abduction", [])

        # Apply stability filtering (reduce trunk sway contamination)
        flex = self._filter_samples_for_movement("arm_flexion", flex)
        abd = self._filter_samples_for_movement("arm_abduction", abd)

        if len(flex) < 20 or len(abd) < 20:
            # Not enough yet; allow sequence to continue
            print("[Calibration] Shoulder calibration waiting for both arm movements...")
            return True

        # Combine to give richer excitation (helps ball joints)
        samples = flex + abd

        # Use right arm as reference for left arm (same as your previous approach, but with better excitation)
        for seg_name in target_segments:
            if "_left" not in seg_name:
                continue
            right_seg = seg_name.replace("left", "right")
            if right_seg not in BODY_MODEL:
                continue

            parent_name = self.body_model[seg_name].parent
            if not parent_name:
                continue

            best_alignment = self._find_optimal_alignment_bilateral(
                samples,
                parent_name,
                right_seg,
                seg_name
            )
            if best_alignment is not None:
                self.alignments[seg_name] = best_alignment
                print(f"[Calibration] Aligned {seg_name} with {right_seg} (multi-motion)")
            else:
                print(f"[Calibration] Failed to align {seg_name} (multi-motion)")

        return True



    
    def _calibrate_elbow_hinge(self, movement_name: str, target_segments: List[str]) -> bool:
        """Calibrate forearms using hinge joint assumption."""
        samples = self.recorded_movements[movement_name]
        
        for seg_name in target_segments:
            seg_def = self.body_model[seg_name]
            
            # For elbow, primary motion should be flexion/extension (first axis)
            best_alignment = self._optimize_hinge_alignment(
                seg_name, samples, expected_axis=0
            )
            
            if best_alignment is not None:
                self.alignments[seg_name] = best_alignment
                print(f"[Calibration] Aligned {seg_name} as hinge joint")
        
        return True
    
    def _calibrate_wrist_alignment(self, movement_name: str, target_segments: List[str]) -> bool:
        """Calibrate hands using expected wrist motion patterns."""
        samples = self.recorded_movements[movement_name]
        
        for seg_name in target_segments:
            best_alignment = self._find_wrist_alignment(seg_name, samples)
            if best_alignment is not None:
                self.alignments[seg_name] = best_alignment
                print(f"[Calibration] Aligned {seg_name}")
        
        return True
    
    def _calibrate_ankle_alignment(self, movement_name: str, target_segments: List[str]) -> bool:
        """Calibrate feet using expected ankle motion patterns."""
        samples = self.recorded_movements[movement_name]
        
        for seg_name in target_segments:
            best_alignment = self._find_ankle_alignment(seg_name, samples)
            if best_alignment is not None:
                self.alignments[seg_name] = best_alignment
                print(f"[Calibration] Aligned {seg_name}")
        
        return True
    
    
    def _filter_samples_for_movement(self, movement_name: str, samples: List[Dict[str, R]]) -> List[Dict[str, R]]:
        """Filter samples to reduce parent-segment contamination (simple stability gating).

        - For trunk movements, keep samples where pelvis is relatively stable.
        - For head movements, keep samples where trunk is relatively stable.
        """
        if not samples:
            return samples

        def rot_mag_deg(rot: R) -> float:
            return float(np.degrees(np.linalg.norm(rot.as_rotvec())))

        # Reference orientations from first sample
        ref = samples[0]

        if movement_name.startswith("trunk"):
            if "pelvis" not in ref:
                return samples
            ref_pelvis = ref["pelvis"]
            kept = []
            for s in samples:
                if "pelvis" not in s:
                    continue
                d = ref_pelvis.inv() * s["pelvis"]
                if rot_mag_deg(d) <= 15.0:  # degrees
                    kept.append(s)
            if len(kept) >= 20:
                return kept
            return samples

        if movement_name.startswith("head"):
            if "trunk" not in ref:
                return samples
            ref_trunk = ref["trunk"]
            kept = []
            for s in samples:
                if "trunk" not in s:
                    continue
                d = ref_trunk.inv() * s["trunk"]
                if rot_mag_deg(d) <= 10.0:
                    kept.append(s)
            if len(kept) >= 20:
                return kept
            return samples

        if movement_name.startswith("arm"):
            # Keep samples where trunk is stable (helps shoulder calibration)
            if "trunk" not in ref:
                return samples
            ref_trunk = ref["trunk"]
            kept = []
            for s in samples:
                if "trunk" not in s:
                    continue
                d = ref_trunk.inv() * s["trunk"]
                if rot_mag_deg(d) <= 10.0:
                    kept.append(s)
            if len(kept) >= 20:
                return kept
            return samples

        return samples
    def _align_to_expected_axis(self, seg_name: str, samples: List[Dict[str, R]], 
                               expected_axis: int) -> bool:
        """Find alignment that maps observed primary motion to expected anatomical axis."""
        seg_def = self.body_model[seg_name]
        parent_name = seg_def.parent
        
        candidates = self._generate_alignment_candidates()
        best_score = -np.inf
        best_rot = None
        
        for name, candidate_rot in candidates:
            # Test this alignment
            aligned_angles = []
            for sample in samples:
                parent_rot = sample[parent_name]
                child_raw = sample[seg_name]
                child_aligned = candidate_rot * child_raw
                rel_rot = parent_rot.inv() * child_aligned
                angles = rel_rot.as_euler(seg_def.euler_seq, degrees=True)
                aligned_angles.append(angles)
            
            aligned_angles = np.array(aligned_angles)
            
            # Score: variance on expected axis - variance on other axes
            variances = np.var(aligned_angles, axis=0)
            score = variances[expected_axis] - np.sum(variances[np.arange(3) != expected_axis])
            
            if score > best_score:
                best_score = score
                best_rot = candidate_rot
        
        if best_rot is not None and best_score > 10.0:  # Threshold
            self.alignments[seg_name] = best_rot
            return True
        
        return False
    
    def _optimize_hinge_alignment(self, seg_name: str, samples: List[Dict[str, R]], 
                                 expected_axis: int) -> Optional[R]:
        """Optimize alignment for hinge joints."""
        seg_def = self.body_model[seg_name]
        parent_name = seg_def.parent
        
        candidates = self._generate_alignment_candidates()
        best_score = -np.inf
        best_rot = None
        
        for name, candidate_rot in candidates:
            aligned_angles = []
            for sample in samples:
                parent_rot = sample[parent_name]
                child_raw = sample[seg_name]
                child_aligned = candidate_rot * child_raw
                rel_rot = parent_rot.inv() * child_aligned
                angles = rel_rot.as_euler(seg_def.euler_seq, degrees=True)
                aligned_angles.append(angles)
            
            aligned_angles = np.array(aligned_angles)
            
            # For hinge joints: high variance on primary, low on others
            variances = np.var(aligned_angles, axis=0)
            primary_var = variances[expected_axis]
            secondary_vars = np.sum(variances[np.arange(3) != expected_axis])
            
            # Score: primary variance / (secondary variance + epsilon)
            score = primary_var / (secondary_vars + 1.0)
            
            if score > best_score:
                best_score = score
                best_rot = candidate_rot
        
        return best_rot if best_score > 5.0 else None
    
    def _find_wrist_alignment(self, seg_name: str, samples: List[Dict[str, R]]) -> Optional[R]:
        """Find optimal alignment for wrist."""
        seg_def = self.body_model[seg_name]
        parent_name = seg_def.parent
        
        candidates = self._generate_alignment_candidates()
        best_score = -np.inf
        best_rot = None
        
        for name, candidate_rot in candidates:
            aligned_angles = []
            for sample in samples:
                parent_rot = sample[parent_name]
                child_raw = sample[seg_name]
                child_aligned = candidate_rot * child_raw
                rel_rot = parent_rot.inv() * child_aligned
                angles = rel_rot.as_euler(seg_def.euler_seq, degrees=True)
                aligned_angles.append(angles)
            
            aligned_angles = np.array(aligned_angles)
            
            # Wrist: expect significant motion in first two axes (flex/ext, radial/ulnar)
            variances = np.var(aligned_angles, axis=0)
            wrist_score = np.sum(variances[:2]) - variances[2]  # Penalize rotation
            
            if wrist_score > best_score:
                best_score = wrist_score
                best_rot = candidate_rot
        
        return best_rot if best_score > 10.0 else None
    
    def _find_ankle_alignment(self, seg_name: str, samples: List[Dict[str, R]]) -> Optional[R]:
        """Find optimal alignment for ankle/foot."""
        seg_def = self.body_model[seg_name]
        parent_name = seg_def.parent
        
        candidates = self._generate_alignment_candidates()
        best_score = -np.inf
        best_rot = None
        
        for name, candidate_rot in candidates:
            aligned_angles = []
            for sample in samples:
                parent_rot = sample[parent_name]
                child_raw = sample[seg_name]
                child_aligned = candidate_rot * child_raw
                rel_rot = parent_rot.inv() * child_aligned
                angles = rel_rot.as_euler(seg_def.euler_seq, degrees=True)
                aligned_angles.append(angles)
            
            aligned_angles = np.array(aligned_angles)
            
            # Ankle: primary motion in dorsi/plantar flexion (axis 0)
            # Some inversion/eversion (axis 1)
            variances = np.var(aligned_angles, axis=0)
            ankle_score = variances[0] + 0.5 * variances[1] - 0.1 * variances[2]
            
            if ankle_score > best_score:
                best_score = ankle_score
                best_rot = candidate_rot
        
        return best_rot if ankle_score > 5.0 else None
    
    def _generate_alignment_candidates(self) -> List[Tuple[str, R]]:
        """Generate candidate alignment rotations."""
        axes = ['x', 'y', 'z']
        angles = [0, 180]
        
        candidates = [("identity", R.identity())]
        
        # Single 180° rotations
        for axis in axes:
            candidates.append((f"{axis}180", R.from_euler(axis, 180, degrees=True)))
        
        # Double 180° rotations
        for i, axis1 in enumerate(axes):
            for axis2 in axes[i+1:]:
                candidates.append((f"{axis1}{axis2}180", 
                                 R.from_euler(f"{axis1}{axis2}", [180, 180], degrees=True)))
        
        # Triple 180° rotation
        candidates.append(("xyz180", R.from_euler('xyz', [180, 180, 180], degrees=True)))
        
        # Small angle corrections (±30° increments)
        for axis in axes:
            for angle in [-30, -15, 15, 30]:
                candidates.append((f"{axis}{angle:+d}", R.from_euler(axis, angle, degrees=True)))
        
        return candidates
    
    def _find_optimal_alignment_bilateral(
        self,
        samples: List[Dict[str, R]],
        parent_seg: str,
        child_seg_right: str,
        child_seg_left: str
    ) -> Optional[R]:
        """Find optimal alignment for left side using right as reference."""
        euler_seq = BODY_MODEL[child_seg_right].euler_seq
        
        # Extract right side angles
        right_angles = []
        for sample in samples:
            parent = sample[parent_seg]
            child = sample[child_seg_right]
            rel = parent.inv() * child
            angles = rel.as_euler(euler_seq, degrees=True)
            right_angles.append(angles)
        right_angles = np.array(right_angles)
        
        # Find axis with most variation
        variances = np.var(right_angles, axis=0)
        primary_axis = np.argmax(variances)
        
        if variances[primary_axis] < 5.0:
            print(f"[Calibration] Insufficient movement variation: {variances}")
            return None
        
        # Test candidate alignments
        candidates = self._generate_alignment_candidates()
        best_corr = -2.0
        best_rot = None
        
        for name, candidate_rot in candidates:
            left_angles = []
            for sample in samples:
                parent = sample[parent_seg]
                child_raw = sample[child_seg_left]
                child_aligned = candidate_rot * child_raw
                rel = parent.inv() * child_aligned
                angles = rel.as_euler(euler_seq, degrees=True)
                left_angles.append(angles)
            left_angles = np.array(left_angles)
            
            # Compute correlation on primary axis
            right_primary = right_angles[:, primary_axis]
            left_primary = left_angles[:, primary_axis]
            
            # Center signals
            right_primary = right_primary - right_primary.mean()
            left_primary = left_primary - left_primary.mean()
            
            if left_primary.std() < 1.0:
                continue
            
            corr = np.corrcoef(right_primary, left_primary)[0, 1]
            
            if corr > best_corr:
                best_corr = corr
                best_rot = candidate_rot
        
        if best_corr > 0.5:  # Reasonable correlation threshold
            print(f"[Calibration] Found alignment with correlation: {best_corr:.3f}")
            return best_rot
        else:
            print(f"[Calibration] No good alignment found (best corr: {best_corr:.3f})")
            return None
    
    def _get_current_movement(self) -> Optional[Dict]:
        """Get current movement configuration."""
        if self.current_movement_index < len(self.movement_sequence):
            return self.movement_sequence[self.current_movement_index]
        return None
    
    def get_calibration_progress_text(self) -> str:
        """Get text describing current calibration progress."""
        if not self.is_static_calibrated:
            return "Awaiting initial static calibration"
        
        if self.needs_post_static_calibration:
            return "Ready for final static calibration - Stand in neutral pose"
        
        if self.post_static_ready and self.is_functional_calibrated:
            return "Calibration complete - System ready!"
        
        if not self.is_functional_calibrated:
            # Show movement progress
            if self.in_pause_phase:
                movement = self._get_current_movement()
                return f"Pause before: {movement['description']}"
            
            movement = self._get_current_movement()
            if movement:
                progress_pct = int(self.calibration_progress * 100)
                return f"{movement['description']} ({progress_pct}%)"
        
        return "Calibration in progress"
    
    def get_pause_text(self) -> str:
        """Get text for pause phase."""
        if self.in_pause_phase:
            movement = self._get_current_movement()
            return f"Get ready for: {movement['description']}"
        return ""
    
    def get_calibrated_orientations(self, sensor_orientations: Dict[str, R]) -> Dict[str, R]:
        """
        Apply calibration to raw sensor data using pelvis-relative approach.
        
        This ensures calibration is rotation-invariant - joint angles remain
        correct regardless of body orientation in world frame.
        """
        calibrated = {}
        
        # Get raw pelvis orientation
        pelvis_sensor = sensor_orientations['pelvis']
        
        # Apply full pelvis calibration (removes ALL rotation including yaw)
        pelvis_calibrated_full = self.offsets['pelvis'] * pelvis_sensor * self.alignments['pelvis']
        
        # For pelvis: we want to keep the yaw (body direction) but remove pitch/roll
        # Extract yaw from the RAW aligned sensor
        pelvis_aligned = pelvis_sensor * self.alignments['pelvis']
        
        # Project pelvis forward direction onto horizontal plane to get yaw
        forward = pelvis_aligned.apply([1, 0, 0])  # Forward direction
        yaw_angle = np.arctan2(forward[2], forward[0])  # Heading in horizontal plane
        
        # Get the up direction from calibrated pelvis (removes pitch/roll)
        up_calibrated = pelvis_calibrated_full.apply([0, 1, 0])
        
        # Construct a rotation that:
        # 1. Points up in the calibrated direction (removes pitch/roll tilt)
        # 2. Points forward at the raw yaw angle (preserves body direction)
        
        # Create target forward direction (horizontal, at yaw angle)
        target_forward = np.array([np.cos(yaw_angle), 0, np.sin(yaw_angle)])
        target_forward = target_forward / np.linalg.norm(target_forward)
        
        # Create target up direction (from calibrated, should be close to [0,1,0])
        target_up = up_calibrated.copy()
        target_up[1] = max(abs(target_up[1]), 0.1) * np.sign(target_up[1]) if target_up[1] != 0 else 0.1
        target_up = target_up / np.linalg.norm(target_up)
        
        # Create right direction (perpendicular to both)
        # For right-handed system: right = forward × up
        target_right = np.cross(target_forward, target_up)
        target_right = target_right / np.linalg.norm(target_right)
        
        # Recompute up to ensure perfect orthogonality
        # up = right × forward (for right-handed system)
        target_up = np.cross(target_right, target_forward)
        target_up = target_up / np.linalg.norm(target_up)
        
        # Build rotation matrix from these axes
        # Columns are: X (forward), Y (up), Z (right) for standard right-handed frame
        pelvis_matrix = np.column_stack([target_forward, target_up, target_right])
        
        # Verify determinant is positive (right-handed)
        det = np.linalg.det(pelvis_matrix)
        if det < 0:
            # Flip the right vector to make it right-handed
            target_right = -target_right
            pelvis_matrix = np.column_stack([target_forward, target_up, target_right])
        
        pelvis_calibrated = R.from_matrix(pelvis_matrix)
        
        calibrated['pelvis'] = pelvis_calibrated
        
        # For all other segments, calibrate relative to pelvis SENSOR (not calibrated pelvis)
        # This is crucial - we need the relative orientations in the sensor frame
        for seg, sensor_rot in sensor_orientations.items():
            if seg == 'pelvis':
                continue
            
            # Express sensor orientation relative to pelvis SENSOR
            sensor_rel_pelvis = pelvis_sensor.inv() * sensor_rot
            
            # Apply calibration in pelvis-relative frame
            calibrated_rel = self.offsets[seg] * sensor_rel_pelvis * self.alignments[seg]
            
            # Transform to world frame using calibrated pelvis
            calibrated[seg] = pelvis_calibrated * calibrated_rel
        
        return calibrated