"""IMU data sources - real and fake"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional, Any
import traceback


class FakeIMUStream:
    """Simulates IMU data for testing"""
    
    def __init__(self):
        self.start_time = time.monotonic()
        
        # Simulate realistic sensor mounting orientations
        self.sensor_misalignments = {
            "pelvis": R.identity(),
            "trunk": R.from_euler('y', 10, degrees=True),
            "head": R.from_euler('y', -5, degrees=True),
            
            # Arms
            "upper_arm_right": R.from_euler('y', 15, degrees=True),
            "forearm_right": R.from_euler('y', -10, degrees=True),
            "hand_right": R.from_euler('y', 20, degrees=True),
            "upper_arm_left": R.from_euler('y', -15, degrees=True),
            "forearm_left": R.from_euler('y', 10, degrees=True),
            "hand_left": R.from_euler('y', -20, degrees=True),
            
            # Right leg
            "upper_leg_right": R.from_euler('y', 5, degrees=True),
            "lower_leg_right": R.from_euler('y', -10, degrees=True),
            "foot_right": R.from_euler('y', 15, degrees=True),
            
            # Left leg - 180° flip to simulate the mirroring problem
            "upper_leg_left": R.from_euler('xyz', [0, 5, 180], degrees=True),
            "lower_leg_left": R.from_euler('xyz', [0, -10, 180], degrees=True),
            "foot_left": R.from_euler('xyz', [0, 15, 180], degrees=True),
        }
    
    def get_sample(self) -> Dict[str, np.ndarray]:
        """Returns {segment_name: [w, x, y, z]}"""
        t = time.monotonic() - self.start_time
        
        # Phase 1: Static neutral pose (0-10s)
        if t < 10.0:
            pelvis_tilt = 0
            trunk_flex = 0
            head_flex = 0
            hip_r_flex = 0
            knee_r_flex = 0
            hip_l_flex = 0
            knee_l_flex = 0
            shoulder_r_flex = 0
            elbow_r_flex = 0
            shoulder_l_flex = 0
            elbow_l_flex = 0
            
        # Phase 2: Squatting motion (10-20s)
        elif t < 20.0:
            t_squat = t - 10.0
            squat_cycle = np.sin(2 * np.pi * 0.5 * t_squat)
            
            squat_depth = 60 * max(0, squat_cycle)
            hip_r_flex = squat_depth * 0.5
            knee_r_flex = squat_depth
            hip_l_flex = squat_depth * 0.5
            knee_l_flex = squat_depth
            
            pelvis_tilt = -10 * max(0, squat_cycle)
            trunk_flex = 15 * max(0, squat_cycle)
            head_flex = 5 * max(0, squat_cycle)
            
            # Arms mostly still during squat
            shoulder_r_flex = 10
            elbow_r_flex = 30
            shoulder_l_flex = 10
            elbow_l_flex = 30
            
        # Phase 3: Trunk and head movements (20-30s)
        elif t < 30.0:
            t_movement = t - 20.0
            
            # Trunk lateral bend
            trunk_flex = 10 * np.sin(0.5 * t_movement)
            pelvis_tilt = 5 * np.sin(0.3 * t_movement)
            
            # Head movements
            head_flex = 20 * np.sin(0.8 * t_movement)  # Nodding
            
            # Legs stationary
            hip_r_flex = 10
            knee_r_flex = 5
            hip_l_flex = 10
            knee_l_flex = 5
            
            # Arms
            shoulder_r_flex = 10
            elbow_r_flex = 30
            shoulder_l_flex = 10
            elbow_l_flex = 30
            
        # Phase 4: Arm movements (30-40s)
        elif t < 40.0:
            t_arm = t - 30.0
            
            # Arm flexion
            shoulder_r_flex = 20 + 30 * np.sin(1.0 * t_arm)
            elbow_r_flex = 30 + 40 * np.sin(1.0 * t_arm + 0.3)
            shoulder_l_flex = 20 + 30 * np.sin(1.0 * t_arm + np.pi)
            elbow_l_flex = 30 + 40 * np.sin(1.0 * t_arm + np.pi + 0.3)
            
            # Trunk mostly still
            trunk_flex = 5 * np.sin(0.2 * t_arm)
            pelvis_tilt = 0
            head_flex = 0
            
            # Legs
            hip_r_flex = 10
            knee_r_flex = 5
            hip_l_flex = 10
            knee_l_flex = 5
            
        # Phase 5: Walking motion (40-60s)
        else:
            t_walk = t - 40.0
            
            pelvis_tilt = 5 * np.sin(0.3 * t_walk)
            trunk_flex = 10 * np.sin(0.4 * t_walk)
            head_flex = 5 * np.sin(0.5 * t_walk)
            
            # Right leg - walking motion
            hip_r_flex = 20 * np.sin(1.6 * t_walk)
            knee_r_flex = max(0, 40 * np.sin(1.6 * t_walk + 0.5))
            
            # Left leg - opposite phase
            hip_l_flex = 20 * np.sin(1.6 * t_walk + np.pi)
            knee_l_flex = max(0, 40 * np.sin(1.6 * t_walk + np.pi + 0.5))
            
            # Arms in anti-phase with legs
            shoulder_r_flex = 10 + 15 * np.sin(1.6 * t_walk + np.pi)
            elbow_r_flex = 30 + 20 * np.sin(1.6 * t_walk + np.pi + 0.3)
            shoulder_l_flex = 10 + 15 * np.sin(1.6 * t_walk)
            elbow_l_flex = 30 + 20 * np.sin(1.6 * t_walk + 0.3)
        
        # Build rotations
        pelvis_rot = R.from_euler('x', pelvis_tilt, degrees=True)
        trunk_rel = R.from_euler('x', trunk_flex, degrees=True)
        trunk_rot = pelvis_rot * trunk_rel
        head_rel = R.from_euler('x', head_flex, degrees=True)
        head_rot = trunk_rot * head_rel
        
        # Right arm
        shoulder_r_rel = R.from_euler('x', shoulder_r_flex, degrees=True)
        upper_arm_r = trunk_rot * shoulder_r_rel
        elbow_r_rel = R.from_euler('x', elbow_r_flex, degrees=True)
        forearm_r = upper_arm_r * elbow_r_rel
        hand_r = forearm_r
        
        # Left arm
        shoulder_l_rel = R.from_euler('x', shoulder_l_flex, degrees=True)
        upper_arm_l = trunk_rot * shoulder_l_rel
        elbow_l_rel = R.from_euler('x', elbow_l_flex, degrees=True)
        forearm_l = upper_arm_l * elbow_l_rel
        hand_l = forearm_l
        
        # Right leg
        hip_r_rel = R.from_euler('z', hip_r_flex, degrees=True)
        upper_leg_r = pelvis_rot * hip_r_rel
        knee_r_rel = R.from_euler('z', knee_r_flex, degrees=True)
        lower_leg_r = upper_leg_r * knee_r_rel
        foot_r = lower_leg_r
        
        # Left leg
        hip_l_rel = R.from_euler('z', hip_l_flex, degrees=True)
        upper_leg_l = pelvis_rot * hip_l_rel
        knee_l_rel = R.from_euler('z', knee_l_flex, degrees=True)
        lower_leg_l = upper_leg_l * knee_l_rel
        foot_l = lower_leg_l
        
        # True anatomical rotations
        true_rots = {
            "pelvis": pelvis_rot,
            "trunk": trunk_rot,
            "head": head_rot,
            "upper_arm_right": upper_arm_r,
            "forearm_right": forearm_r,
            "hand_right": hand_r,
            "upper_arm_left": upper_arm_l,
            "forearm_left": forearm_l,
            "hand_left": hand_l,
            "upper_leg_right": upper_leg_r,
            "lower_leg_right": lower_leg_r,
            "foot_right": foot_r,
            "upper_leg_left": upper_leg_l,
            "lower_leg_left": lower_leg_l,
            "foot_left": foot_l,
        }
        
        # Apply sensor misalignments
        sample = {}
        for name, true_rot in true_rots.items():
            sensor_reading = true_rot * self.sensor_misalignments[name]
            x, y, z, w = sensor_reading.as_quat()
            sample[name] = np.array([w, x, y, z], dtype=float)
        
        return sample
    
    def stop(self):
        """Clean shutdown for compatibility"""
        pass


class IMUStreamFactory:
    """Factory for creating IMU streams"""
    
    @staticmethod
    def create_stream(use_real_imu: bool = True, radio_channel: int = 11):
        """
        Create IMU stream, attempting real hardware first.
        
        Args:
            use_real_imu: If True, try to connect to real hardware first
                         If False, use fake stream immediately
            radio_channel: Awinda radio channel (11-25), default 11
        """
        if not use_real_imu:
            print("[IMU] Using fake IMU stream (by request).")
            return FakeIMUStream()
        
        # Try to import and use real Awinda stream
        real_stream = IMUStreamFactory._try_real_stream(radio_channel)
        if real_stream:
            return real_stream
        
        # Fall back to fake stream
        print("[IMU] Using fake IMU stream (hardware not available).")
        return FakeIMUStream()
    
    @staticmethod
    def _try_real_stream(radio_channel: int = 11) -> Optional[Any]:
        """Try to create a real Awinda stream."""
        awinda_modules = [
            "awindastream",  # Lowercase version
            "Awindastream",  # Original case version
        ]
        
        for module_name in awinda_modules:
            try:
                print(f"[IMU] Trying to import {module_name}...")
                module = __import__(module_name, fromlist=['AwindaStream'])
                AwindaStream = module.AwindaStream
                
                print(f"[IMU] Creating AwindaStream from {module_name}...")
                stream = AwindaStream(
                    mapping_csv="mapping.csv",
                    radio_channel=radio_channel,
                    timeout=2.0,
                    sensor_connect_wait=2.0,
                )
                
                print("[IMU] Starting Awinda stream...")
                stream.start()
                
                # Test connection
                print("[IMU] Testing connection...")
                quats = stream.get_latest_quats()
                if quats:
                    print(f"[IMU] Awinda connected successfully. Got {len(quats)} sensor readings.")
                    return stream
                else:
                    print("[IMU] No data from Awinda stream.")
                    if hasattr(stream, 'stop'):
                        stream.stop()
                    
            except ImportError as e:
                print(f"[IMU] Could not import {module_name}: {e}")
                continue
            except Exception as e:
                print(f"[IMU] Error connecting to Awinda from {module_name}: {e}")
                traceback.print_exc()
                continue
        
        print("[IMU] No real IMU stream available.")
        return None