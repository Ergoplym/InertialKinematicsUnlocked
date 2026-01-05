"""
OpenSim Export Module
Exports IMU data to OpenSim .sto format for Inverse Kinematics
"""

import os
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class OpenSimExporter:
    """Export IMU data to OpenSim .sto format for Rajagopal model."""
    
    # Mapping from our segment names to OpenSim IMU names (Rajagopal model)
    # Format: our_name -> opensim_imu_name
    SEGMENT_TO_IMU_MAPPING = {
        'pelvis': 'pelvis_imu',
        'trunk': 'torso_imu',
        'head': 'head_imu',
        'upper_arm_right': 'humerus_r_imu',
        'forearm_right': 'radius_r_imu',
        'hand_right': 'hand_r_imu',
        'upper_arm_left': 'humerus_l_imu',
        'forearm_left': 'radius_l_imu',
        'hand_left': 'hand_l_imu',
        'upper_leg_right': 'femur_r_imu',
        'lower_leg_right': 'tibia_r_imu',
        'foot_right': 'calcn_r_imu',
        'upper_leg_left': 'femur_l_imu',
        'lower_leg_left': 'tibia_l_imu',
        'foot_left': 'calcn_l_imu',
    }
    
    def __init__(self):
        pass
    
    def export_placer_files(self, calibration_csv_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Export placer (calibration) files from calibration CSV.
        
        Args:
            calibration_csv_path: Path to calibration CSV file
            output_dir: Directory to save .sto files
            
        Returns:
            Tuple of (sensor_sto_path, world_sto_path)
        """
        import pandas as pd
        
        print(f"[OpenSimExporter] Loading calibration: {os.path.basename(calibration_csv_path)}")
        
        # Load calibration CSV
        df = pd.read_csv(calibration_csv_path)
        
        # Extract timestamp from filename for naming
        basename = os.path.basename(calibration_csv_path).replace('.csv', '')
        
        # Extract orientations (both sensor and world frames)
        sensor_data = {}  # Raw sensor quaternions (raw_ prefix)
        world_data = {}   # World-transformed quaternions (world_ prefix)
        
        # Parse columns to find sensor and world quaternions
        # Format: raw_pelvis_qw, raw_pelvis_qx, raw_pelvis_qy, raw_pelvis_qz
        #         world_pelvis_qw, world_pelvis_qx, world_pelvis_qy, world_pelvis_qz
        
        for col in df.columns:
            if col.startswith('raw_') and col.endswith('_qw'):
                # Raw sensor quaternion
                segment_name = col.replace('raw_', '').replace('_qw', '')
                if segment_name in self.SEGMENT_TO_IMU_MAPPING:
                    imu_name = self.SEGMENT_TO_IMU_MAPPING[segment_name]
                    sensor_data[imu_name] = {
                        'qw': df[f'raw_{segment_name}_qw'].values,
                        'qx': df[f'raw_{segment_name}_qx'].values,
                        'qy': df[f'raw_{segment_name}_qy'].values,
                        'qz': df[f'raw_{segment_name}_qz'].values,
                    }
            
            if col.startswith('world_') and col.endswith('_qw'):
                # World-transformed quaternion
                segment_name = col.replace('world_', '').replace('_qw', '')
                if segment_name in self.SEGMENT_TO_IMU_MAPPING:
                    imu_name = self.SEGMENT_TO_IMU_MAPPING[segment_name]
                    world_data[imu_name] = {
                        'qw': df[f'world_{segment_name}_qw'].values,
                        'qx': df[f'world_{segment_name}_qx'].values,
                        'qy': df[f'world_{segment_name}_qy'].values,
                        'qz': df[f'world_{segment_name}_qz'].values,
                    }
        
        # Get time series (all frames)
        if 'time' in df.columns:
            times = df['time'].values
        elif 'timestamp' in df.columns:
            times = df['timestamp'].values
        else:
            times = np.arange(len(df)) * 0.01  # Default 100 Hz
        
        print(f"[OpenSimExporter] Found {len(sensor_data)} sensor IMUs, {len(world_data)} world IMUs")
        print(f"[OpenSimExporter] Calibration frames: {len(times)} (full time series)")
        
        # Write sensor placer file (all frames)
        sensor_path = os.path.join(output_dir, f'placer_{basename}_sensor.sto')
        self._write_sto(sensor_path, sensor_data, times)
        print(f"[OpenSimExporter] Created: {os.path.basename(sensor_path)}")
        
        # Write world placer file (all frames)
        world_path = os.path.join(output_dir, f'placer_{basename}_world.sto')
        self._write_sto(world_path, world_data, times)
        print(f"[OpenSimExporter] Created: {os.path.basename(world_path)}")
        
        return sensor_path, world_path
    
    def export_motion_files(self, recording_csv_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Export motion files from recording CSV.
        
        Args:
            recording_csv_path: Path to recording CSV file
            output_dir: Directory to save .sto files
            
        Returns:
            Tuple of (sensor_sto_path, world_sto_path)
        """
        import pandas as pd
        
        print(f"[OpenSimExporter] Loading recording: {os.path.basename(recording_csv_path)}")
        
        # Load recording CSV
        df = pd.read_csv(recording_csv_path)
        
        # Extract base name
        basename = os.path.basename(recording_csv_path).replace('.csv', '')
        if basename.startswith('recording_'):
            basename = basename[10:]  # Remove "recording_" prefix
        basename = basename.replace('_recording', '')  # Remove _recording if present
        
        # Extract orientations (both sensor and world frames)
        # Recording format: raw_ prefix for sensor, seg_ prefix for world
        sensor_data = {}  # Raw sensor quaternions
        world_data = {}   # World-transformed quaternions (segment orientations)
        
        # Parse columns
        for col in df.columns:
            if col.startswith('raw_') and col.endswith('_qw'):
                # Raw sensor quaternion
                segment_name = col.replace('raw_', '').replace('_qw', '')
                if segment_name in self.SEGMENT_TO_IMU_MAPPING:
                    imu_name = self.SEGMENT_TO_IMU_MAPPING[segment_name]
                    sensor_data[imu_name] = {
                        'qw': df[f'raw_{segment_name}_qw'].values,
                        'qx': df[f'raw_{segment_name}_qx'].values,
                        'qy': df[f'raw_{segment_name}_qy'].values,
                        'qz': df[f'raw_{segment_name}_qz'].values,
                    }
            
            if col.startswith('seg_') and col.endswith('_qw'):
                # World-transformed quaternion (segment orientation)
                segment_name = col.replace('seg_', '').replace('_qw', '')
                if segment_name in self.SEGMENT_TO_IMU_MAPPING:
                    imu_name = self.SEGMENT_TO_IMU_MAPPING[segment_name]
                    world_data[imu_name] = {
                        'qw': df[f'seg_{segment_name}_qw'].values,
                        'qx': df[f'seg_{segment_name}_qx'].values,
                        'qy': df[f'seg_{segment_name}_qy'].values,
                        'qz': df[f'seg_{segment_name}_qz'].values,
                    }
        
        # Get timestamps
        if 'time' in df.columns:
            times = df['time'].values
        elif 'timestamp' in df.columns:
            times = df['timestamp'].values
        else:
            # Generate timestamps based on frame count
            times = np.arange(len(df)) * 0.01  # Assume 100 Hz
        
        print(f"[OpenSimExporter] Found {len(sensor_data)} sensor IMUs, {len(world_data)} world IMUs")
        print(f"[OpenSimExporter] Time range: {times[0]:.3f}s to {times[-1]:.3f}s ({len(times)} frames)")
        
        # Write sensor motion file
        sensor_path = os.path.join(output_dir, f'motions_{basename}_sensor.sto')
        self._write_sto(sensor_path, sensor_data, times)
        print(f"[OpenSimExporter] Created: {os.path.basename(sensor_path)}")
        
        # Write world motion file
        world_path = os.path.join(output_dir, f'motions_{basename}_world.sto')
        self._write_sto(world_path, world_data, times)
        print(f"[OpenSimExporter] Created: {os.path.basename(world_path)}")
        
        return sensor_path, world_path
    
    def _write_sto(self, filepath: str, imu_data: Dict, times: np.ndarray):
        """
        Write .sto file (placer or motion - same format, full time series).

        Formatting is designed to match OpenSim IMU placer outputs:
        - Header lines:
            DataRate=<int>
            DataType=Quaternion
            version= 3
            OpenSimVersion=4.1
            endheader
        - Column names: time, then one column per IMU (not per quaternion component)
        - Data rows: time followed by a single cell per IMU containing:
            qw, qx, qy, qz   (comma + space separated)

        Notes:
        - time is written using Python's normal float string (no forced rounding)
        - quaternion components are written with up to 6 dp, trailing zeros stripped
        - missing / non-finite quaternions are written as a single dash: -
        """
        num_frames = len(times)

        # DataRate from median positive dt (more robust than using endpoints)
        if num_frames > 1:
            t = np.asarray(times, dtype=float)
            dts = np.diff(t)
            pos = dts[dts > 0]
            dt = float(np.median(pos)) if pos.size else (1.0 / 60.0)
            data_rate = int(round(1.0 / dt)) if dt > 0 else 60
        else:
            data_rate = 60

        # Ensure strictly increasing time vector (OpenSim can fail on duplicate/non-monotonic times)
        t_out = np.asarray(times, dtype=float).copy()
        if t_out.size >= 2:
            dt_est = 1.0 / float(data_rate) if float(data_rate) > 0 else None
            if np.any(np.diff(t_out) <= 0):
                t0 = float(t_out[0])
                if dt_est is None:
                    finite = t_out[np.isfinite(t_out)]
                    dt_est = float(np.median(np.diff(finite))) if finite.size > 2 else (1.0 / 60.0)
                t_out = t0 + np.arange(t_out.size, dtype=float) * float(dt_est)
                print("[OpenSimExporter] Warning: Non-increasing time stamps detected; rebuilt uniform time vector for OpenSim compatibility.")
        else:
            t_out = np.asarray(times, dtype=float)

        def fmt_time(x: float) -> str:
            return str(float(x))

        def fmt_q(x: float) -> str:
            s = f"{float(x):.6f}".rstrip("0").rstrip(".")
            return "0" if s == "-0" else s

        # Preserve deterministic column ordering:
        # If a canonical order exists on the class, use it; otherwise fall back to sorted keys.
        canonical = getattr(self, "CANONICAL_IMU_ORDER", None)
        if canonical:
            imu_names = [n for n in canonical if n in imu_data]
        else:
            imu_names = sorted(imu_data.keys())

        with open(filepath, "w", newline="\n") as f:
            f.write(f"DataRate={data_rate}\n")
            f.write("DataType=Quaternion\n")
            f.write("version= 3\n")
            f.write("OpenSimVersion=4.1\n")
            f.write("endheader\n")

            # Header row
            f.write("\t".join(["time"] + imu_names) + "\n")

            # Data rows
            for i in range(num_frames):
                row = [fmt_time(t_out[i])]
                for imu_name in imu_names:
                    quat = imu_data.get(imu_name)
                    if quat is None:
                        row.append("-")
                        continue

                    try:
                        qw = quat["qw"][i]
                        qx = quat["qx"][i]
                        qy = quat["qy"][i]
                        qz = quat["qz"][i]
                    except Exception:
                        row.append("-")
                        continue

                    vals = np.array([qw, qx, qy, qz], dtype=float)
                    if np.any(~np.isfinite(vals)):
                        row.append("-")
                        continue

                    row.append(f"{fmt_q(qw)}, {fmt_q(qx)}, {fmt_q(qy)}, {fmt_q(qz)}")

                f.write("\t".join(row) + "\n")




def export_trial_to_opensim(trial_folder: str, output_folder: str = None) -> Dict[str, List[str]]:
    """
    Export entire trial folder to OpenSim format.
    
    Creates placer files from calibrations and motion files from recordings.
    
    Args:
        trial_folder: Path to trial folder
        output_folder: Output directory (default: trial_folder/opensim_export)
        
    Returns:
        Dictionary with created files: {'placer_sensor': [...], 'placer_world': [...], 
                                        'motions_sensor': [...], 'motions_world': [...]}
    """
    if output_folder is None:
        output_folder = os.path.join(trial_folder, 'opensim_export')
    
    os.makedirs(output_folder, exist_ok=True)
    
    exporter = OpenSimExporter()
    created_files = {
        'placer_sensor': [],
        'placer_world': [],
        'motions_sensor': [],
        'motions_world': []
    }
    
    print(f"[OpenSimExporter] Exporting trial: {os.path.basename(trial_folder)}")
    print(f"[OpenSimExporter] Output folder: {output_folder}")
    
    # Find all calibration files
    calibration_files = []
    recording_files = []
    
    for filename in os.listdir(trial_folder):
        if filename.endswith('.csv'):
            if 'calibration' in filename.lower():
                calibration_files.append(os.path.join(trial_folder, filename))
            elif filename.startswith('recording') or 'recording' in filename:
                recording_files.append(os.path.join(trial_folder, filename))
    
    # Export calibration files (placer)
    print(f"\n[OpenSimExporter] Found {len(calibration_files)} calibration file(s)")
    for calib_file in sorted(calibration_files):
        sensor_path, world_path = exporter.export_placer_files(calib_file, output_folder)
        created_files['placer_sensor'].append(sensor_path)
        created_files['placer_world'].append(world_path)
    
    # Export recording files (motions)
    print(f"\n[OpenSimExporter] Found {len(recording_files)} recording file(s)")
    for rec_file in sorted(recording_files):
        sensor_path, world_path = exporter.export_motion_files(rec_file, output_folder)
        created_files['motions_sensor'].append(sensor_path)
        created_files['motions_world'].append(world_path)
    
    print(f"\n[OpenSimExporter] Export complete!")
    print(f"[OpenSimExporter] Created {sum(len(v) for v in created_files.values())} files")
    
    return created_files
