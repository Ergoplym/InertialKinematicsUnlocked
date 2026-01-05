"""
Trial Viewer for IKU - Load and playback recorded trials.
Mirrors the main recorder UI layout with playback controls.
"""

import os
import csv
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, QTimer

from visualization_optimized import Skeleton3DCanvas, CompactAngleTableWidget
from config import BODY_MODEL
from kinematics import KinematicsEngine
from anthropometry import AnthropometryProfile


class TrialData:
    """Container for loaded trial data."""
    
    def __init__(self):
        self.trial_folder = ""
        self.subject_id = ""
        self.session_id = ""
        
        # Time series data
        self.timestamps = []  # List of timestamps (seconds)
        self.segment_orientations = []  # List of dicts {segment_name: Rotation}
        self.joint_angles = []  # List of dicts {joint_name: angle}
        self.segment_positions = []  # List of dicts {segment_name: [x,y,z]}
        
        # Metadata
        self.fps = 100.0
        self.num_frames = 0
        self.duration = 0.0
        self.body_config = "whole_body"
        self.anthro_profile = None
        
        # Video files
        self.video_files = []
        
        # Camera timestamps (for video sync)
        self.camera_timestamps = {}  # {video_index: [timestamps]}
        
        # Available recordings in trial folder
        self.available_recordings = []  # List of (display_name, filepath) tuples
        self.current_recording_path = None
    
    def load_from_folder(self, folder_path: str) -> bool:
        """
        Load trial data from folder.
        
        Args:
            folder_path: Path to trial folder
            
        Returns:
            True if successful, False otherwise
        """
        self.trial_folder = folder_path
        
        if not os.path.exists(folder_path):
            print(f"[TrialData] Folder not found: {folder_path}")
            return False
        
        # Load session info
        session_info_path = os.path.join(folder_path, "session_info.json")
        if os.path.exists(session_info_path):
            try:
                with open(session_info_path, 'r') as f:
                    session_info = json.load(f)
                    self.subject_id = session_info.get('subject_id', 'Unknown')
                    self.session_id = session_info.get('session_id', 'Unknown')
                    self.body_config = session_info.get('body_config', 'whole_body')
            except Exception as e:
                print(f"[TrialData] Could not load session_info.json: {e}")
        
        # Find CSV files - look for all recording*.csv files
        csv_files = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv') and not filename.startswith('.'):
                # Skip calibration files
                if 'calibration' in filename.lower():
                    continue
                csv_files.append(os.path.join(folder_path, filename))
        
        if not csv_files:
            print(f"[TrialData] No recording CSV files found in {folder_path}")
            return False
        
        # Sort by filename (handles _1, _2, _3 numbering)
        csv_files.sort()
        
        # Store all available recordings
        self.available_recordings = []
        for i, csv_path in enumerate(csv_files, 1):
            filename = os.path.basename(csv_path)
            
            # Create display name based on actual filename
            # recording_P001_Squat_20250104_143022.csv → "Recording"
            # recording_P001_Squat_20250104_143022_1.csv → "Recording_1"
            # recording_P001_Squat_20250104_143022_2.csv → "Recording_2"
            
            import re
            # Look for _X.csv at the end where X is a number
            match = re.search(r'_(\d+)\.csv$', filename)
            if match:
                num = match.group(1)
                display_name = f"Recording_{num}"
            else:
                # No number = first recording
                display_name = "Recording"
            
            self.available_recordings.append((display_name, csv_path))
        
        print(f"[TrialData] Found {len(self.available_recordings)} recording(s)")
        for display_name, path in self.available_recordings:
            print(f"  - {display_name}: {os.path.basename(path)}")
        
        # Load the first recording
        csv_path = self.available_recordings[0][1]
        self.current_recording_path = csv_path
        print(f"[TrialData] Loading data from: {os.path.basename(csv_path)}")
        
        # Load CSV with joint data
        if not self._load_joint_data_csv(csv_path):
            return False
        
        # Find video files
        self._find_videos(folder_path)
        
        # Calculate metadata
        self.num_frames = len(self.timestamps)
        if self.num_frames > 1:
            self.duration = self.timestamps[-1] - self.timestamps[0]
            self.fps = self.num_frames / self.duration if self.duration > 0 else 100.0
        
        print(f"[TrialData] Loaded trial: {self.subject_id}/{self.session_id}")
        print(f"[TrialData] Frames: {self.num_frames}, Duration: {self.duration:.2f}s, FPS: {self.fps:.1f}")
        print(f"[TrialData] Videos: {len(self.video_files)}")
        
        return True
    
    def _load_joint_data_csv(self, filepath: str) -> bool:
        """Load joint data from CSV (handles actual IKU recording format)."""
        if not os.path.exists(filepath):
            print(f"[TrialData] CSV file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                
                # Get column names to understand format
                fieldnames = reader.fieldnames
                print(f"[TrialData] CSV has {len(fieldnames)} columns")
                
                # Check if we have segment orientations (new format) or only joints (old format)
                has_segment_orientations = any('seg_' in col for col in fieldnames)
                
                for row in reader:
                    # Parse timestamp
                    try:
                        timestamp = float(row.get('time', 0.0))
                        self.timestamps.append(timestamp)
                    except (ValueError, TypeError):
                        print(f"[TrialData] Skipping row with invalid time")
                        continue
                    
                    # Parse segment orientations if available
                    orientations = {}
                    if has_segment_orientations:
                        # New format with seg_ prefix
                        for segment in BODY_MODEL.keys():
                            qw = row.get(f'seg_{segment}_qw', '1.0')
                            qx = row.get(f'seg_{segment}_qx', '0.0')
                            qy = row.get(f'seg_{segment}_qy', '0.0')
                            qz = row.get(f'seg_{segment}_qz', '0.0')
                            
                            try:
                                # Create Rotation object (scipy uses x,y,z,w order)
                                orientations[segment] = R.from_quat([
                                    float(qx), float(qy), float(qz), float(qw)
                                ])
                            except (ValueError, TypeError):
                                # Use identity if invalid
                                orientations[segment] = R.from_quat([0, 0, 0, 1])
                    else:
                        # Old format - use identity rotations
                        for segment in BODY_MODEL.keys():
                            orientations[segment] = R.from_quat([0, 0, 0, 1])
                    
                    self.segment_orientations.append(orientations)
                    
                    # Parse euler angles (joint angles in degrees)
                    angles = {}
                    for key, value in row.items():
                        if 'euler' in key and value:
                            try:
                                # Clean up the key for better display
                                clean_key = key.replace('joint_', '')  # Remove joint_ prefix if present
                                # The key is like "forearm_left_hand_left_euler_0"
                                # Keep it as-is for now - angle table should handle it
                                angles[clean_key] = float(value)
                            except (ValueError, TypeError):
                                angles[clean_key] = 0.0
                    
                    self.joint_angles.append(angles)
            
            print(f"[TrialData] Loaded {len(self.timestamps)} frames")
            if len(self.joint_angles) > 0:
                print(f"[TrialData] Loaded {len(self.joint_angles)} angle frames")
                print(f"[TrialData] Sample angles from first frame: {list(self.joint_angles[0].keys())[:5]}")
            else:
                print(f"[TrialData] WARNING: No joint angles loaded!")
            
            if has_segment_orientations:
                print(f"[TrialData] ✓ Loaded segment orientations (new format)")
            else:
                print(f"[TrialData] ⚠ No segment orientations found (old format) - skeleton won't move")
            return True
            
        except Exception as e:
            print(f"[TrialData] Error loading CSV: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_recording_videos(self, csv_path: str):
        """
        Load video and timestamp files that match a specific recording CSV.
        
        Args:
            csv_path: Path to recording CSV file
        """
        # Clear existing video data
        self.video_files = []
        self.camera_timestamps = {}
        
        # Extract base name from CSV
        # recording_P001_Squat_20250104_143022.csv → P001_Squat_20250104_143022
        # recording_P001_Squat_20250104_143022_1.csv → P001_Squat_20250104_143022_1
        csv_filename = os.path.basename(csv_path)
        
        print(f"[TrialData] CSV filename: {csv_filename}")
        
        # Remove "recording_" prefix and ".csv" suffix
        if csv_filename.startswith('recording_'):
            base_name = csv_filename[10:]  # Remove "recording_"
        else:
            base_name = csv_filename
        
        base_name = base_name.replace('.csv', '')  # Remove .csv
        
        # Remove "_recording" anywhere in the name (videos don't have this)
        # "Alex Bush_Sit to stand_20260105_100813_recording" → "Alex Bush_Sit to stand_20260105_100813"  
        # "Alex Bush_Sit to stand_20260105_100813_recording_1" → "Alex Bush_Sit to stand_20260105_100813_1"
        base_name_for_video = base_name.replace('_recording', '')
        
        print(f"[TrialData] Extracted base name: {base_name}")
        print(f"[TrialData] Base name for video matching: {base_name_for_video}")
        print(f"[TrialData] Looking for videos matching: {base_name_for_video}")
        
        # Look for matching video files (camera_0_*.mp4, camera_1_*.mp4, etc.)
        folder_path = os.path.dirname(csv_path)
        
        print(f"[TrialData] Searching in folder: {folder_path}")
        
        # List all files in folder
        all_files = os.listdir(folder_path)
        video_files_in_folder = [f for f in all_files if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"[TrialData] All video files in folder: {video_files_in_folder}")
        
        for camera_id in range(10):  # Support up to 10 cameras
            # Try different naming patterns
            # Handle both: camera_0_Base_1.mp4 AND Base_camera_0_1.mp4
            patterns = [
                f"camera_{camera_id}_{base_name_for_video}.mp4",  # camera_0_Alex Bush_Sit to stand_20260105_100813.mp4
                f"{base_name_for_video}_camera_{camera_id}.mp4",  # Alex Bush_Sit to stand_20260105_100813_camera_0.mp4
                f"camera{camera_id}_{base_name_for_video}.mp4",   # camera0_Alex Bush_Sit to stand_20260105_100813.mp4
            ]
            
            # If base_name_for_video ends with _N (like _1, _2), also try with _camera_0_N pattern
            import re
            match = re.search(r'_(\d+)$', base_name_for_video)
            if match:
                # base_name_for_video = "Alex Bush_Sit to stand_20260105_100813_1"
                # Extract: base = "Alex Bush_Sit to stand_20260105_100813", num = "1"
                num = match.group(1)
                base_without_num = base_name_for_video.rsplit(f'_{num}', 1)[0]
                # Also try: "Alex Bush_Sit to stand_20260105_100813_camera_0_1.mp4"
                patterns.append(f"{base_without_num}_camera_{camera_id}_{num}.mp4")
            
            print(f"[TrialData] Trying patterns for camera {camera_id}: {patterns}")
            
            for pattern in patterns:
                video_path = os.path.join(folder_path, pattern)
                
                print(f"[TrialData]   Checking: {pattern} - exists: {os.path.exists(video_path)}")
                
                if os.path.exists(video_path):
                    self.video_files.append(video_path)
                    print(f"[TrialData] ✓ Found video: {os.path.basename(video_path)}")
                    
                    # Look for matching timestamp file
                    timestamp_patterns = [
                        pattern.replace('.mp4', '_timestamps.txt'),
                        f"camera_{camera_id}_{base_name_for_video}_timestamps.txt",
                        f"{base_name_for_video}_camera_{camera_id}_timestamps.txt",
                    ]
                    
                    print(f"[TrialData]   Looking for timestamp files: {timestamp_patterns}")
                    
                    for ts_pattern in timestamp_patterns:
                        ts_path = os.path.join(folder_path, ts_pattern)
                        print(f"[TrialData]     Checking: {ts_pattern} - exists: {os.path.exists(ts_path)}")
                        
                        if os.path.exists(ts_path):
                            # Load timestamps
                            try:
                                timestamps = []
                                with open(ts_path, 'r') as f:
                                    next(f)  # Skip header
                                    for line in f:
                                        parts = line.strip().split(',')
                                        if len(parts) >= 2:
                                            timestamps.append(float(parts[1]))
                                
                                video_index = len(self.video_files) - 1
                                self.camera_timestamps[video_index] = timestamps
                                print(f"[TrialData] ✓ Loaded {len(timestamps)} timestamps: {os.path.basename(ts_path)}")
                                
                                if len(timestamps) > 0:
                                    print(f"[TrialData]   Camera time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s")
                                break
                            except Exception as e:
                                print(f"[TrialData] ✗ Error loading timestamps: {e}")
                    
                    break  # Found video for this camera, move to next
        
        if len(self.video_files) == 0:
            print(f"[TrialData] ✗ No videos found for this recording")
            print(f"[TrialData] Expected patterns like: camera_0_{base_name_for_video}.mp4 or {base_name_for_video}_camera_0.mp4")
        else:
            print(f"[TrialData] ✓ Loaded {len(self.video_files)} video(s)")
    
    def _find_videos(self, folder_path: str):
        """Find videos for initial trial load - delegates to load_recording_videos."""
        if self.current_recording_path:
            self.load_recording_videos(self.current_recording_path)
        """Find video files in trial folder."""
        video_extensions = ['.mp4', '.avi', '.mov']
        
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in video_extensions:
                full_path = os.path.join(folder_path, filename)
                self.video_files.append(full_path)
        
        self.video_files.sort()  # Consistent ordering
        
        # Load camera timestamps for each video
        for i, video_path in enumerate(self.video_files):
            # Look for matching timestamp file
            # e.g., "camera_0.mp4" → "camera_0_timestamps.txt"
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp_file = os.path.join(folder_path, f"{base_name}_timestamps.txt")
            
            if os.path.exists(timestamp_file):
                try:
                    timestamps = []
                    with open(timestamp_file, 'r') as f:
                        # Skip header
                        next(f)
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) == 2:
                                timestamps.append(float(parts[1]))
                    
                    self.camera_timestamps[i] = timestamps
                    print(f"[TrialData] Loaded {len(timestamps)} camera timestamps for video {i}")
                    if len(timestamps) > 0:
                        print(f"[TrialData] Camera time range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s")
                except Exception as e:
                    print(f"[TrialData] Error loading camera timestamps: {e}")
            else:
                print(f"[TrialData] No timestamp file found: {timestamp_file}")
    
    def get_frame(self, frame_idx: int) -> Tuple[Dict, Dict]:
        """
        Get data for specific frame.
        
        Args:
            frame_idx: Frame index (0 to num_frames-1)
            
        Returns:
            (orientations_dict, angles_dict) for that frame
        """
        if 0 <= frame_idx < self.num_frames:
            orientations = self.segment_orientations[frame_idx]
            angles = self.joint_angles[frame_idx] if frame_idx < len(self.joint_angles) else {}
            return orientations, angles
        
        return {}, {}


class TrialViewerWindow(QtWidgets.QMainWindow):
    """Trial viewer window - mirrors recorder UI layout."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("IKU Trial Viewer")
        self.showMaximized()  # Open maximized
        
        # Trial data
        self.trial_data = TrialData()
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
        
        # Kinematics for computing positions
        self.kinematics = KinematicsEngine(BODY_MODEL)
        
        # Playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._advance_frame)
        
        # Video display using OpenCV (more reliable than QMediaPlayer)
        self.video_captures = []     # cv2.VideoCapture objects
        self.video_labels = []        # QLabel widgets to display frames
        self.current_video_frame = 0  # Current frame number in video
        
        # Build UI
        self._init_ui()
    
    def _init_ui(self):
        """Build the UI - mirrors recorder layout."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setSpacing(10)
        
        # ==================== LEFT COLUMN ====================
        left_column = QtWidgets.QVBoxLayout()
        left_column.setSpacing(10)
        
        # Logo
        logo_label = QtWidgets.QLabel("IKU")
        logo_label.setStyleSheet(
            "font-size: 48pt; font-weight: bold; color: #2c3e50; "
            "padding: 20px; background-color: white; border-radius: 10px;"
        )
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        left_column.addWidget(logo_label)
        
        # Trial Info Display
        trial_info_group = QtWidgets.QGroupBox("Trial Information")
        trial_info_layout = QtWidgets.QVBoxLayout()
        
        self.trial_info_label = QtWidgets.QLabel("No trial loaded")
        self.trial_info_label.setStyleSheet("padding: 10px; font-size: 10pt;")
        self.trial_info_label.setWordWrap(True)
        trial_info_layout.addWidget(self.trial_info_label)
        
        # Recording selector (for multiple recordings in same trial)
        recording_select_layout = QtWidgets.QHBoxLayout()
        recording_label = QtWidgets.QLabel("Recording:")
        recording_label.setStyleSheet("font-size: 9pt;")
        recording_select_layout.addWidget(recording_label)
        
        self.recording_combo = QtWidgets.QComboBox()
        self.recording_combo.setEnabled(False)
        self.recording_combo.currentIndexChanged.connect(self._on_recording_changed)
        recording_select_layout.addWidget(self.recording_combo)
        
        trial_info_layout.addLayout(recording_select_layout)
        
        trial_info_group.setLayout(trial_info_layout)
        left_column.addWidget(trial_info_group)
        
        # Load Trial Button
        load_btn = QtWidgets.QPushButton("📂 Load Trial")
        load_btn.setStyleSheet(
            "padding: 12px; font-size: 11pt; font-weight: bold; "
            "background-color: #3498db; color: white; border-radius: 5px;"
        )
        load_btn.clicked.connect(self._load_trial)
        left_column.addWidget(load_btn)
        
        # Playback Controls
        playback_group = QtWidgets.QGroupBox("Playback Controls")
        playback_layout = QtWidgets.QVBoxLayout()
        
        # Play/Pause button
        self.play_pause_btn = QtWidgets.QPushButton("▶ Play")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setStyleSheet("padding: 10px; font-size: 10pt; font-weight: bold;")
        self.play_pause_btn.clicked.connect(self._toggle_play_pause)
        playback_layout.addWidget(self.play_pause_btn)
        
        # Step buttons
        step_layout = QtWidgets.QHBoxLayout()
        
        step_back_btn = QtWidgets.QPushButton("◀ Step")
        step_back_btn.setEnabled(False)
        step_back_btn.clicked.connect(lambda: self._step_frame(-1))
        step_layout.addWidget(step_back_btn)
        self.step_back_btn = step_back_btn
        
        step_fwd_btn = QtWidgets.QPushButton("Step ▶")
        step_fwd_btn.setEnabled(False)
        step_fwd_btn.clicked.connect(lambda: self._step_frame(1))
        step_layout.addWidget(step_fwd_btn)
        self.step_fwd_btn = step_fwd_btn
        
        playback_layout.addLayout(step_layout)
        
        # Frame info
        self.frame_info_label = QtWidgets.QLabel("Frame: 0 / 0")
        self.frame_info_label.setStyleSheet("padding: 5px; font-family: 'Courier New';")
        playback_layout.addWidget(self.frame_info_label)
        
        playback_group.setLayout(playback_layout)
        left_column.addWidget(playback_group)
        
        # Timeline Slider
        timeline_group = QtWidgets.QGroupBox("Timeline")
        timeline_layout = QtWidgets.QVBoxLayout()
        
        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self._on_timeline_changed)
        timeline_layout.addWidget(self.timeline_slider)
        
        self.time_label = QtWidgets.QLabel("Time: 0.00s / 0.00s")
        self.time_label.setStyleSheet("font-family: 'Courier New'; font-size: 9pt;")
        timeline_layout.addWidget(self.time_label)
        
        timeline_group.setLayout(timeline_layout)
        left_column.addWidget(timeline_group)
        
        left_column.addStretch()
        
        left_column.addStretch()  # Push export button to bottom
        
        # OpenSim Export Button (at bottom)
        self.export_opensim_btn = QtWidgets.QPushButton("📤 Export to OpenSim")
        self.export_opensim_btn.setEnabled(False)
        self._update_opensim_button_style(False)  # Start gray (disabled)
        self.export_opensim_btn.setToolTip(
            "Export trial data to OpenSim .sto format\n"
            "Creates placer (calibration) and motion files\n"
            "Compatible with Rajagopal model"
        )
        self.export_opensim_btn.clicked.connect(self._export_to_opensim)
        left_column.addWidget(self.export_opensim_btn)
        
        # Add left column to main layout
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_column)
        left_widget.setFixedWidth(300)
        main_layout.addWidget(left_widget)
        
        # ==================== MIDDLE/RIGHT COLUMNS ====================
        right_area = QtWidgets.QHBoxLayout()  # ← HORIZONTAL not vertical!
        right_area.setSpacing(10)
        
        # Left side: Skeleton with video below
        left_side = QtWidgets.QVBoxLayout()
        
        # Skeleton visualization
        self.skeleton_canvas = Skeleton3DCanvas()
        self.skeleton_canvas.setMinimumSize(500, 400)
        left_side.addWidget(self.skeleton_canvas, stretch=2)
        
        # Video area (always visible, shows placeholder when no video)
        self.video_container = QtWidgets.QWidget()
        self.video_container.setMinimumHeight(240)  # Ensure height
        self.video_layout = QtWidgets.QHBoxLayout(self.video_container)
        self.video_layout.setSpacing(5)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add placeholder label for when no video
        self.no_video_label = QtWidgets.QLabel("No video available")
        self.no_video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.no_video_label.setStyleSheet(
            "background-color: #2c3e50; color: #95a5a6; "
            "font-size: 14pt; padding: 40px; border-radius: 5px;"
        )
        self.video_layout.addWidget(self.no_video_label)
        
        left_side.addWidget(self.video_container, stretch=1)
        
        # Right side: Joint angle table
        self.angle_table = CompactAngleTableWidget()
        self.angle_table.setMinimumWidth(400)
        
        # Add both sides to right area (HORIZONTAL)
        right_area.addLayout(left_side, stretch=2)
        right_area.addWidget(self.angle_table, stretch=1)
        
        main_layout.addLayout(right_area, stretch=1)
    
    def _load_trial(self):
        """Open dialog to select and load trial folder."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Trial Folder",
            "recordings",
            QtWidgets.QFileDialog.ShowDirsOnly
        )
        
        if folder:
            # Stop playback if running
            if self.is_playing:
                self._toggle_play_pause()
            
            # Load trial data
            if self.trial_data.load_from_folder(folder):
                self._on_trial_loaded()
            else:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Failed to load trial from:\n{folder}\n\nCheck that the folder contains valid trial data."
                )
    
    def _on_trial_loaded(self):
        """Handle successful trial load."""
        # Update trial info display
        info_text = f"<b>Subject:</b> {self.trial_data.subject_id}<br>"
        info_text += f"<b>Session:</b> {self.trial_data.session_id}<br>"
        info_text += f"<b>Frames:</b> {self.trial_data.num_frames}<br>"
        info_text += f"<b>Duration:</b> {self.trial_data.duration:.2f}s<br>"
        info_text += f"<b>FPS:</b> {self.trial_data.fps:.1f}"
        self.trial_info_label.setText(info_text)
        
        # Populate recording selector
        self.recording_combo.clear()
        for display_name, filepath in self.trial_data.available_recordings:
            self.recording_combo.addItem(display_name, filepath)
        
        # Enable recording selector if multiple recordings
        if len(self.trial_data.available_recordings) > 1:
            self.recording_combo.setEnabled(True)
        else:
            self.recording_combo.setEnabled(False)
        
        # Enable controls
        self.play_pause_btn.setEnabled(True)
        self.step_back_btn.setEnabled(True)
        self.step_fwd_btn.setEnabled(True)
        self.timeline_slider.setEnabled(True)
        self._update_opensim_button_style(True)  # Enable and update style
        
        # Setup timeline
        self.timeline_slider.setMaximum(self.trial_data.num_frames - 1)
        self.timeline_slider.setValue(0)
        
        # Load videos
        self._setup_videos()
        
        # Show first frame
        self.current_frame = 0
        self._update_display()
        
        print(f"[TrialViewer] Trial loaded successfully")
    
    def _on_recording_changed(self, index):
        """Handle recording selection change."""
        if index < 0 or index >= len(self.trial_data.available_recordings):
            return
        
        # Stop playback
        if self.is_playing:
            self._toggle_play_pause()
        
        # Get selected recording path
        display_name, filepath = self.trial_data.available_recordings[index]
        
        print(f"[TrialViewer] Switching to: {display_name}")
        
        # Load the selected recording
        if self._load_recording(filepath):
            # Reset playback position
            self.current_frame = 0
            self.timeline_slider.setMaximum(self.trial_data.num_frames - 1)
            self.timeline_slider.setValue(0)
            self._update_display()
            
            # Update info display
            info_text = f"<b>Subject:</b> {self.trial_data.subject_id}<br>"
            info_text += f"<b>Session:</b> {self.trial_data.session_id}<br>"
            info_text += f"<b>Frames:</b> {self.trial_data.num_frames}<br>"
            info_text += f"<b>Duration:</b> {self.trial_data.duration:.2f}s<br>"
            info_text += f"<b>FPS:</b> {self.trial_data.fps:.1f}"
            self.trial_info_label.setText(info_text)
    
    def _load_recording(self, filepath: str) -> bool:
        """Load a specific recording CSV file."""
        # Clear existing data
        self.trial_data.timestamps = []
        self.trial_data.segment_orientations = []
        self.trial_data.joint_angles = []
        self.trial_data.current_recording_path = filepath
        
        # Load CSV
        if not self.trial_data._load_joint_data_csv(filepath):
            return False
        
        # Load matching video and timestamps
        self.trial_data.load_recording_videos(filepath)
        
        # Recalculate metadata
        self.trial_data.num_frames = len(self.trial_data.timestamps)
        if self.trial_data.num_frames > 1:
            self.trial_data.duration = self.trial_data.timestamps[-1] - self.trial_data.timestamps[0]
            self.trial_data.fps = self.trial_data.num_frames / self.trial_data.duration if self.trial_data.duration > 0 else 100.0
        
        # Reload videos in UI
        self._setup_videos()
        
        return True
        """Load a specific recording CSV file."""
        # Clear existing data
        self.trial_data.timestamps = []
        self.trial_data.segment_orientations = []
        self.trial_data.joint_angles = []
        self.trial_data.current_recording_path = filepath
        
        # Load CSV
        if not self.trial_data._load_joint_data_csv(filepath):
            return False
        
        # Recalculate metadata
        self.trial_data.num_frames = len(self.trial_data.timestamps)
        if self.trial_data.num_frames > 1:
            self.trial_data.duration = self.trial_data.timestamps[-1] - self.trial_data.timestamps[0]
            self.trial_data.fps = self.trial_data.num_frames / self.trial_data.duration if self.trial_data.duration > 0 else 100.0
        
        return True
    
    def _setup_videos(self):
        """Setup video display using OpenCV - supports up to 2 cameras side-by-side."""
        # Close existing video captures
        for cap in self.video_captures:
            cap.release()
        self.video_captures.clear()
        self.video_labels.clear()
        
        # Clear layout
        while self.video_layout.count():
            item = self.video_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Check if we have videos
        num_videos = len(self.trial_data.video_files)
        
        if num_videos == 0:
            # Show placeholder
            self.no_video_label = QtWidgets.QLabel("No video available")
            self.no_video_label.setAlignment(QtCore.Qt.AlignCenter)
            self.no_video_label.setStyleSheet(
                "background-color: #2c3e50; color: #95a5a6; "
                "font-size: 14pt; padding: 40px; border-radius: 5px;"
            )
            self.video_layout.addWidget(self.no_video_label)
            print(f"[TrialViewer] No videos found")
            return
        
        # Use up to 2 cameras
        cameras_to_show = min(num_videos, 2)
        
        print(f"[TrialViewer] Setting up {cameras_to_show} camera(s)")
        
        for i in range(cameras_to_show):
            video_path = self.trial_data.video_files[i]
            
            print(f"[TrialViewer] Camera {i}: {os.path.basename(video_path)}")
            
            # Open video with OpenCV
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"[TrialViewer] ERROR: Could not open video file {i}")
                continue
            
            # Create QLabel to display video frames
            video_label = QtWidgets.QLabel()
            video_label.setMinimumSize(320, 240)
            video_label.setScaledContents(False)  # Don't stretch - maintain aspect ratio
            video_label.setStyleSheet("background-color: black; border: 1px solid #555;")
            video_label.setAlignment(QtCore.Qt.AlignCenter)
            
            # Read and display first frame
            ret, frame = cap.read()
            if ret:
                self._display_video_frame(video_label, frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
            
            self.video_captures.append(cap)
            self.video_labels.append(video_label)
            self.video_layout.addWidget(video_label)
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[TrialViewer] Camera {i}: {width}x{height}, {total_frames} frames @ {fps:.1f} fps")
            print(f"[TrialViewer] Camera {i} has timestamps: {i in self.trial_data.camera_timestamps}")
        
        self.current_video_frame = 0
        print(f"[TrialViewer] Video setup complete - {len(self.video_captures)} camera(s) ready")
    
    def _display_video_frame(self, label, frame):
        """Display an OpenCV frame in a QLabel with correct aspect ratio."""
        import cv2
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtCore import Qt
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Display in label
        label.setPixmap(scaled_pixmap)

    def _toggle_play_pause(self):
        """Toggle playback - skeleton and video advance frame-by-frame together."""
        if self.is_playing:
            # Pause
            self.is_playing = False
            self.playback_timer.stop()
            self.play_pause_btn.setText("▶ Play")
        else:
            # Play
            self.is_playing = True
            
            # Sync video to current position before starting
            self._sync_video_to_frame()
            
            # Start skeleton playback timer
            interval_ms = int(1000.0 / (self.trial_data.fps * self.playback_speed))
            self.playback_timer.start(interval_ms)
            self.play_pause_btn.setText("⏸ Pause")
            
            print(f"[TrialViewer] Playback started - skeleton @ {self.trial_data.fps * self.playback_speed:.1f} fps")

    def _sync_video_to_frame(self):
        """Sync all videos to current skeleton frame using camera timestamps."""
        if len(self.video_captures) == 0:
            return
        
        import cv2
        
        # Get current CSV timestamp
        csv_time = self.trial_data.timestamps[self.current_frame] if self.current_frame < len(self.trial_data.timestamps) else 0
        
        # Sync each camera
        for i, (cap, label) in enumerate(zip(self.video_captures, self.video_labels)):
            # Check if we have camera timestamps for this camera
            if i in self.trial_data.camera_timestamps:
                camera_times = self.trial_data.camera_timestamps[i]
                
                # Find closest camera frame
                closest_idx = min(range(len(camera_times)), key=lambda j: abs(camera_times[j] - csv_time))
                
                # Seek to that frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, closest_idx)
                ret, frame = cap.read()
                
                if ret:
                    self._display_video_frame(label, frame)
                    if i == 0:  # Only store for primary camera
                        self.current_video_frame = closest_idx
                
                if self.current_frame == 0 and i == 0:  # Debug first frame only for first camera
                    print(f"[TrialViewer] Camera {i} synced: CSV={csv_time:.3f}s, Cam={camera_times[closest_idx]:.3f}s, Frame={closest_idx}")
            else:
                # Fallback: use relative time and video FPS
                first_time = self.trial_data.timestamps[0] if len(self.trial_data.timestamps) > 0 else 0
                relative_time = csv_time - first_time
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_number = int(relative_time * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    self._display_video_frame(label, frame)
                    if i == 0:
                        self.current_video_frame = frame_number
                
                if self.current_frame == 0 and i == 0:
                    print(f"[TrialViewer] Camera {i} synced (no timestamps): {relative_time:.3f}s, frame {frame_number}")
    
    def _advance_frame(self):
        """Advance to next frame during playback."""
        if self.current_frame < self.trial_data.num_frames - 1:
            self.current_frame += 1
            self._update_display()
        else:
            # Reached end, stop playback
            self._toggle_play_pause()
    
    def _step_frame(self, direction: int):
        """Step one frame forward or backward."""
        new_frame = self.current_frame + direction
        new_frame = max(0, min(new_frame, self.trial_data.num_frames - 1))
        
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self._update_display()
    
    def _on_timeline_changed(self, value: int):
        """Handle timeline slider change."""
        if value != self.current_frame:
            self.current_frame = value
            self._update_display()
    
    def _update_display(self):
        """Update skeleton, angles, and videos for current frame."""
        if self.trial_data.num_frames == 0:
            return
        
        # Get frame data
        orientations, angles = self.trial_data.get_frame(self.current_frame)
        
        # Compute segment positions
        positions = self.kinematics.compute_segment_positions(orientations)
        
        # Update skeleton
        self.skeleton_canvas.update_skeleton(positions)
        
        # Update angle table - convert flat euler format to grouped format
        grouped_angles = self._group_euler_angles(angles)
        
        if grouped_angles and self.current_frame == 0:  # Debug first frame only
            print(f"[TrialViewer] Updating angle table with {len(grouped_angles)} joints")
            print(f"[TrialViewer] Sample joints: {list(grouped_angles.keys())[:3]}")
        
        self.angle_table.update_angles(grouped_angles)
        
        # Update timeline slider (without triggering signal)
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(self.current_frame)
        self.timeline_slider.blockSignals(False)
        
        # Update frame/time labels
        timestamp = self.trial_data.timestamps[self.current_frame] if self.current_frame < len(self.trial_data.timestamps) else 0
        self.frame_info_label.setText(f"Frame: {self.current_frame + 1} / {self.trial_data.num_frames}")
        self.time_label.setText(f"Time: {timestamp:.2f}s / {self.trial_data.duration:.2f}s")
        
        # Update video to match current frame (works frame-by-frame with OpenCV)
        self._sync_video_to_frame()
    
    def _group_euler_angles(self, flat_angles: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Convert flat euler angle dictionary to grouped format.
        
        Input:  {'pelvis_trunk_euler_0': 10.5, 'pelvis_trunk_euler_1': 5.2, ...}
        Output: {'pelvis_trunk': array([10.5, 5.2, -3.1]), ...}
        """
        import numpy as np
        
        grouped = {}
        
        # Group by joint name
        for key, value in flat_angles.items():
            # Extract joint name and euler component
            # Format: "pelvis_trunk_euler_0" → joint="pelvis_trunk", component=0
            if '_euler_' in key:
                parts = key.rsplit('_euler_', 1)
                if len(parts) == 2:
                    joint_name = parts[0]
                    try:
                        component = int(parts[1])
                    except ValueError:
                        continue
                    
                    # Initialize array if first time seeing this joint
                    if joint_name not in grouped:
                        grouped[joint_name] = np.zeros(3)
                    
                    # Set the component
                    if 0 <= component < 3:
                        grouped[joint_name][component] = value
        
        return grouped
    
    def _update_opensim_button_style(self, enabled: bool):
        """Update OpenSim button appearance based on enabled state."""
        if enabled:
            # Green when enabled
            self.export_opensim_btn.setEnabled(True)
            self.export_opensim_btn.setStyleSheet(
                "padding: 10px; font-size: 10pt; font-weight: bold; "
                "background-color: #27ae60; color: white; border-radius: 5px;"
            )
        else:
            # Gray when disabled
            self.export_opensim_btn.setEnabled(False)
            self.export_opensim_btn.setStyleSheet(
                "padding: 10px; font-size: 10pt; font-weight: bold; "
                "background-color: #95a5a6; color: #bdc3c7; border-radius: 5px;"
            )
    
    def _export_to_opensim(self):
        """Export trial data to OpenSim format."""
        from PyQt5.QtWidgets import QMessageBox, QFileDialog
        from opensim_exporter import export_trial_to_opensim
        
        if not self.trial_data or not self.trial_data.trial_folder:
            QMessageBox.warning(self, "No Trial Loaded", "Please load a trial first.")
            return
        
        # Ask user for output folder
        default_output = os.path.join(self.trial_data.trial_folder, 'opensim_export')
        output_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder for OpenSim Files",
            default_output,
            QFileDialog.ShowDirsOnly
        )
        
        if not output_folder:
            return  # User cancelled
        
        try:
            # Export
            print(f"\n[TrialViewer] Starting OpenSim export...")
            created_files = export_trial_to_opensim(self.trial_data.trial_folder, output_folder)
            
            # Show success message
            total_files = sum(len(v) for v in created_files.values())
            msg = f"Successfully exported {total_files} OpenSim files!\n\n"
            msg += f"Placer (calibration) files: {len(created_files['placer_sensor']) + len(created_files['placer_world'])}\n"
            msg += f"Motion files: {len(created_files['motions_sensor']) + len(created_files['motions_world'])}\n\n"
            msg += f"Output folder:\n{output_folder}"
            
            QMessageBox.information(self, "Export Complete", msg)
            
            # Open output folder
            import platform
            import subprocess
            if platform.system() == 'Windows':
                os.startfile(output_folder)
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', output_folder])
            else:
                subprocess.Popen(['xdg-open', output_folder])
                
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting to OpenSim:\n{str(e)}")
            print(f"[TrialViewer] Export error: {e}")
            import traceback
            traceback.print_exc()


