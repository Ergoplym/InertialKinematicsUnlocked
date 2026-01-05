"""
Recording management controller for IKU application.
Handles session naming, file organization, and data saving.
"""

import os
import time
import csv
import json
from typing import Optional, Dict, List
from PyQt5 import QtCore, QtWidgets
from body_config import BODY_CONFIGS, DEFAULT_BODY_CONFIG, get_body_config
from anthropometry import AnthropometryProfile, AnthropometryManager


class RecordingController(QtCore.QObject):
    """Manages recording sessions with subject/session IDs and custom directories."""
    
    # Signals
    recording_started = QtCore.pyqtSignal(str)  # Session folder path
    recording_stopped = QtCore.pyqtSignal(str)  # Session folder path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Session information
        self.subject_id = ""
        self.session_id = ""
        self.base_directory = os.path.join(os.getcwd(), "recordings")
        self.body_config = DEFAULT_BODY_CONFIG  # 'whole_body', 'upper_body', 'lower_body'
        
        # Current session
        self.current_session_folder = None
        self.session_timestamp = None
        
        # Recording state
        self.is_recording = False
        self.record_data = []
    
    def set_session_info(self, subject_id: str, session_id: str, base_directory: str = None,
                        body_config: str = None):
        """
        Set session information for file naming.
        
        Args:
            subject_id: Subject identifier (e.g., "P001", "John_Doe")
            session_id: Session identifier (e.g., "Baseline", "PostTraining")
            base_directory: Base folder for all recordings (optional)
            body_config: Body configuration ('whole_body', 'upper_body', 'lower_body')
        """
        self.subject_id = subject_id.strip()
        self.session_id = session_id.strip()
        
        if base_directory:
            self.base_directory = base_directory
        
        if body_config:
            self.body_config = body_config
        
        print(f"[RecordingController] Subject: {self.subject_id}, Session: {self.session_id}")
        print(f"[RecordingController] Body config: {self.body_config}")
        print(f"[RecordingController] Save directory: {self.base_directory}")
    
    def prompt_session_info(self, parent=None, current_anthro_profile=None) -> tuple[bool, Optional['AnthropometryProfile']]:
        """
        Show dialog to get subject ID, session ID, body config, anthropometry, and save directory.
        
        Args:
            parent: Parent widget for dialog
            current_anthro_profile: Current anthropometry profile
            
        Returns:
            (success, anthropometry_profile) tuple
        """
        dialog = SessionInfoDialog(
            current_subject=self.subject_id,
            current_session=self.session_id,
            current_directory=self.base_directory,
            current_body_config=self.body_config,
            current_anthro_profile=current_anthro_profile,
            parent=parent
        )
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.subject_id = dialog.get_subject_id()
            self.session_id = dialog.get_session_id()
            self.base_directory = dialog.get_directory()
            self.body_config = dialog.get_body_config()
            anthro_profile = dialog.get_anthropometry_profile()
            return True, anthro_profile
        
        return False, None
    
    def create_session_folder(self) -> str:
        """
        Create timestamped session folder with subject/session IDs.
        If folder already exists (from session configuration), reuse it.
        
        Returns:
            Path to created/existing session folder
        """
        # If folder already exists, reuse it
        if self.current_session_folder and os.path.exists(self.current_session_folder):
            print(f"[RecordingController] Reusing existing session folder: {self.current_session_folder}")
            return self.current_session_folder
        
        # Generate timestamp
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create folder name: SubjectID_SessionID_Timestamp
        folder_name = f"{self.subject_id}_{self.session_id}_{self.session_timestamp}"
        
        # Create full path
        self.current_session_folder = os.path.join(self.base_directory, folder_name)
        
        # Create directory
        os.makedirs(self.current_session_folder, exist_ok=True)
        
        print(f"[RecordingController] Created session folder: {self.current_session_folder}")
        
        return self.current_session_folder
    
    def get_filename(self, file_type: str, extension: str = "") -> str:
        """
        Generate standardized filename.
        
        Args:
            file_type: Type of file (e.g., "recording", "calibrationNpose", "camera_0")
            extension: File extension (e.g., ".csv", ".mp4")
            
        Returns:
            Filename in format: SubjectID_SessionID_Timestamp_FileType.ext
        """
        if not extension.startswith('.') and extension:
            extension = f".{extension}"
        
        filename = f"{self.subject_id}_{self.session_id}_{self.session_timestamp}_{file_type}{extension}"
        return filename
    
    def get_filepath(self, file_type: str, extension: str = "") -> str:
        """
        Generate full filepath in current session folder.
        Auto-increments filename if file already exists (_1, _2, _3, etc.)
        
        Args:
            file_type: Type of file (e.g., "recording", "calibrationNpose")
            extension: File extension (e.g., ".csv", ".mp4")
            
        Returns:
            Full path to file (guaranteed not to exist)
        """
        if not self.current_session_folder:
            raise RuntimeError("No active session folder. Call create_session_folder() first.")
        
        # Get base filename
        filename = self.get_filename(file_type, extension)
        filepath = os.path.join(self.current_session_folder, filename)
        
        # If file doesn't exist, return as-is
        if not os.path.exists(filepath):
            return filepath
        
        # File exists - add incrementing number
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        
        counter = 1
        while True:
            new_filename = f"{base_name}_{counter}{ext}"
            new_filepath = os.path.join(self.current_session_folder, new_filename)
            
            if not os.path.exists(new_filepath):
                print(f"[RecordingController] File exists, using: {new_filename}")
                return new_filepath
            
            counter += 1
            
            # Safety limit
            if counter > 100:
                raise RuntimeError("Too many files with same name (>100)")
        
        return filepath
    
    def start_recording(self):
        """Start a new recording session."""
        if self.is_recording:
            print("[RecordingController] Already recording")
            return
        
        # Validate session info
        if not self.subject_id or not self.session_id:
            raise ValueError("Subject ID and Session ID must be set before recording")
        
        # Create session folder
        session_folder = self.create_session_folder()
        
        # Initialize recording data
        self.record_data = []
        self.is_recording = True
        
        print(f"[RecordingController] Recording started")
        self.recording_started.emit(session_folder)
    
    def add_data_sample(self, data_dict: Dict):
        """
        Add a data sample to the recording.
        
        Args:
            data_dict: Dictionary of data values (e.g., joint angles, timestamps)
        """
        if self.is_recording:
            self.record_data.append(data_dict)
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save data.
        
        Returns:
            Path to session folder, or None if not recording
        """
        if not self.is_recording:
            print("[RecordingController] Not currently recording")
            return None
        
        self.is_recording = False
        
        # Save IMU data
        filepath = self._save_imu_data()
        
        # Save session metadata
        self._save_session_metadata()
        
        print(f"[RecordingController] Recording stopped")
        print(f"[RecordingController] Saved {len(self.record_data)} samples")
        
        session_folder = self.current_session_folder
        self.recording_stopped.emit(session_folder)
        
        return session_folder
    
    def _save_session_metadata(self):
        """Save session and calibration info to JSON files."""
        if not self.current_session_folder:
            return
        
        # Save session info
        session_info = {
            'subject_id': self.subject_id,
            'session_id': self.session_id,
            'body_config': self.body_config,
            'timestamp': self.session_timestamp,
            'base_directory': self.base_directory
        }
        
        session_info_path = os.path.join(self.current_session_folder, 'session_info.json')
        with open(session_info_path, 'w') as f:
            json.dump(session_info, f, indent=2)
        
        print(f"[RecordingController] Session info saved to: {session_info_path}")
        
        # Save calibration info (placeholder for now - will be populated by main app)
        calibration_info_path = os.path.join(self.current_session_folder, 'calibration_info.json')
        if not os.path.exists(calibration_info_path):
            calib_info = {
                'type': 'pending',
                'note': 'Calibration info should be saved by main application'
            }
            with open(calibration_info_path, 'w') as f:
                json.dump(calib_info, f, indent=2)
    
    def save_calibration_info(self, calibration_data: Dict):
        """
        Save calibration information to the session folder.
        
        Args:
            calibration_data: Dictionary with calibration parameters
        """
        if not self.current_session_folder:
            print("[RecordingController] No active session folder")
            return
        
        calibration_info_path = os.path.join(self.current_session_folder, 'calibration_info.json')
        with open(calibration_info_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"[RecordingController] Calibration info saved to: {calibration_info_path}")
    
    def _save_imu_data(self) -> str:
        """
        Save IMU recording data to CSV.
        
        Returns:
            Path to saved CSV file
        """
        if not self.record_data:
            print("[RecordingController] No data to save")
            return None
        
        # Generate filepath
        filepath = self.get_filepath("recording", ".csv")
        
        # Build union of all keys for CSV columns
        all_keys = set()
        for r in self.record_data:
            all_keys.update(r.keys())
        
        # Organize column order: time, event, quaternions, euler angles, other
        quat_keys = sorted([k for k in all_keys if '_q' in k and k not in ("time", "event")])
        euler_keys = sorted([k for k in all_keys if '_euler_' in k])
        other_keys = sorted([k for k in all_keys if k not in quat_keys + euler_keys + ["time", "event"]])
        
        fieldnames = ["time", "event"] + quat_keys + euler_keys + other_keys
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.record_data)
        
        print(f"[RecordingController] IMU data saved to: {filepath}")
        return filepath
    
    def save_calibration_data(self, pose_type: str, samples: List[Dict]) -> str:
        """
        Save calibration pose samples.
        
        Args:
            pose_type: "Npose" or "Tpose"
            samples: List of sample dictionaries
            
        Returns:
            Path to saved file
        """
        if not self.current_session_folder:
            self.create_session_folder()
        
        filepath = self.get_filepath(f"calibration{pose_type}", ".csv")
        
        if not samples:
            print(f"[RecordingController] No {pose_type} samples to save")
            return None
        
        # Get all keys from samples
        all_keys = set()
        for sample in samples:
            all_keys.update(sample.keys())
        
        fieldnames = sorted(all_keys)
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)
        
        print(f"[RecordingController] {pose_type} calibration saved to: {filepath}")
        return filepath
    
    def get_session_folder(self) -> Optional[str]:
        """Get current session folder path."""
        return self.current_session_folder
    
    def get_session_info(self) -> Dict[str, str]:
        """
        Get current session information.
        
        Returns:
            Dictionary with subject_id, session_id, body_config, timestamp, folder
        """
        return {
            'subject_id': self.subject_id,
            'session_id': self.session_id,
            'body_config': self.body_config,
            'timestamp': self.session_timestamp or '',
            'folder': self.current_session_folder or ''
        }


class SessionInfoDialog(QtWidgets.QDialog):
    """Dialog to collect subject ID, session ID, body config, anthropometry, and save directory."""
    
    def __init__(self, current_subject="", current_session="", 
                 current_directory="", current_body_config="whole_body",
                 current_anthro_profile=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recording Session Information")
        self.setModal(True)
        self.resize(650, 750)  # Increased height for anthropometry
        
        # Store values
        self.subject_id = current_subject
        self.session_id = current_session
        self.directory = current_directory or os.path.join(os.getcwd(), "recordings")
        self.body_config = current_body_config
        
        # Anthropometry
        self.anthro_manager = AnthropometryManager()
        self.anthro_profile = current_anthro_profile or AnthropometryProfile()
        
        self._init_ui()
    
    def _init_ui(self):
        """Build dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QtWidgets.QLabel("Configure Recording Session")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        # Subject ID
        subject_group = QtWidgets.QGroupBox("Subject Identifier")
        subject_layout = QtWidgets.QVBoxLayout()
        
        subject_label = QtWidgets.QLabel("Enter Subject ID (e.g., P001, SubjectA, John_Doe):")
        subject_label.setStyleSheet("font-size: 10pt; color: #555;")
        subject_layout.addWidget(subject_label)
        
        self.subject_input = QtWidgets.QLineEdit(self.subject_id)
        self.subject_input.setPlaceholderText("Required: e.g., P001")
        self.subject_input.setStyleSheet("font-size: 11pt; padding: 5px;")
        subject_layout.addWidget(self.subject_input)
        
        subject_group.setLayout(subject_layout)
        layout.addWidget(subject_group)
        
        # Session ID
        session_group = QtWidgets.QGroupBox("Session Identifier")
        session_layout = QtWidgets.QVBoxLayout()
        
        session_label = QtWidgets.QLabel("Enter Session ID (e.g., Baseline, Trial1, PostTraining):")
        session_label.setStyleSheet("font-size: 10pt; color: #555;")
        session_layout.addWidget(session_label)
        
        self.session_input = QtWidgets.QLineEdit(self.session_id)
        self.session_input.setPlaceholderText("Required: e.g., Baseline")
        self.session_input.setStyleSheet("font-size: 11pt; padding: 5px;")
        session_layout.addWidget(self.session_input)
        
        session_group.setLayout(session_layout)
        layout.addWidget(session_group)
        
        # Body Configuration
        body_config_group = QtWidgets.QGroupBox("Body Configuration")
        body_config_layout = QtWidgets.QVBoxLayout()
        
        body_config_label = QtWidgets.QLabel("Select sensor setup:")
        body_config_label.setStyleSheet("font-size: 10pt; color: #555;")
        body_config_layout.addWidget(body_config_label)
        
        # Radio buttons for body config
        self.body_config_buttons = {}
        for config_key, config in BODY_CONFIGS.items():
            radio = QtWidgets.QRadioButton(f"{config.display_name} ({config.num_sensors} sensors)")
            radio.setStyleSheet("font-size: 10pt; padding: 3px;")
            
            # Add description as tooltip
            radio.setToolTip(config.description)
            
            # Check current selection
            if config_key == self.body_config:
                radio.setChecked(True)
            
            self.body_config_buttons[config_key] = radio
            body_config_layout.addWidget(radio)
        
        # Add detailed info label
        self.body_config_info = QtWidgets.QLabel()
        self.body_config_info.setStyleSheet(
            "font-size: 9pt; color: #666; padding: 5px; "
            "background-color: #f8f9fa; border-radius: 3px; margin-top: 5px;"
        )
        self.body_config_info.setWordWrap(True)
        body_config_layout.addWidget(self.body_config_info)
        
        # Update info when selection changes
        for config_key, radio in self.body_config_buttons.items():
            radio.toggled.connect(lambda checked, key=config_key: 
                                 self._update_body_config_info(key) if checked else None)
        
        # Don't set initial info yet - wait until preview_label exists
        
        body_config_group.setLayout(body_config_layout)
        layout.addWidget(body_config_group)
        
        # Anthropometry
        anthro_group = QtWidgets.QGroupBox("Subject Anthropometry")
        anthro_layout = QtWidgets.QVBoxLayout()
        
        # Top row: buttons
        button_row = QtWidgets.QHBoxLayout()
        
        anthro_load_subject_btn = QtWidgets.QPushButton("📂 Load Subject")
        anthro_load_subject_btn.setStyleSheet("font-size: 9pt; padding: 4px 10px;")
        anthro_load_subject_btn.setToolTip("Load saved anthropometry profile")
        anthro_load_subject_btn.clicked.connect(self._load_subject_profile)
        button_row.addWidget(anthro_load_subject_btn)
        
        anthro_defaults_btn = QtWidgets.QPushButton("Load Defaults")
        anthro_defaults_btn.setStyleSheet("font-size: 9pt; padding: 4px 10px;")
        anthro_defaults_btn.setToolTip("Quick select from standard profiles")
        anthro_defaults_btn.clicked.connect(self._load_anthro_defaults)
        button_row.addWidget(anthro_defaults_btn)
        
        anthro_advanced_btn = QtWidgets.QPushButton("📏 Advanced...")
        anthro_advanced_btn.setStyleSheet("font-size: 9pt; padding: 4px 10px; background-color: #3498db; color: white;")
        anthro_advanced_btn.setToolTip("Open full anthropometry configuration dialog")
        anthro_advanced_btn.clicked.connect(self._open_advanced_anthropometry)
        button_row.addWidget(anthro_advanced_btn)
        
        button_row.addStretch()
        anthro_layout.addLayout(button_row)
        
        # Simple inputs (always visible)
        simple_layout = QtWidgets.QHBoxLayout()
        
        height_label = QtWidgets.QLabel("Height:")
        simple_layout.addWidget(height_label)
        
        self.height_input = QtWidgets.QDoubleSpinBox()
        self.height_input.setRange(1.0, 2.5)
        self.height_input.setSingleStep(0.01)
        self.height_input.setDecimals(2)
        self.height_input.setValue(self.anthro_profile.height_m)
        self.height_input.setSuffix(" m")
        self.height_input.setFixedWidth(100)
        simple_layout.addWidget(self.height_input)
        
        weight_label = QtWidgets.QLabel("Weight:")
        simple_layout.addWidget(weight_label)
        
        self.weight_input = QtWidgets.QDoubleSpinBox()
        self.weight_input.setRange(30.0, 200.0)
        self.weight_input.setSingleStep(0.5)
        self.weight_input.setDecimals(1)
        self.weight_input.setValue(self.anthro_profile.weight_kg)
        self.weight_input.setSuffix(" kg")
        self.weight_input.setFixedWidth(100)
        simple_layout.addWidget(self.weight_input)
        
        # Mode indicator
        self.anthro_mode_label = QtWidgets.QLabel(f"[{self.anthro_profile.mode.upper()}]")
        self.anthro_mode_label.setStyleSheet(
            "font-size: 8pt; color: #666; font-family: 'Courier New';"
        )
        simple_layout.addWidget(self.anthro_mode_label)
        
        simple_layout.addStretch()
        anthro_layout.addLayout(simple_layout)
        
        anthro_group.setLayout(anthro_layout)
        layout.addWidget(anthro_group)
        
        # Save Directory
        dir_group = QtWidgets.QGroupBox("Save Location")
        dir_layout = QtWidgets.QVBoxLayout()
        
        dir_label = QtWidgets.QLabel("Recordings will be saved to:")
        dir_label.setStyleSheet("font-size: 10pt; color: #555;")
        dir_layout.addWidget(dir_label)
        
        dir_row = QtWidgets.QHBoxLayout()
        
        self.directory_input = QtWidgets.QLineEdit(self.directory)
        self.directory_input.setStyleSheet("font-size: 10pt; padding: 5px;")
        self.directory_input.setReadOnly(True)
        dir_row.addWidget(self.directory_input)
        
        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.setStyleSheet("padding: 5px 15px;")
        browse_btn.clicked.connect(self._browse_directory)
        dir_row.addWidget(browse_btn)
        
        dir_layout.addLayout(dir_row)
        
        # Preview
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setStyleSheet(
            "font-size: 9pt; color: #666; padding: 8px; "
            "background-color: #f0f0f0; border-radius: 3px; font-family: 'Courier New';"
        )
        self.preview_label.setWordWrap(True)
        dir_layout.addWidget(self.preview_label)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # Update preview on text change
        self.subject_input.textChanged.connect(self._update_preview)
        self.session_input.textChanged.connect(self._update_preview)
        
        # Set initial body config info and preview (now that all widgets exist)
        self._update_body_config_info(self.body_config)
        self._update_preview()
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setFixedSize(100, 35)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.setFixedSize(100, 35)
        ok_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        ok_btn.clicked.connect(self._on_ok)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
    
    def _update_body_config_info(self, config_key: str):
        """Update body configuration info label."""
        config = BODY_CONFIGS[config_key]
        
        # Get segment list
        segments = config.required_segments
        
        # Format info text
        info_text = f"<b>{config.description}</b><br>"
        info_text += f"Required sensors: {', '.join(segments[:5])}"
        if len(segments) > 5:
            info_text += f", + {len(segments)-5} more"
        
        self.body_config_info.setText(info_text)
        self._update_preview()
    
    def _load_subject_profile(self):
        """Load saved subject anthropometry profile."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Subject Profile",
            self.anthro_manager.profiles_dir,
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                self.anthro_profile = AnthropometryProfile.load(filename)
                
                # Update UI with loaded values
                if self.anthro_profile.mode == "simple":
                    self.height_input.setValue(self.anthro_profile.height_m)
                    self.weight_input.setValue(self.anthro_profile.weight_kg)
                
                self.anthro_mode_label.setText(f"[{self.anthro_profile.mode.upper()}]")
                
                QtWidgets.QMessageBox.information(
                    self,
                    "Profile Loaded",
                    f"Loaded: {self.anthro_profile.name}\n"
                    f"Mode: {self.anthro_profile.mode}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Failed to load profile:\n{str(e)}"
                )
    
    def _load_anthro_defaults(self):
        """Load default anthropometry profiles."""
        defaults = self.anthro_manager.get_default_profiles()
        items = [profile.name for profile in defaults.values()]
        
        item, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Load Default Profile",
            "Select a default profile:",
            items,
            0,
            False
        )
        
        if ok and item:
            # Find the profile by name
            for profile in defaults.values():
                if profile.name == item:
                    self.anthro_profile = profile
                    
                    # Update UI
                    self.height_input.setValue(self.anthro_profile.height_m)
                    self.weight_input.setValue(self.anthro_profile.weight_kg)
                    self.anthro_mode_label.setText(f"[{self.anthro_profile.mode.upper()}]")
                    break
    
    def _open_advanced_anthropometry(self):
        """Open full anthropometry configuration dialog."""
        from anthropometry_dialog import AnthropometryDialog
        
        # Update profile with current simple values first
        self.anthro_profile.height_m = self.height_input.value()
        self.anthro_profile.weight_kg = self.weight_input.value()
        
        dialog = AnthropometryDialog(
            current_profile=self.anthro_profile,
            parent=self
        )
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.anthro_profile = dialog.get_profile()
            
            # Update UI with new values
            if self.anthro_profile.mode == "simple":
                self.height_input.setValue(self.anthro_profile.height_m)
                self.weight_input.setValue(self.anthro_profile.weight_kg)
            
            self.anthro_mode_label.setText(f"[{self.anthro_profile.mode.upper()}]")
            
            QtWidgets.QMessageBox.information(
                self,
                "Anthropometry Updated",
                f"Profile: {self.anthro_profile.name}\n"
                f"Mode: {self.anthro_profile.mode}"
            )
    
    def _update_preview(self):
        """Update filename preview."""
        subject = self.subject_input.text().strip() or "SubjectID"
        session = self.session_input.text().strip() or "SessionID"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        preview_folder = f"{subject}_{session}_{timestamp}"
        preview_file = f"{subject}_{session}_{timestamp}_recording.csv"
        
        self.preview_label.setText(
            f"Folder: {preview_folder}/\n"
            f"Example file: {preview_file}"
        )
    
    def _browse_directory(self):
        """Open directory picker."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Recordings Folder",
            self.directory_input.text()
        )
        
        if directory:
            self.directory = directory
            self.directory_input.setText(directory)
    
    def _on_ok(self):
        """Validate and accept."""
        subject = self.subject_input.text().strip()
        session = self.session_input.text().strip()
        
        if not subject:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Subject ID",
                "Please enter a Subject ID."
            )
            return
        
        if not session:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Session ID",
                "Please enter a Session ID."
            )
            return
        
        self.subject_id = subject
        self.session_id = session
        self.accept()
    
    def get_subject_id(self) -> str:
        """Get entered subject ID."""
        return self.subject_id
    
    def get_session_id(self) -> str:
        """Get entered session ID."""
        return self.session_id
    
    def get_directory(self) -> str:
        """Get selected directory."""
        return self.directory
    
    def get_body_config(self) -> str:
        """Get selected body configuration."""
        for config_key, radio in self.body_config_buttons.items():
            if radio.isChecked():
                return config_key
        return DEFAULT_BODY_CONFIG
    
    def get_anthropometry_profile(self) -> AnthropometryProfile:
        """Get configured anthropometry profile."""
        # Update profile with current UI values
        if self.anthro_profile.mode == "simple":
            self.anthro_profile.height_m = self.height_input.value()
            self.anthro_profile.weight_kg = self.weight_input.value()
        
        # Compute segment lengths
        self.anthro_profile.compute_segment_lengths()
        
        return self.anthro_profile
