"""Main application entry point - Modified for quaternion support"""

import sys
import time
import os
import csv
import warnings
from collections import deque
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from PyQt5 import QtWidgets, QtCore, QtGui

from config import *
from calibration import CalibrationManager
from dual_pose_calibration import DualPoseCalibration
from pelvis_relative_functional_calibration import PelvisRelativeFunctionalCalibration
from kinematics import KinematicsEngine
from mounting import apply_mounting
from visualization_optimized import Skeleton3DCanvas, CompactAngleTableWidget
from imu_stream import IMUStreamFactory
from world_coordinate_conversion import WorldCoordinateConverter
from welcome_screen import WelcomeDialog
from webcam_recorder import WebcamRecorder
from camera_widgets import CameraGridWidget, CameraPreviewWidget
from recording_controller import RecordingController
from body_config import get_body_config, BODY_CONFIGS
from sensor_mapping_dialog import SensorMappingDialog
from anthropometry import AnthropometryProfile, AnthropometryManager
from anthropometry_dialog import AnthropometryDialog
from trial_viewer import TrialViewerWindow


# Suppress gimbal lock warnings from scipy
warnings.filterwarnings('ignore', message='Gimbal lock detected')


class BiomechApp(QtWidgets.QMainWindow):
    """Main application window"""
    
    def __init__(self, radio_channel=11):
        super().__init__()
        self.setWindowTitle("IKU - Inertial Kinematics Unlocked")
        
        # Store configuration
        self.radio_channel = radio_channel
        print(f"[App] Initializing with radio channel: {radio_channel}")
        
        # Trial viewer window (created on demand)
        self.trial_viewer_window = None
        
        # Auto-scale window to screen size
        self._setup_window_size()
        
        # Initialize components
        self.calibration = CalibrationManager(BODY_MODEL)
        self.dual_pose_calib = DualPoseCalibration()
        self.func_calib_v2 = PelvisRelativeFunctionalCalibration()
        self.kinematics = KinematicsEngine(BODY_MODEL)
        self.world_converter = WorldCoordinateConverter("xsens_enu_to_app")
        self._latest_world_rots = None
        self._calibration_updating = False
        self._last_calib_refresh_time = None
        
        # Webcam recorder (uses same timestamp source as IMU)
        class TimestampManager:
            def __init__(self, app):
                self.app = app
            def get_timestamp(self):
                return time.monotonic() - self.app.start_time
        
        self.webcam_recorder = WebcamRecorder(TimestampManager(self))
        
        # Recording controller for session management
        self.recording_ctrl = RecordingController()
        # Don't set default session info - require user to configure
        # This ensures buttons stay disabled until session is properly set up
        
        # Anthropometry manager for skeleton scaling
        self.anthro_manager = AnthropometryManager()
        print(f"[App] Default anthropometry: {self.anthro_manager.current_profile.name}")

        # Cache for segment positions (avoid recomputation)
        self._cached_segment_positions = None
        self._cached_joint_data = None
        self._cached_segment_rots = None
        
        # Recording state
        self.is_recording = False
        self.record_data = []
        
        # Calibration movement state
        self.movement_recording = None
        self.movement_start_time = None
        self.current_movement_duration = 0.0
        self.in_pause_phase = False
        self.pause_start_time = None
        
        # Static calibration delays
        self.static_calib_pending = False
        self.static_calib_start_time = None
        self.post_static_calib_pending = False
        self.post_static_calib_start_time = None
        
        # Build UI
        self._init_ui()
        
        # Connect to IMU stream with selected radio channel
        self.imu_stream = IMUStreamFactory.create_stream(
            use_real_imu=True, 
            radio_channel=self.radio_channel
        )
        
        # Detect if using fake stream
        self.using_fake_stream = self.imu_stream.__class__.__name__ == 'FakeIMUStream'
        if self.using_fake_stream:
            print("[App] Using fake IMU stream (no hardware detected)")
        
        # Start timers
        self.imu_timer = QtCore.QTimer()
        self.imu_timer.timeout.connect(self._on_imu_update)
        self.imu_timer.start(int(1000 / UPDATE_RATE_HZ))
        
        self.display_timer = QtCore.QTimer()
        self.display_timer.timeout.connect(self._on_display_update)
        self.display_timer.start(int(1000 / DISPLAY_RATE_HZ))
        
        # Camera preview timer (separate, slower rate to reduce CPU load)
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.timeout.connect(self._on_camera_update)
        self.camera_timer.start(100)  # 10 FPS preview updates
        
        self.start_time = time.monotonic()
        
        # Detect available cameras on startup
        self._detect_cameras()
        
        # Set initial status message
        self.status_label.setText("⚠️ Configure session to begin")
        self.instruction_label.setText("Click '⚙️ Configure Session Info' to set Subject/Session IDs")
        
        # Update sensor count display if using fake stream
        if self.using_fake_stream:
            self.sensor_count_label.setText("0 / ? - Using Fake Stream")
            self.sensor_count_label.setStyleSheet(
                "font-size: 11pt; font-weight: bold; font-family: 'Courier New'; color: #9b59b6;"
            )

    def _refresh_static_calibration_now(self):
        """Instant static calibration (no countdown). Works during recording."""
        self._calibration_updating = True
        try:
            raw_sample = self._get_imu_sample()
            if not raw_sample:
                self.status_label.setText("Error: No IMU data available")
                return

            sensor_rots = {
                name: R.from_quat([q[1], q[2], q[3], q[0]])
                for name, q in raw_sample.items()
            }

            world_rots = self.world_converter.convert_orientations(sensor_rots)
            world_rots = apply_mounting(world_rots)

            # Recompute offsets immediately with selected pose
            pose_type = getattr(self, 'selected_pose', 'n_pose')
            self.calibration.static_calibration(world_rots, is_post_calibration=False, pose_type=pose_type)

            # Marker for analysis (and optional UI)
            t = time.monotonic() - self.start_time
            self._last_calib_refresh_time = t

            # If recording, add an event row so you can split the file later
            if self.recording_ctrl.is_recording:
                self.recording_ctrl.add_data_sample({"time": t, "event": "CALIB_REFRESH"})

            self.status_label.setText("Static calibration refreshed (instant)")
            self.instruction_label.setText("Pose re-zeroed to current neutral stance")

        finally:
            self._calibration_updating = False

    def _setup_window_size(self):
        """Auto-scale window to fit screen comfortably."""
        # Get screen geometry
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()
        
        print(f"[App] Screen resolution: {screen_width}x{screen_height}")
        
        # Determine optimal window size based on screen
        if screen_width >= 1920 and screen_height >= 1080:
            # Large screen (Full HD or better) - use 85% of screen
            window_width = int(screen_width * 0.85)
            window_height = int(screen_height * 0.85)
        elif screen_width >= 1366 and screen_height >= 768:
            # Medium screen (HD) - use 90% of screen
            window_width = int(screen_width * 0.90)
            window_height = int(screen_height * 0.85)
        elif screen_width >= 1024 and screen_height >= 768:
            # Small screen - use 95% of screen, enable scroll if needed
            window_width = int(screen_width * 0.95)
            window_height = int(screen_height * 0.90)
        else:
            # Very small screen - maximize window, will need scrolling
            window_width = int(screen_width * 0.98)
            window_height = int(screen_height * 0.92)
        
        # Set minimum size to ensure UI is usable
        self.setMinimumSize(1000, 700)
        
        # Set window size
        self.resize(window_width, window_height)
        
        # Center window on screen
        self.move(
            (screen_width - window_width) // 2,
            (screen_height - window_height) // 2
        )
        
        print(f"[App] Window size: {window_width}x{window_height}")

    def _init_ui(self):
        """Build user interface: Logo/Controls left | Joint Angles middle | Skeleton+Cameras right"""
        # Create menu bar
        menubar = self.menuBar()
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        trial_viewer_action = QtWidgets.QAction("Trial Viewer", self)
        trial_viewer_action.setShortcut("Ctrl+T")
        trial_viewer_action.triggered.connect(self._open_trial_viewer)
        tools_menu.addAction(trial_viewer_action)
        
        # Main container - check if we need scrolling
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        needs_scroll = screen.width() < 1280 or screen.height() < 800
        
        if needs_scroll:
            # Wrap everything in a scroll area for small screens
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            
            # Container widget for scroll area
            container = QtWidgets.QWidget()
            main_layout = QtWidgets.QHBoxLayout(container)
            
            scroll.setWidget(container)
            self.setCentralWidget(scroll)
        else:
            # No scroll needed - direct layout
            container = QtWidgets.QWidget()
            main_layout = QtWidgets.QHBoxLayout(container)
            self.setCentralWidget(container)
        
        main_layout.setSpacing(15)
        
        # ==================== LEFT COLUMN: Logo + Controls ====================
        left_column = QtWidgets.QVBoxLayout()
        left_column.setSpacing(10)
        
        # IKU logo at top of left column
        logo_label = QtWidgets.QLabel()
        logo_pixmap = QtGui.QPixmap("IKU_logo.png")
        if not logo_pixmap.isNull():
            # Scale logo based on screen size
            screen = QtWidgets.QApplication.desktop().screenGeometry()
            if screen.width() >= 1920:
                logo_width = 320
            elif screen.width() >= 1366:
                logo_width = 280
            elif screen.width() >= 1024:
                logo_width = 240
            else:
                logo_width = 200
            
            scaled_pixmap = logo_pixmap.scaledToWidth(logo_width, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(QtCore.Qt.AlignCenter)
            left_column.addWidget(logo_label)
        else:
            print("[App] Warning: IKU_logo.png not found")
        
        # ==================== Sensor Count Display ====================
        sensor_count_group = QtWidgets.QGroupBox("Connected Sensors")
        sensor_count_layout = QtWidgets.QHBoxLayout()
        sensor_count_layout.setSpacing(10)
        
        sensor_icon_label = QtWidgets.QLabel("📡")
        sensor_icon_label.setStyleSheet("font-size: 14pt;")
        sensor_count_layout.addWidget(sensor_icon_label)
        
        self.sensor_count_label = QtWidgets.QLabel("Detecting...")
        self.sensor_count_label.setStyleSheet(
            "font-size: 11pt; font-weight: bold; font-family: 'Courier New';"
        )
        sensor_count_layout.addWidget(self.sensor_count_label, 1)
        
        sensor_count_layout.addStretch()
        sensor_count_group.setLayout(sensor_count_layout)
        left_column.addWidget(sensor_count_group)
        
        # ==================== Session Configuration ====================
        session_group = QtWidgets.QGroupBox("Recording Session")
        session_layout = QtWidgets.QVBoxLayout()
        session_layout.setSpacing(5)
        
        # Session info display
        session_info_layout = QtWidgets.QHBoxLayout()
        
        session_info_label = QtWidgets.QLabel("Subject:")
        session_info_label.setStyleSheet("font-size: 9pt; color: #555;")
        session_info_layout.addWidget(session_info_label)
        
        self.session_info_display = QtWidgets.QLabel("⚠️ Not configured")
        self.session_info_display.setStyleSheet(
            "font-size: 9pt; font-family: 'Courier New'; color: #e67e22; font-weight: bold; "
            "background-color: #fef5e7; padding: 3px; border: 1px solid #f39c12; border-radius: 2px;"
        )
        session_info_layout.addWidget(self.session_info_display, 1)
        
        session_layout.addLayout(session_info_layout)
        
        # Configure button
        config_btn = QtWidgets.QPushButton("⚙️ Configure Session Info")
        config_btn.setStyleSheet(
            "padding: 6px; font-size: 10pt; background-color: #3498db; "
            "color: white; font-weight: bold; border-radius: 3px;"
        )
        config_btn.clicked.connect(self._configure_session)
        session_layout.addWidget(config_btn)
        
        session_group.setLayout(session_layout)
        left_column.addWidget(session_group)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.status_label.setWordWrap(True)
        left_column.addWidget(self.status_label)
        
        # Instruction label
        self.instruction_label = QtWidgets.QLabel("")
        self.instruction_label.setStyleSheet("color: #666; font-size: 9pt;")
        self.instruction_label.setWordWrap(True)
        left_column.addWidget(self.instruction_label)
        
        # Calibration controls
        calib_group = QtWidgets.QGroupBox("Calibration")
        calib_layout = QtWidgets.QVBoxLayout()
        
        # Calibration mode selector
        mode_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        self.calib_mode_combo = QtWidgets.QComboBox()
        self.calib_mode_combo.addItem("Dual-Pose (N + T)")
        self.calib_mode_combo.addItem("Single N-Pose")
        self.calib_mode_combo.addItem("Single T-Pose")
        self.calib_mode_combo.setToolTip("Select calibration mode")
        self.calib_mode_combo.currentIndexChanged.connect(self._on_calib_mode_changed)
        mode_layout.addWidget(self.calib_mode_combo)
        mode_layout.addStretch()
        calib_layout.addLayout(mode_layout)
        
        # N-Pose calibration row
        n_pose_layout = QtWidgets.QHBoxLayout()
        self.n_pose_btn = QtWidgets.QPushButton("1. Capture N-Pose")
        self.n_pose_btn.clicked.connect(self._on_n_pose_calibration)
        self.n_pose_btn.setEnabled(False)  # Disabled until session configured
        self.n_pose_btn.setToolTip("Configure session first")
        n_pose_layout.addWidget(self.n_pose_btn)
        
        self.n_pose_status = QtWidgets.QLabel("Not captured")
        self.n_pose_status.setStyleSheet("color: #888; font-size: 9pt;")
        n_pose_layout.addWidget(self.n_pose_status)
        n_pose_layout.addStretch()
        calib_layout.addLayout(n_pose_layout)
        
        # T-Pose calibration row
        t_pose_layout = QtWidgets.QHBoxLayout()
        self.t_pose_btn = QtWidgets.QPushButton("2. Capture T-Pose")
        self.t_pose_btn.clicked.connect(self._on_t_pose_calibration)
        self.t_pose_btn.setEnabled(False)
        t_pose_layout.addWidget(self.t_pose_btn)
        
        self.t_pose_status = QtWidgets.QLabel("Not captured")
        self.t_pose_status.setStyleSheet("color: #888; font-size: 9pt;")
        t_pose_layout.addWidget(self.t_pose_status)
        t_pose_layout.addStretch()
        calib_layout.addLayout(t_pose_layout)

        # Functional calibration (hidden)
        self.func_calib_btn = QtWidgets.QPushButton("Functional Calibration")
        self.func_calib_btn.clicked.connect(self._on_functional_calibration_v2)
        self.func_calib_btn.setEnabled(False)
        self.func_calib_btn.setVisible(False)
        calib_layout.addWidget(self.func_calib_btn)
        
        self.restart_func_btn = QtWidgets.QPushButton("↺")
        self.restart_func_btn.clicked.connect(self._on_restart_functional_calibration_v2)
        self.restart_func_btn.setEnabled(False)
        self.restart_func_btn.setVisible(False)
        
        self.func_status_label = QtWidgets.QLabel("Not performed")
        self.func_status_label.setStyleSheet("color: #888;")
        self.func_status_label.setVisible(False)
        
        # Movement instruction display (for functional calibration)
        self.movement_instruction_label = QtWidgets.QLabel("")
        self.movement_instruction_label.setWordWrap(True)
        self.movement_instruction_label.setVisible(False)
        
        # Movement progress display
        self.movement_progress_label = QtWidgets.QLabel("")
        self.movement_progress_label.setVisible(False)
        
        # Post-static calibration (hidden)
        self.post_static_calib_btn = QtWidgets.QPushButton("3. Final Static")
        self.post_static_calib_btn.clicked.connect(self._on_post_static_calibration)
        self.post_static_calib_btn.setEnabled(False)
        self.post_static_calib_btn.setVisible(False)
        
        self.post_static_status_label = QtWidgets.QLabel("Not performed")
        self.post_static_status_label.setStyleSheet("color: #888;")
        self.post_static_status_label.setVisible(False)
        
        # Progress bar
        self.calib_progress_bar = QtWidgets.QProgressBar()
        self.calib_progress_bar.setVisible(False)
        
        # Movement timer
        self.movement_timer_label = QtWidgets.QLabel("")
        self.movement_timer_label.setVisible(False)
        
        calib_group.setLayout(calib_layout)
        left_column.addWidget(calib_group)
        
        # Recording control
        self.record_btn = QtWidgets.QPushButton("Start Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.toggled.connect(self._on_record_toggled)
        self.record_btn.setEnabled(False)  # Disabled until session configured
        self.record_btn.setToolTip("Configure session first")
        self.record_btn.setStyleSheet("background-color: #95a5a6; color: white; font-weight: bold; padding: 10px;")  # Grey when disabled
        left_column.addWidget(self.record_btn)
        
        left_column.addStretch()  # Push content to top
        
        # Sensor-Segment Mapping Configurator (at bottom)
        sensor_config_btn = QtWidgets.QPushButton("⚙️ Configure Sensor-Segment Mapping")
        sensor_config_btn.setStyleSheet(
            "padding: 6px; font-size: 9pt; background-color: #95a5a6; "
            "color: white; border-radius: 3px;"
        )
        sensor_config_btn.setToolTip("Edit sensor ID to segment mappings (for sensor replacement)")
        sensor_config_btn.clicked.connect(self._configure_sensor_mapping)
        left_column.addWidget(sensor_config_btn)
        
        # Add left column to main layout
        main_layout.addLayout(left_column, stretch=1)
        
        # ==================== MIDDLE COLUMN: Joint Angle Table ====================
        middle_column = QtWidgets.QVBoxLayout()
        
        # Joint angle table
        self.angle_table = CompactAngleTableWidget(self, use_quaternions=USE_QUATERNIONS)
        middle_column.addWidget(self.angle_table, 1)
        
        # Add middle column to main layout (larger now!)
        main_layout.addLayout(middle_column, stretch=3)
        
        # ==================== RIGHT COLUMN: Skeleton Visualization ====================
        right_column = QtWidgets.QVBoxLayout()
        right_column.setSpacing(10)
        
        # Skeleton label and zoom controls
        skeleton_header = QtWidgets.QHBoxLayout()
        skeleton_label = QtWidgets.QLabel("3D Skeleton")
        skeleton_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        skeleton_header.addWidget(skeleton_label)
        
        skeleton_header.addStretch()
        
        # Zoom controls
        zoom_in_btn = QtWidgets.QPushButton("🔍+")
        zoom_in_btn.setToolTip("Zoom In")
        zoom_in_btn.setMaximumWidth(50)
        zoom_in_btn.clicked.connect(self._on_zoom_in)
        skeleton_header.addWidget(zoom_in_btn)
        
        zoom_out_btn = QtWidgets.QPushButton("🔍-")
        zoom_out_btn.setToolTip("Zoom Out")
        zoom_out_btn.setMaximumWidth(50)
        zoom_out_btn.clicked.connect(self._on_zoom_out)
        skeleton_header.addWidget(zoom_out_btn)
        
        zoom_reset_btn = QtWidgets.QPushButton("Reset")
        zoom_reset_btn.setToolTip("Reset Zoom")
        zoom_reset_btn.setMaximumWidth(60)
        zoom_reset_btn.clicked.connect(self._on_zoom_reset)
        skeleton_header.addWidget(zoom_reset_btn)
        
        right_column.addLayout(skeleton_header)
        
        # 3D skeleton view (larger now!)
        self.skeleton_canvas = Skeleton3DCanvas(self)
        right_column.addWidget(self.skeleton_canvas, 1)
        
        # Camera view controls
        cam_layout = QtWidgets.QHBoxLayout()
        cam_label = QtWidgets.QLabel("View:")
        cam_layout.addWidget(cam_label)
        
        btn_default = QtWidgets.QPushButton("Default")
        btn_front = QtWidgets.QPushButton("Front")
        btn_back = QtWidgets.QPushButton("Back")
        btn_left = QtWidgets.QPushButton("Left")
        btn_right = QtWidgets.QPushButton("Right")
        btn_top = QtWidgets.QPushButton("Top")

        btn_default.clicked.connect(self.skeleton_canvas.view_default)
        btn_front.clicked.connect(self.skeleton_canvas.view_front)
        btn_back.clicked.connect(self.skeleton_canvas.view_back)
        btn_left.clicked.connect(self.skeleton_canvas.view_left)
        btn_right.clicked.connect(self.skeleton_canvas.view_right)
        btn_top.clicked.connect(self.skeleton_canvas.view_top)
        
        for btn in (btn_default, btn_front, btn_back, btn_left, btn_right, btn_top):
            btn.setMaximumWidth(70)
            cam_layout.addWidget(btn)
        
        cam_layout.addStretch()
        right_column.addLayout(cam_layout)
        
        # ==================== Camera Feeds (Tabbed under skeleton) ====================
        camera_section = QtWidgets.QVBoxLayout()
        camera_section.setSpacing(5)
        
        # Camera tabs
        self.camera_tabs = QtWidgets.QTabWidget()
        self.camera_tabs.setMaximumHeight(300)  # Limit height
        
        # Create tab for each camera
        self.camera_preview_widgets = {}
        for i in range(4):  # Always create 4 tabs
            preview = CameraPreviewWidget(camera_id=i)
            # Make preview larger since it's in a tab
            preview.setFixedSize(480, 360)  # 4:3 aspect ratio, larger
            self.camera_preview_widgets[i] = preview
            self.camera_tabs.addTab(preview, f"Camera {i}")
        
        camera_section.addWidget(self.camera_tabs)
        
        # Camera checkboxes (always visible below tabs)
        checkbox_layout = QtWidgets.QHBoxLayout()
        checkbox_layout.setSpacing(10)
        
        checkbox_label = QtWidgets.QLabel("Enable:")
        checkbox_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        checkbox_layout.addWidget(checkbox_label)
        
        self.camera_checkboxes = {}
        for i in range(4):
            checkbox = QtWidgets.QCheckBox(f"Cam {i}")
            checkbox.setStyleSheet("font-size: 9pt;")
            checkbox.stateChanged.connect(
                lambda state, cam_id=i: self._on_camera_toggled(cam_id, state)
            )
            self.camera_checkboxes[i] = checkbox
            checkbox_layout.addWidget(checkbox)
        
        checkbox_layout.addStretch()
        
        # Recording indicator
        self.camera_recording_indicator = QtWidgets.QLabel("⚫ Not Recording")
        self.camera_recording_indicator.setStyleSheet(
            "font-size: 9pt; color: #7f8c8d; padding: 3px; "
            "background-color: #ecf0f1; border-radius: 3px;"
        )
        checkbox_layout.addWidget(self.camera_recording_indicator)
        
        camera_section.addLayout(checkbox_layout)
        
        right_column.addLayout(camera_section)
        
        # Add right column to main layout
        main_layout.addLayout(right_column, stretch=2)
        
        # Finalize scroll area if needed
        if needs_scroll:
            scroll.setWidget(container)
            print("[App] Scroll area enabled for small screen")
        else:
            self.setLayout(main_layout)
    
    def _on_zoom_in(self):
        """Zoom in on skeleton view"""
        # Get current view angles
        elev = self.skeleton_canvas.ax.elev
        azim = self.skeleton_canvas.ax.azim
        
        # Matplotlib 3D doesn't expose distance directly, so we scale the axes limits
        xlim = self.skeleton_canvas.ax.get_xlim()
        ylim = self.skeleton_canvas.ax.get_ylim()
        zlim = self.skeleton_canvas.ax.get_zlim()
        
        # Zoom in by reducing limits by 10%
        factor = 0.9
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        x_range = (xlim[1] - xlim[0]) * factor / 2
        y_range = (ylim[1] - ylim[0]) * factor / 2
        z_range = (zlim[1] - zlim[0]) * factor / 2
        
        self.skeleton_canvas.ax.set_xlim([x_center - x_range, x_center + x_range])
        self.skeleton_canvas.ax.set_ylim([y_center - y_range, y_center + y_range])
        self.skeleton_canvas.ax.set_zlim([z_center - z_range, z_center + z_range])
        
        self.skeleton_canvas.draw()
    
    def _on_zoom_out(self):
        """Zoom out from skeleton view"""
        # Get current limits
        xlim = self.skeleton_canvas.ax.get_xlim()
        ylim = self.skeleton_canvas.ax.get_ylim()
        zlim = self.skeleton_canvas.ax.get_zlim()
        
        # Zoom out by increasing limits by 10%
        factor = 1.1
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        x_range = (xlim[1] - xlim[0]) * factor / 2
        y_range = (ylim[1] - ylim[0]) * factor / 2
        z_range = (zlim[1] - zlim[0]) * factor / 2
        
        self.skeleton_canvas.ax.set_xlim([x_center - x_range, x_center + x_range])
        self.skeleton_canvas.ax.set_ylim([y_center - y_range, y_center + y_range])
        self.skeleton_canvas.ax.set_zlim([z_center - z_range, z_center + z_range])
        
        self.skeleton_canvas.draw()
    
    def _on_zoom_reset(self):
        """Reset zoom to default"""
        # Reset to default axis limits (tighter, optimized for skeleton)
        self.skeleton_canvas.ax.set_xlim([-0.8, 0.8])
        self.skeleton_canvas.ax.set_ylim([-0.8, 0.8])
        self.skeleton_canvas.ax.set_zlim([-0.1, 2.1])
        self.skeleton_canvas.draw()
    
    def _on_static_calibration(self):
        """Start static calibration countdown"""
        if self.static_calib_pending:
            return
        
        # Get selected pose type
        pose_index = self.pose_combo.currentIndex()
        self.selected_pose = "n_pose" if pose_index == 0 else "t_pose"
        
        self.static_calib_pending = True
        self.static_calib_start_time = time.monotonic()
        self.status_label.setText(f"Static calibration starting in 5 seconds ({self.selected_pose.replace('_', '-').upper()})")
        
        if self.selected_pose == "n_pose":
            self.instruction_label.setText("Get into N-POSE: feet shoulder-width apart, arms at sides, looking forward")
        else:
            self.instruction_label.setText("Get into T-POSE: feet shoulder-width apart, arms extended laterally (out to sides), palms down, looking forward")
        
        self.static_status_label.setText("Waiting...")
        print(f"[App] Static calibration countdown started ({self.selected_pose})")
        self.record_btn.setEnabled(True)
        self.refresh_calib_btn.setVisible(True)
        self.refresh_calib_btn.setEnabled(True)
    
    def _on_post_static_calibration(self):
        """Start post-functional static calibration countdown"""
        if self.post_static_calib_pending:
            return
        
        # Check if functional calibration v2 is complete
        if not self.func_calib_v2.is_complete():
            self.status_label.setText("Complete functional calibration first")
            return
        
        self.post_static_calib_pending = True
        self.post_static_calib_start_time = time.monotonic()
        
        # Use the SAME pose as initial static calibration
        pose_type = getattr(self, 'selected_pose', 'n_pose')
        
        self.status_label.setText(f"Final static calibration starting in 5 seconds ({pose_type.replace('_', '-').upper()})")
        
        if pose_type == "n_pose":
            self.instruction_label.setText("Get into N-POSE: feet shoulder-width apart, arms at sides, looking forward")
        else:
            self.instruction_label.setText("Get into T-POSE: feet shoulder-width apart, arms extended laterally (out to sides), palms down, looking forward")
        
        self.post_static_status_label.setText("Waiting...")
        print(f"[App] Post-functional static calibration countdown started ({pose_type})")
    
    def _on_restart_static_calibration(self):
        """Restart static calibration"""
        self.calibration.reset_static_calibration()
        self.func_calib_btn.setEnabled(False)
        self.func_status_label.setText("Not performed")
        self.post_static_calib_btn.setEnabled(False)
        self.post_static_calib_btn.setVisible(False)
        self.record_btn.setEnabled(True)
        self.refresh_calib_btn.setVisible(True)
        self.status_label.setText("Static calibration reset - Ready to recalibrate")
        self.instruction_label.setText("Click 'Initial Static Calibration' to restart")
    
    def _on_restart_functional_calibration(self):
        """Restart functional calibration"""
        self.calibration.reset_full_calibration()
        self.func_calib_btn.setEnabled(False)
        self.post_static_calib_btn.setEnabled(False)
        self.post_static_calib_btn.setVisible(False)
        self.post_static_status_label.setText("Not performed")
        self.post_static_status_label.setVisible(False)
        self.record_btn.setEnabled(False)
        self.status_label.setText("Functional calibration reset")
        self.instruction_label.setText("Click 'Full Functional Calibration' to restart")
    
    def _perform_static_calibration(self):
        """Actually perform the static calibration"""
        raw_sample = self._get_imu_sample()
        if not raw_sample:
            self.status_label.setText("Error: No IMU data available")
            self.static_status_label.setText("Failed")
            return
        
        # Convert to rotations
        sensor_rots = {
            name: R.from_quat([q[1], q[2], q[3], q[0]]) 
            for name, q in raw_sample.items()
        }

        world_rots = self.world_converter.convert_orientations(sensor_rots)
        world_rots = apply_mounting(world_rots)
        self._latest_world_rots = world_rots
        
        # Get selected pose type
        pose_type = getattr(self, 'selected_pose', 'n_pose')
        
        # Check if this is post-static calibration
        if self.post_static_calib_pending:
            # Apply functional calibration alignments first, then compute offsets
            self.calibration.static_calibration(world_rots, is_post_calibration=True, pose_type=pose_type)
            
            # Clear pending flag
            self.post_static_calib_pending = False
            
            self.status_label.setText(f"Final static calibration complete! System ready ({pose_type.replace('_', '-').upper()})")
            self.instruction_label.setText("Calibration complete! You can now record data with accurate joint angles.")
            self.post_static_status_label.setText("✓ Complete")
            self.static_status_label.setText("✓ Complete")
            self.record_btn.setEnabled(True)

            self.restart_static_btn.setEnabled(True)
            self.static_calib_btn.setEnabled(False)
            self.post_static_calib_btn.setEnabled(False)
            
            print(f"[App] Post-static calibration complete with alignments applied")

        else:
            self.calibration.static_calibration(world_rots, is_post_calibration=False, pose_type=pose_type)
            self.status_label.setText(f"Initial static calibration complete ({pose_type.replace('_', '-').upper()})")
            self.instruction_label.setText("System ready! You can now record data.")
            self.static_status_label.setText("✓ Complete")
            self.restart_static_btn.setEnabled(True)
            
            # Disable functional calibration (not working well with pelvis-relative yet)
            self.func_calib_btn.setEnabled(False)
            self.func_calib_btn.setVisible(False)
            self.static_calib_btn.setEnabled(False)
            
            # Enable recording immediately
            self.record_btn.setEnabled(True)
        
        self.static_calib_pending = False
    
    def _on_functional_calibration(self):
        """Start functional calibration sequence"""
        if not self.calibration.is_static_calibrated:
            self.status_label.setText("Perform static calibration first")
            return
        
        print("[App] Starting full functional calibration sequence")
        self.calibration.start_functional_calibration_sequence()
        self._start_next_calibration_movement()
        self.func_calib_btn.setEnabled(False)
        self.restart_func_btn.setEnabled(False)
        self.func_status_label.setText("In progress...")
    
    def _start_next_calibration_movement(self):
        """Start recording the next calibration movement."""
        movement = self.calibration._get_current_movement()
        if not movement:
            # All movements complete
            self.calibration.functional_calibration_complete()
            self.status_label.setText("Functional calibration complete!")
            self.instruction_label.setText("Ready for final static calibration - return to neutral pose")
            self.func_status_label.setText("✓ Complete")
            
            # Enable post-static calibration button
            self.post_static_calib_btn.setEnabled(True)
            self.post_static_calib_btn.setVisible(True)
            
            self.calib_progress_bar.setVisible(False)
            self.movement_timer_label.setVisible(False)
            return
        
        # Start pause phase
        self.in_pause_phase = True
        self.pause_start_time = time.monotonic()
        self.calibration.start_pause_phase()
        
        self.status_label.setText(f"Get ready: {movement['description']}")
        self.instruction_label.setText(f"Pause: {movement['pause']} seconds. Next: {movement['instruction']}")
        
        # Show timer
        self.movement_timer_label.setVisible(True)
        self.movement_timer_label.setText(f"Pause: {movement['pause']:.0f}s")
        
        print(f"[App] Pause before movement: {movement['name']}")
    
    def _start_movement_recording(self):
        """Actually start recording the movement (after pause)."""
        movement = self.calibration._get_current_movement()
        if not movement:
            return
        
        self.movement_recording = movement["name"]
        self.movement_start_time = time.monotonic()
        self.current_movement_duration = movement["duration"]
        self.calibration.start_movement_recording(self.movement_recording)
        
        self.status_label.setText(f"Calibration: {movement['description']}")
        self.instruction_label.setText(movement["instruction"])
        
        # Update progress bar
        self.calib_progress_bar.setVisible(True)
        overall_progress = int(self.calibration.calibration_progress * 100)
        self.calib_progress_bar.setValue(overall_progress)
        
        print(f"[App] Starting movement: {movement['name']}")
    
    def _on_record_toggled(self, checked):
        """Toggle recording (IMU + webcam)"""
        if checked:
            # Check if session is configured
            if not self.recording_ctrl.subject_id or not self.recording_ctrl.session_id:
                # Prompt for session info
                if not self.recording_ctrl.prompt_session_info(parent=self):
                    # User cancelled
                    self.record_btn.setChecked(False)
                    return
                # Update display
                self._update_session_display()
            
            try:
                # Start recording through controller
                self.recording_ctrl.start_recording()
                session_folder = self.recording_ctrl.get_session_folder()
                
                # Save calibration info to session folder
                self._save_calibration_info()
                
                # Get filename prefix for cameras
                info = self.recording_ctrl.get_session_info()
                filename_prefix = f"{info['subject_id']}_{info['session_id']}_{info['timestamp']}"
                
                # Start webcam recording with custom filenames
                self.webcam_recorder.start_recording(session_folder, filename_prefix=filename_prefix)
                self.camera_recording_indicator.setText("🔴 Recording")
                self.camera_recording_indicator.setStyleSheet(
                    "font-size: 9pt; color: white; padding: 3px; "
                    "background-color: #e74c3c; border-radius: 3px; font-weight: bold;"
                )
                
                # Update UI
                self.record_btn.setText("Stop Recording")
                self.record_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;")
                self.status_label.setText(f"Recording: {info['subject_id']} - {info['session_id']}")
                self.instruction_label.setText(f"Saving to: {session_folder}")
                
                print(f"[App] Started recording: {info['subject_id']}/{info['session_id']}")
                print(f"[App] Session folder: {session_folder}")
                
            except ValueError as e:
                # Missing subject/session ID
                QtWidgets.QMessageBox.warning(self, "Configuration Required", str(e))
                self.record_btn.setChecked(False)
                return
        else:
            # Stop webcam recording
            self.webcam_recorder.stop_recording()
            self.camera_recording_indicator.setText("⚫ Not Recording")
            self.camera_recording_indicator.setStyleSheet(
                "font-size: 9pt; color: #7f8c8d; padding: 3px; "
                "background-color: #ecf0f1; border-radius: 3px;"
            )
            
            # Stop controller recording (saves IMU data)
            session_folder = self.recording_ctrl.stop_recording()
            
            # Update UI
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
            
            if session_folder:
                self.status_label.setText(f"Recording saved!")
                self.instruction_label.setText(f"Saved to: {session_folder}")
                info = self.recording_ctrl.get_session_info()
                print(f"[App] Recording complete: {info['subject_id']}/{info['session_id']}")
                print(f"[App] Files saved to: {session_folder}")

    
    def _get_imu_sample(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current IMU sample"""
        if hasattr(self.imu_stream, 'get_sample'):
            return self.imu_stream.get_sample()
        elif hasattr(self.imu_stream, 'get_latest_quats'):
            quats = self.imu_stream.get_latest_quats()
            if len(quats) < len(SEGMENT_NAMES):
                return None
            return quats
        return None
    
    def _on_imu_update(self):
        """Process new IMU data"""
        if self._calibration_updating:
            return

        raw_sample = self._get_imu_sample()
        if not raw_sample:
            return
        
        # Validate sensor count (only on first successful read)
        if not hasattr(self, '_sensor_count_validated'):
            self._validate_sensor_count(len(raw_sample))
            self._sensor_count_validated = True
        
        # Convert to rotations early so we can use them in all paths
        sensor_rots = {name: R.from_quat([q[1], q[2], q[3], q[0]]) 
                      for name, q in raw_sample.items()}
        world_rots = self.world_converter.convert_orientations(sensor_rots)
        world_rots = apply_mounting(world_rots)
        
        # Handle N-pose calibration countdown and recording
        if hasattr(self, 'n_pose_pending') and self.n_pose_pending:
            elapsed = time.monotonic() - self.n_pose_start_time
            remaining = STATIC_CALIB_DELAY - elapsed
            if remaining > 0:
                self.status_label.setText(f"N-pose calibration in {remaining:.1f}s")
                self.n_pose_status.setText(f"{remaining:.1f}s")
            else:
                # Countdown finished, start recording
                if not hasattr(self, 'n_pose_recording_active'):
                    self._capture_n_pose()
                    return
        
        # Handle N-pose recording (5 seconds)
        if hasattr(self, 'n_pose_recording_active') and self.n_pose_recording_active:
            elapsed = time.monotonic() - self.n_pose_recording_start
            remaining = 5.0 - elapsed
            
            # Record sample with both raw and processed data
            row = {'time': elapsed}
            
            # Record raw sensor quaternions
            for sensor_name, quat in raw_sample.items():
                row[f"raw_{sensor_name}_qw"] = quat[0]
                row[f"raw_{sensor_name}_qx"] = quat[1]
                row[f"raw_{sensor_name}_qy"] = quat[2]
                row[f"raw_{sensor_name}_qz"] = quat[3]
            
            # Record world-frame orientations (after conversion)
            for seg_name in SEGMENT_NAMES:
                quat = world_rots[seg_name].as_quat()  # [x, y, z, w]
                row[f"world_{seg_name}_qw"] = quat[3]
                row[f"world_{seg_name}_qx"] = quat[0]
                row[f"world_{seg_name}_qy"] = quat[1]
                row[f"world_{seg_name}_qz"] = quat[2]
            
            self.n_pose_recording.append(row)
            
            self.status_label.setText(f"Recording N-pose... {remaining:.1f}s")
            self.n_pose_status.setText(f"Recording {remaining:.1f}s")
            
            if elapsed >= 5.0:
                self._finalize_n_pose()
            return
        
        # Handle T-pose calibration countdown and recording
        if hasattr(self, 't_pose_pending') and self.t_pose_pending:
            elapsed = time.monotonic() - self.t_pose_start_time
            remaining = STATIC_CALIB_DELAY - elapsed
            if remaining > 0:
                self.status_label.setText(f"T-pose calibration in {remaining:.1f}s")
                self.t_pose_status.setText(f"{remaining:.1f}s")
            else:
                # Countdown finished, start recording
                if not hasattr(self, 't_pose_recording_active'):
                    self._capture_t_pose()
                    return
        
        # Handle T-pose recording (5 seconds)
        if hasattr(self, 't_pose_recording_active') and self.t_pose_recording_active:
            elapsed = time.monotonic() - self.t_pose_recording_start
            remaining = 5.0 - elapsed
            
            # Record sample with both raw and processed data
            row = {'time': elapsed}
            
            # Record raw sensor quaternions
            for sensor_name, quat in raw_sample.items():
                row[f"raw_{sensor_name}_qw"] = quat[0]
                row[f"raw_{sensor_name}_qx"] = quat[1]
                row[f"raw_{sensor_name}_qy"] = quat[2]
                row[f"raw_{sensor_name}_qz"] = quat[3]
            
            # Record world-frame orientations (after conversion)
            for seg_name in SEGMENT_NAMES:
                quat = world_rots[seg_name].as_quat()  # [x, y, z, w]
                row[f"world_{seg_name}_qw"] = quat[3]
                row[f"world_{seg_name}_qx"] = quat[0]
                row[f"world_{seg_name}_qy"] = quat[1]
                row[f"world_{seg_name}_qz"] = quat[2]
            
            self.t_pose_recording.append(row)
            
            self.status_label.setText(f"Recording T-pose... {remaining:.1f}s")
            self.t_pose_status.setText(f"Recording {remaining:.1f}s")
            
            if elapsed >= 5.0:
                self._finalize_t_pose()
            return
        
        # Handle initial static calibration countdown
        if self.static_calib_pending:
            elapsed = time.monotonic() - self.static_calib_start_time
            remaining = STATIC_CALIB_DELAY - elapsed
            if remaining > 0:
                self.status_label.setText(f"Initial static calibration in {remaining:.1f}s")
                self.static_status_label.setText(f"{remaining:.1f}s")
            else:
                print("[App] Starting initial static calibration now!")
                self.static_calib_pending = False
                self._perform_static_calibration()
            return
        
        # Handle post-static calibration countdown
        if self.post_static_calib_pending and hasattr(self, 'post_static_calib_start_time'):
            elapsed = time.monotonic() - self.post_static_calib_start_time
            remaining = STATIC_CALIB_DELAY - elapsed
            if remaining > 0:
                self.status_label.setText(f"Final static calibration in {remaining:.1f}s")
                self.post_static_status_label.setText(f"{remaining:.1f}s")
            else:
                print("[App] Starting final static calibration now!")
                # Don't clear flag here - let _perform_static_calibration handle it
                self._perform_static_calibration()
            return
        
        # Handle pause phase between movements
        if self.in_pause_phase and self.pause_start_time:
            elapsed = time.monotonic() - self.pause_start_time
            movement = self.calibration._get_current_movement()
            if movement:
                remaining = max(0, movement["pause"] - elapsed)
                self.movement_timer_label.setText(f"Pause: {remaining:.1f}s")
                
                if elapsed >= movement["pause"]:
                    self.in_pause_phase = False
                    self._start_movement_recording()
            return
        
        # Handle functional calibration movement recording
        if self.movement_recording is not None:
            elapsed = time.monotonic() - self.movement_start_time
            self.calibration.record_movement_sample(self.movement_recording, world_rots)
            
            # Update timer display
            remaining = max(0, self.current_movement_duration - elapsed)
            self.movement_timer_label.setText(f"{remaining:.1f}s")
            
            # Update progress bar
            progress_pct = min(100, int((elapsed / self.current_movement_duration) * 100))
            self.calib_progress_bar.setValue(progress_pct)
            
            if elapsed > self.current_movement_duration:
                # Movement complete, process it
                print(f"[App] Movement {self.movement_recording} complete, processing...")
                success = self.calibration.movement_completed(self.movement_recording)
                
                if success:
                    # Start next movement (which begins with pause)
                    self.movement_recording = None
                    self._start_next_calibration_movement()
                else:
                    self.status_label.setText(f"Failed to process {self.movement_recording}")
                    self.instruction_label.setText("Please try the movement again")
                    self.movement_recording = None
                    self.func_calib_btn.setEnabled(True)
            
            # Don't continue to regular processing during movement recording
            return
        
        # Apply calibration if available
        if not self.calibration.is_static_calibrated:
            return
        
        segment_rots = self.calibration.get_calibrated_orientations(world_rots)
        
        # Compute joint rotations (returns both quaternions and Euler angles)
        joint_data = self.kinematics.compute_joint_rotations(segment_rots)
        
        # Compute segment positions once and cache for display update
        self._cached_segment_positions = self.kinematics.compute_segment_positions(segment_rots)
        self._cached_joint_data = joint_data
        self._cached_segment_rots = segment_rots  # Cache for heading lock
        
        # Record samples for functional calibration v2 if active
        if self.func_calib_v2.is_recording:
            self.func_calib_v2.record_sample(world_rots)
        
        # Store time for recording
        t = time.monotonic() - self.start_time
        
        # Record if enabled
        if self.recording_ctrl.is_recording:
            row = {'time': t, 'event': ''}
            
            # Record raw sensor data (before calibration)
            for sensor_name, quat in raw_sample.items():
                # Raw quaternions [w, x, y, z]
                row[f"raw_{sensor_name}_qw"] = quat[0]
                row[f"raw_{sensor_name}_qx"] = quat[1]
                row[f"raw_{sensor_name}_qy"] = quat[2]
                row[f"raw_{sensor_name}_qz"] = quat[3]
            
            # Record segment orientations (absolute world-frame rotations, after calibration)
            for segment_name, rotation in segment_rots.items():
                # Get quaternion (w, x, y, z)
                quat = rotation.as_quat()  # Returns [x, y, z, w] in scipy
                row[f"seg_{segment_name}_qw"] = quat[3]  # w is last in scipy
                row[f"seg_{segment_name}_qx"] = quat[0]
                row[f"seg_{segment_name}_qy"] = quat[1]
                row[f"seg_{segment_name}_qz"] = quat[2]
            
            # Record joint rotations (relative rotations between segments)
            # Both quaternions and Euler angles for maximum compatibility
            for joint_name, (quat, angles) in joint_data.items():
                # Quaternion components [w, x, y, z]
                row[f"joint_{joint_name}_qw"] = quat[0]
                row[f"joint_{joint_name}_qx"] = quat[1]
                row[f"joint_{joint_name}_qy"] = quat[2]
                row[f"joint_{joint_name}_qz"] = quat[3]
                
                # Euler angles (degrees)
                row[f"joint_{joint_name}_euler_0"] = angles[0]
                row[f"joint_{joint_name}_euler_1"] = angles[1]
                row[f"joint_{joint_name}_euler_2"] = angles[2]
            
            self.recording_ctrl.add_data_sample(row)
    
    def _on_display_update(self):
        """Update visualization - OPTIMIZED to use cached data"""
        # Only update if we have at least initial static calibration
        if not self.calibration.is_static_calibrated:
            return
        
        # Update angle table (FAST!)
        if self._cached_joint_data is not None:
            # Extract just the Euler angles from (quat, angles) tuples
            angles_only = {joint: data[1] for joint, data in self._cached_joint_data.items()}
            self.angle_table.update_angles(angles_only)
        
        # Update pelvis orientation for pelvis-relative camera views
        if self._cached_segment_rots is not None and 'pelvis' in self._cached_segment_rots:
            self.skeleton_canvas.update_pelvis_orientation(self._cached_segment_rots['pelvis'])
            # Enable heading lock to keep skeleton facing forward in visualization
            self.skeleton_canvas.set_heading_lock(self._cached_segment_rots['pelvis'], enabled=True)
        
        # Update skeleton using cached positions (no recomputation!)
        if self._cached_segment_positions is not None:
            self.skeleton_canvas.update_skeleton(self._cached_segment_positions)
        
        # Static-calibration-only mode
        if self.calibration.is_static_calibrated:
            self.status_label.setText("Static calibrated – streaming")
    
    def _on_camera_update(self):
        """Update camera previews (runs at 10 FPS to save CPU)."""
        for camera_id in list(self.webcam_recorder.cameras.keys()):
            frame = self.webcam_recorder.get_preview_frame(camera_id)
            if frame is not None and camera_id in self.camera_preview_widgets:
                self.camera_preview_widgets[camera_id].update_frame(frame)
    
    def _detect_cameras(self):
        """Detect available cameras on startup."""
        available_cameras = self.webcam_recorder.detect_cameras(max_cameras=4)
        
        # Update checkbox states based on availability
        for cam_info in available_cameras:
            if cam_info.camera_id in self.camera_checkboxes:
                self.camera_checkboxes[cam_info.camera_id].setEnabled(True)
        
        # Disable checkboxes for unavailable cameras
        for cam_id in range(4):
            if not any(c.camera_id == cam_id for c in available_cameras):
                if cam_id in self.camera_checkboxes:
                    self.camera_checkboxes[cam_id].setEnabled(False)
                    self.camera_checkboxes[cam_id].setToolTip("Camera not detected")
        
        if available_cameras:
            print(f"[App] {len(available_cameras)} camera(s) detected and ready")
        else:
            print("[App] No cameras detected")
    
    def _validate_sensor_count(self, detected_count: int):
        """Validate detected sensor count and update UI display."""
        # Check if using fake stream
        if self.using_fake_stream:
            # Fake stream - show special message
            config = get_body_config(self.recording_ctrl.body_config)
            self.sensor_count_label.setText(f"0 / {config.num_sensors} - Using Fake Stream")
            self.sensor_count_label.setStyleSheet(
                "font-size: 11pt; font-weight: bold; font-family: 'Courier New'; color: #9b59b6;"
            )
            print(f"[App] Using fake IMU stream - {detected_count} simulated sensors")
            return
        
        # Get expected count from body configuration
        config = get_body_config(self.recording_ctrl.body_config)
        expected_count = config.num_sensors
        
        # Update sensor count display
        if detected_count == expected_count:
            # Perfect match
            self.sensor_count_label.setText(f"✓ {detected_count} / {expected_count}")
            self.sensor_count_label.setStyleSheet(
                "font-size: 11pt; font-weight: bold; font-family: 'Courier New'; color: #27ae60;"
            )
            print(f"[App] Sensor count valid: {detected_count}/{expected_count} for {config.display_name}")
        elif detected_count > expected_count:
            # More sensors than needed (OK, extras ignored)
            self.sensor_count_label.setText(f"✓ {detected_count} / {expected_count} (extras)")
            self.sensor_count_label.setStyleSheet(
                "font-size: 11pt; font-weight: bold; font-family: 'Courier New'; color: #3498db;"
            )
            print(f"[App] Sensor count: {detected_count}/{expected_count} (extra sensors will be ignored)")
        else:
            # Fewer sensors than expected (WARNING)
            self.sensor_count_label.setText(f"⚠️ {detected_count} / {expected_count}")
            self.sensor_count_label.setStyleSheet(
                "font-size: 11pt; font-weight: bold; font-family: 'Courier New'; color: #e67e22;"
            )
            print(f"[App] WARNING: Only {detected_count}/{expected_count} sensors detected for {config.display_name}")
            
            # Show warning dialog
            QtWidgets.QMessageBox.warning(
                self,
                "Insufficient Sensors",
                f"Body Configuration: {config.display_name}\n"
                f"Expected: {expected_count} sensors\n"
                f"Detected: {detected_count} sensors\n\n"
                f"Missing sensors will be shown in neutral pose.\n"
                f"Some joints may not move correctly.\n\n"
                f"Please check sensor connections or change body configuration."
            )
    
    def _configure_anthropometry(self):
        """Open anthropometry configuration dialog."""
        dialog = AnthropometryDialog(
            current_profile=self.anthro_manager.current_profile,
            parent=self
        )
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            # Get updated profile
            new_profile = dialog.get_profile()
            self.anthro_manager.set_profile(new_profile)
            
            # Update kinematics with new segment lengths
            self._update_kinematics_with_anthropometry()
            
            self.status_label.setText(f"Anthropometry updated: {new_profile.name}")
            self.instruction_label.setText(f"Skeleton scaled to {new_profile.mode} mode measurements")
            print(f"[App] Anthropometry profile applied: {new_profile.name}")
            print(f"[App] Mode: {new_profile.mode}")
    
    def _update_kinematics_with_anthropometry(self):
        """Update kinematic model with current anthropometry."""
        # Update BODY_MODEL with new segment lengths
        for seg_name, seg_def in BODY_MODEL.items():
            new_length = self.anthro_manager.get_segment_length(seg_name)
            seg_def.length = new_length
        
        # Reinitialize kinematics engine with updated body model
        self.kinematics = KinematicsEngine(BODY_MODEL)
        
        print("[App] Kinematics updated with new segment lengths")
    
    def _open_trial_viewer(self):
        """Open trial viewer window."""
        if self.trial_viewer_window is None or not self.trial_viewer_window.isVisible():
            self.trial_viewer_window = TrialViewerWindow(parent=self)
            self.trial_viewer_window.show()
        else:
            # Window already exists, bring to front
            self.trial_viewer_window.raise_()
            self.trial_viewer_window.activateWindow()
    
    def _configure_sensor_mapping(self):
        """Open sensor-segment mapping configurator dialog."""
        dialog = SensorMappingDialog(mapping_file="mapping.csv", parent=self)
        result = dialog.exec_()
        
        if result == QtWidgets.QDialog.Accepted:
            # Mappings were saved - suggest restart
            self.status_label.setText("Sensor mappings updated")
            self.instruction_label.setText("⚠️ Restart application for changes to take effect")
            print("[App] Sensor mappings updated - restart required")
    
    def _configure_session(self):
        """Open dialog to configure subject/session info."""
        success, anthro_profile = self.recording_ctrl.prompt_session_info(
            parent=self,
            current_anthro_profile=self.anthro_manager.current_profile
        )
        
        if success:
            self._update_session_display()
            info = self.recording_ctrl.get_session_info()
            
            # Create session folder immediately after configuration
            session_folder = self.recording_ctrl.create_session_folder()
            print(f"[App] Session folder created: {session_folder}")
            
            # Get body config details
            config = get_body_config(info['body_config'])
            
            # Apply anthropometry profile
            if anthro_profile:
                self.anthro_manager.set_profile(anthro_profile)
                self._update_kinematics_with_anthropometry()
                print(f"[App] Anthropometry: {anthro_profile.name} ({anthro_profile.mode} mode)")
            
            # Reset sensor count validation (will re-check on next IMU update)
            if hasattr(self, '_sensor_count_validated'):
                delattr(self, '_sensor_count_validated')
            
            # Update sensor count display for fake stream
            if self.using_fake_stream:
                self.sensor_count_label.setText(f"0 / {config.num_sensors} - Using Fake Stream")
                self.sensor_count_label.setStyleSheet(
                    "font-size: 11pt; font-weight: bold; font-family: 'Courier New'; color: #9b59b6;"
                )
            else:
                self.sensor_count_label.setText(f"Detecting... (need {config.num_sensors})")
                self.sensor_count_label.setStyleSheet(
                    "font-size: 11pt; font-weight: bold; font-family: 'Courier New'; color: #7f8c8d;"
                )
            
            # Enable calibration and recording buttons
            self.n_pose_btn.setEnabled(True)
            self.n_pose_btn.setToolTip("")
            self.record_btn.setEnabled(True)
            self.record_btn.setToolTip("")
            self.record_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
            
            anthro_text = f"{anthro_profile.mode}" if anthro_profile else "default"
            self.status_label.setText(f"Session configured: {info['subject_id']} - {info['session_id']}")
            self.instruction_label.setText(f"Body: {config.display_name} | Anthro: {anthro_text}")
            print(f"[App] Session configured: {info['subject_id']}/{info['session_id']}")
            print(f"[App] Body configuration: {config.display_name} ({config.num_sensors} sensors)")
            print(f"[App] Active segments: {', '.join(config.required_segments)}")
            print(f"[App] Save directory: {session_folder}")
            print(f"[App] Calibration and recording enabled")
    
    def _update_kinematics_with_anthropometry(self):
        """Update kinematic model with current anthropometry."""
        # Update BODY_MODEL with new segment lengths
        for seg_name, seg_def in BODY_MODEL.items():
            new_length = self.anthro_manager.get_segment_length(seg_name)
            seg_def.length = new_length
        
        # Reinitialize kinematics engine with updated body model
        self.kinematics = KinematicsEngine(BODY_MODEL)
        
        print("[App] Kinematics updated with new segment lengths")
    
    def _save_calibration_info(self):
        """Save calibration parameters to session folder."""
        if not self.calibration.is_static_calibrated:
            print("[App] No calibration to save")
            return
        
        # Gather calibration info
        calib_info = {
            'type': 'dual_pose' if hasattr(self, 't_pose_recording') else 'static',
            'static_calibrated': self.calibration.is_static_calibrated,
            'n_pose_completed': True,
            't_pose_completed': hasattr(self, 't_pose_recording'),
            'anthropometry': {
                'mode': self.anthro_manager.current_profile.mode if self.anthro_manager.current_profile else 'default',
                'height_m': self.anthro_manager.current_profile.height_m if self.anthro_manager.current_profile else 1.75,
                'weight_kg': self.anthro_manager.current_profile.weight_kg if self.anthro_manager.current_profile else 75.0
            }
        }
        
        # Save through recording controller
        self.recording_ctrl.save_calibration_info(calib_info)
        print("[App] Calibration info saved to session folder")
    
    def _update_session_display(self):
        """Update the session info display label."""
        info = self.recording_ctrl.get_session_info()
        if info['subject_id'] and info['session_id']:
            # Get body config info
            config = get_body_config(info['body_config'])
            display_text = f"✓ {info['subject_id']} - {info['session_id']}<br><small>{config.display_name}</small>"
            
            self.session_info_display.setText(display_text)
            self.session_info_display.setStyleSheet(
                "font-size: 9pt; font-family: 'Courier New'; color: #27ae60; font-weight: bold; "
                "background-color: #e8f8f5; padding: 3px; border: 1px solid #27ae60; border-radius: 2px;"
            )
        else:
            self.session_info_display.setText("⚠️ Not configured")
            self.session_info_display.setStyleSheet(
                "font-size: 9pt; font-family: 'Courier New'; color: #e67e22; font-weight: bold; "
                "background-color: #fef5e7; padding: 3px; border: 1px solid #f39c12; border-radius: 2px;"
            )
    
    def _on_camera_toggled(self, camera_id: int, enabled: bool):
        """Handle camera enable/disable from UI."""
        if enabled:
            # Open camera
            success = self.webcam_recorder.open_camera(camera_id, target_fps=60)
            if success:
                print(f"[App] Camera {camera_id} opened")
            else:
                print(f"[App] Failed to open camera {camera_id}")
                # Uncheck the checkbox
                self.camera_grid.camera_checkboxes[camera_id].setChecked(False)
        else:
            # Close camera
            self.webcam_recorder.close_camera(camera_id)
            print(f"[App] Camera {camera_id} closed")
    
    def _add_calibration_event(self, event_name: str):
        """Add a calibration event marker to the recording data."""
        event_row = {
            'time': time.monotonic() - self.start_time,
            'event': event_name
        }
        
        # Fill in all joint data fields with zeros (event marker row)
        for seg_name, seg_def in BODY_MODEL.items():
            if seg_def.parent is None:
                continue
            joint_name = f"{seg_def.parent}_{seg_name}"
            
            # Quaternion fields
            event_row[f"{joint_name}_qw"] = 0
            event_row[f"{joint_name}_qx"] = 0
            event_row[f"{joint_name}_qy"] = 0
            event_row[f"{joint_name}_qz"] = 0
            
            # Euler angle fields
            event_row[f"{joint_name}_euler_0"] = 0
            event_row[f"{joint_name}_euler_1"] = 0
            event_row[f"{joint_name}_euler_2"] = 0
        
        self.recording_ctrl.add_data_sample(event_row)
        print(f"[App] Added '{event_name}' event marker to recording at t={event_row['time']:.2f}s")
    
    # _save_recording method removed - now handled by RecordingController
    
    # ==================== Dual-Pose Calibration Methods ====================
    
    def _on_calib_mode_changed(self, index):
        """Handle calibration mode change."""
        # Reset if switching modes
        if self.calibration.is_static_calibrated:
            # Ask if user wants to reset
            self.status_label.setText("Mode changed - restart calibration if needed")
        
        # Update button visibility based on mode
        if index == 0:  # Dual-pose
            self.n_pose_btn.setEnabled(True)
            self.t_pose_btn.setEnabled(False)
        elif index == 1:  # Single N-pose
            self.n_pose_btn.setEnabled(True)
            self.t_pose_btn.setEnabled(False)
            self.t_pose_btn.setVisible(False)
            self.t_pose_status.setVisible(False)
        else:  # Single T-pose
            self.n_pose_btn.setEnabled(False)
            self.n_pose_btn.setVisible(False)
            self.n_pose_status.setVisible(False)
            self.t_pose_btn.setEnabled(True)
            self.t_pose_btn.setVisible(True)
            self.t_pose_status.setVisible(True)
    
    def _on_n_pose_calibration(self):
        """Start N-pose capture countdown."""
        if hasattr(self, 'n_pose_pending') and self.n_pose_pending:
            return
        
        # Reset calibration before starting new one
        print("[App] Resetting calibration before new N-pose capture...")
        
        # If currently recording, add a re-calibration marker
        if self.is_recording:
            recalib_marker = {
                'time': time.monotonic() - self.start_time,
                'event': 'RE_CALIBRATION_START',
                'calibration_type': 'n_pose' if self.calib_mode_combo.currentIndex() == 1 else 'dual_pose'
            }
            # Add marker with all joint angles as NaN
            for seg_name, seg_def in BODY_MODEL.items():
                if seg_def.parent is None:
                    continue
                joint_name = f"{seg_def.parent}_{seg_name}"
                for i in range(4):  # quaternion
                    recalib_marker[f"{joint_name}_q{i}"] = np.nan
                for i in range(3):  # euler angles
                    recalib_marker[f"{joint_name}_euler_{i}"] = np.nan
            self.record_data.append(recalib_marker)
            print("[App] Re-calibration marker added to recording")
        
        self.dual_pose_calib.reset()
        if self.calibration.is_static_calibrated:
            # Clear old calibration data
            self.calibration.alignments = {}
            self.calibration.offsets = {}
            self.calibration.is_static_calibrated = False
        
        # Reset ALL calibration state flags
        self.n_pose_pending = False
        if hasattr(self, 'n_pose_recording_active'):
            del self.n_pose_recording_active
        self.t_pose_pending = False
        if hasattr(self, 't_pose_recording_active'):
            del self.t_pose_recording_active
        
        # Reset UI status
        self.n_pose_status.setText("Not captured")
        self.n_pose_status.setStyleSheet("color: #888;")
        self.t_pose_status.setText("Not captured")
        self.t_pose_status.setStyleSheet("color: #888;")
        
        self.n_pose_pending = True
        self.n_pose_start_time = time.monotonic()
        self.n_pose_recording = []  # List to store 5 seconds of samples
        
        # Check calibration mode
        mode = self.calib_mode_combo.currentIndex()
        
        if mode == 1:  # Single N-Pose mode
            self.status_label.setText("N-Pose calibration starting in 5 seconds")
            self.instruction_label.setText(
                "Get into N-POSE: Stand with feet shoulder-width apart, "
                "arms hanging naturally at sides, palms facing inward, looking forward. "
                "HOLD STILL for 5 seconds after countdown."
            )
        else:  # Dual-pose mode
            self.status_label.setText("N-Pose calibration starting in 5 seconds (1/2)")
            self.instruction_label.setText(
                "Get into N-POSE: Stand with feet shoulder-width apart, "
                "arms hanging naturally at sides, palms facing inward, looking forward. "
                "HOLD STILL for 5 seconds after countdown."
            )
        
        self.n_pose_status.setText("Waiting...")
        print("[App] N-pose calibration countdown started")
    
    def _on_t_pose_calibration(self):
        """Start T-pose capture countdown."""
        if hasattr(self, 't_pose_pending') and self.t_pose_pending:
            return
        
        # Check calibration mode
        mode = self.calib_mode_combo.currentIndex()
        
        # For single T-pose mode, reset calibration before starting
        if mode == 2:  # Single T-Pose mode
            print("[App] Resetting calibration before new T-pose capture...")
            self.dual_pose_calib.reset()
            if self.calibration.is_static_calibrated:
                # Clear old calibration data
                self.calibration.alignments = {}
                self.calibration.offsets = {}
                self.calibration.is_static_calibrated = False
            
            # Reset ALL calibration state flags
            self.n_pose_pending = False
            if hasattr(self, 'n_pose_recording_active'):
                del self.n_pose_recording_active
            self.t_pose_pending = False
            if hasattr(self, 't_pose_recording_active'):
                del self.t_pose_recording_active
            
            # Reset UI status
            self.n_pose_status.setText("Not captured")
            self.n_pose_status.setStyleSheet("color: #888;")
            self.t_pose_status.setText("Not captured")
            self.t_pose_status.setStyleSheet("color: #888;")
        
        self.t_pose_pending = True
        self.t_pose_start_time = time.monotonic()
        self.t_pose_recording = []  # List to store 5 seconds of samples
        
        if mode == 2:  # Single T-Pose mode
            self.status_label.setText("T-Pose calibration starting in 5 seconds")
        else:  # Dual-pose mode
            self.status_label.setText("T-Pose calibration starting in 5 seconds (2/2)")
        
        self.instruction_label.setText(
            "Get into T-POSE: Feet shoulder-width apart, "
            "arms extended straight out to sides (90° from body), palms facing down, looking forward. "
            "HOLD STILL for 5 seconds after countdown."
        )
        self.t_pose_status.setText("Waiting...")
        print("[App] T-pose calibration countdown started")
    
    def _capture_n_pose(self):
        """Start 5-second N-pose recording."""
        print("[App] Starting N-pose 5-second recording...")
        self.n_pose_recording = []
        self.n_pose_recording_start = time.monotonic()
        self.n_pose_recording_active = True
        self.n_pose_status.setText("Recording... 5s")
    
    def _finalize_n_pose(self):
        """Finalize N-pose recording and save to CSV."""
        self.n_pose_recording_active = False
        
        if len(self.n_pose_recording) == 0:
            self.status_label.setText("Error: No N-pose data recorded")
            self.n_pose_status.setText("Failed")
            self.n_pose_pending = False
            return
        
        # Save using recording controller (ensures session folder)
        filepath = self.recording_ctrl.save_calibration_data("Npose", self.n_pose_recording)
        
        if filepath:
            print(f"[App] N-pose data saved: {len(self.n_pose_recording)} samples to {filepath}")
        
        # Use median sample for calibration (more robust than single sample)
        median_idx = len(self.n_pose_recording) // 2
        median_sample = self.n_pose_recording[median_idx]
        
        # Convert back to rotations (use world_ prefix now)
        world_rots = {}
        for seg_name in SEGMENT_NAMES:
            qw = median_sample[f"world_{seg_name}_qw"]
            qx = median_sample[f"world_{seg_name}_qx"]
            qy = median_sample[f"world_{seg_name}_qy"]
            qz = median_sample[f"world_{seg_name}_qz"]
            world_rots[seg_name] = R.from_quat([qx, qy, qz, qw])
        
        # Check calibration mode
        mode = self.calib_mode_combo.currentIndex()
        
        if mode == 1:  # Single N-Pose mode
            # Perform single-pose calibration with continuous optimization
            print("[App] Performing single N-pose calibration with continuous optimization...")
            self._perform_single_pose_calibration(world_rots, pose_type='n_pose')
            
            # Add calibration event to recording if active
            if self.is_recording:
                self._add_calibration_event('N_POSE_RECALIBRATION')
            
            self.n_pose_pending = False
            self.n_pose_status.setText("✓ Complete")
            self.n_pose_status.setStyleSheet("color: green;")
            self.status_label.setText(f"N-pose calibration complete! System ready.")
            self.instruction_label.setText("Calibration complete! Click 'Capture N-Pose' again to re-calibrate.")
            
            # Enable recording and keep N-pose button active for re-calibration
            self.record_btn.setEnabled(True)
            self.n_pose_btn.setEnabled(True)
            
        else:  # Dual-pose mode
            # Capture N-pose for dual calibration
            self.dual_pose_calib.capture_n_pose(world_rots)
            
            self.n_pose_pending = False
            self.n_pose_status.setText("✓ Captured (5s)")
            self.n_pose_status.setStyleSheet("color: green;")
            self.status_label.setText(f"N-pose captured! {len(self.n_pose_recording)} samples saved. Now capture T-pose.")
            self.instruction_label.setText("Click '2. Capture T-Pose' when ready")
            
            # Enable T-pose button
            self.t_pose_btn.setEnabled(True)
            self.n_pose_btn.setEnabled(False)
        
        print(f"[App] N-pose captured successfully (median of {len(self.n_pose_recording)} samples)")
    
    def _capture_t_pose(self):
        """Start 5-second T-pose recording."""
        print("[App] Starting T-pose 5-second recording...")
        self.t_pose_recording = []
        self.t_pose_recording_start = time.monotonic()
        self.t_pose_recording_active = True
        self.t_pose_status.setText("Recording... 5s")
    
    def _finalize_t_pose(self):
        """Finalize T-pose recording and save to CSV."""
        self.t_pose_recording_active = False
        
        if len(self.t_pose_recording) == 0:
            self.status_label.setText("Error: No T-pose data recorded")
            self.t_pose_status.setText("Failed")
            self.t_pose_pending = False
            return
        
        # Save using recording controller (ensures session folder)
        filepath = self.recording_ctrl.save_calibration_data("Tpose", self.t_pose_recording)
        
        if filepath:
            print(f"[App] T-pose data saved: {len(self.t_pose_recording)} samples to {filepath}")
        
        # Use median sample for calibration (more robust than single sample)
        median_idx = len(self.t_pose_recording) // 2
        median_sample = self.t_pose_recording[median_idx]
        
        # Convert back to rotations (use world_ prefix now)
        world_rots = {}
        for seg_name in SEGMENT_NAMES:
            qw = median_sample[f"world_{seg_name}_qw"]
            qx = median_sample[f"world_{seg_name}_qx"]
            qy = median_sample[f"world_{seg_name}_qy"]
            qz = median_sample[f"world_{seg_name}_qz"]
            world_rots[seg_name] = R.from_quat([qx, qy, qz, qw])
        
        # Check calibration mode
        mode = self.calib_mode_combo.currentIndex()
        
        if mode == 2:  # Single T-Pose mode
            # Perform single-pose calibration with continuous optimization
            print("[App] Performing single T-pose calibration with continuous optimization...")
            self._perform_single_pose_calibration(world_rots, pose_type='t_pose')
            
            # Add calibration event to recording if active
            if self.is_recording:
                self._add_calibration_event('T_POSE_RECALIBRATION')
            
            self.t_pose_pending = False
            self.t_pose_status.setText("✓ Complete")
            self.t_pose_status.setStyleSheet("color: green;")
            self.status_label.setText(f"T-pose calibration complete! System ready.")
            self.instruction_label.setText("Calibration complete! Click 'Capture T-Pose' again to re-calibrate.")
            
            # Enable recording and keep T-pose button active for re-calibration
            self.record_btn.setEnabled(True)
            self.t_pose_btn.setEnabled(True)
            
        else:  # Dual-pose mode
            # Capture T-pose for dual calibration
            self.dual_pose_calib.capture_t_pose(world_rots)
            
            self.t_pose_pending = False
            self.t_pose_status.setText("✓ Captured (5s)")
            self.t_pose_status.setStyleSheet("color: green;")
            
            print(f"[App] T-pose captured (median of {len(self.t_pose_recording)} samples), computing dual-pose calibration...")
            
            # Compute calibration from both poses
            self._compute_dual_pose_calibration()
    
    def _compute_dual_pose_calibration(self):
        """Compute calibration from N-pose and T-pose."""
        try:
            # Compute alignments and offsets
            alignments, offsets = self.dual_pose_calib.compute_calibration()
            
            # Apply to calibration manager
            for seg, alignment in alignments.items():
                self.calibration.alignments[seg] = alignment
            
            for seg, offset in offsets.items():
                self.calibration.offsets[seg] = offset
            
            # Mark as calibrated
            self.calibration.is_static_calibrated = True
            
            # Add calibration event to recording if active
            if self.is_recording:
                self._add_calibration_event('DUAL_POSE_RECALIBRATION')
            
            # Update UI
            self.status_label.setText("Dual-pose calibration complete! System ready.")
            self.instruction_label.setText(
                "Calibration successful! Click 'Capture N-Pose' to start over, or begin recording."
            )
            
            # Enable recording and keep both buttons active for re-calibration
            self.record_btn.setEnabled(True)
            self.n_pose_btn.setEnabled(True)
            self.t_pose_btn.setEnabled(False)  # Will be enabled after capturing N-pose again
            
            print("[App] Dual-pose calibration complete and applied!")
            
        except Exception as e:
            self.status_label.setText(f"Calibration failed: {str(e)}")
            print(f"[App] Dual-pose calibration error: {e}")
    
    def _perform_single_pose_calibration(self, world_rots: Dict[str, R], pose_type: str):
        """
        Perform single-pose calibration with continuous offset optimization.
        
        Args:
            world_rots: Sensor orientations in world frame
            pose_type: 'n_pose' or 't_pose'
        """
        from scipy.optimize import minimize
        
        print(f"[SinglePoseCalib] Starting {pose_type} calibration with continuous optimization...")
        
        # Convert to pelvis-relative frame
        pelvis = world_rots['pelvis']
        pelvis_relative = {}
        for seg, rot in world_rots.items():
            if seg == 'pelvis':
                pelvis_relative[seg] = R.identity()
            else:
                pelvis_relative[seg] = pelvis.inv() * rot
        
        # Expected angles for this pose
        if pose_type == 'n_pose':
            expected_angles = {seg: np.array([0, 0, 0]) for seg in SEGMENT_NAMES if seg != 'pelvis'}
        else:  # t_pose
            expected_angles = {}
            for seg in SEGMENT_NAMES:
                if seg == 'pelvis':
                    continue
                elif 'upper_arm' in seg:
                    expected_angles[seg] = np.array([0, 90, 0])  # 90° abduction
                else:
                    expected_angles[seg] = np.array([0, 0, 0])
        
        # Optimize offsets for each segment
        for seg_name in SEGMENT_NAMES:
            if seg_name == 'pelvis':
                self.calibration.offsets[seg_name] = pelvis.inv()
                self.calibration.alignments[seg_name] = R.identity()
                continue
            
            expected = expected_angles[seg_name]
            sensor_rel = pelvis_relative[seg_name]
            
            # Objective: find offset that gives expected angles
            def objective(rotvec):
                offset = R.from_rotvec(rotvec)
                calibrated = offset * sensor_rel
                angles = calibrated.as_euler('ZXY', degrees=True)
                error = np.linalg.norm(angles - expected)
                return error
            
            # Initial guess: identity
            x0 = R.identity().as_rotvec()
            
            # Optimize
            result = minimize(objective, x0, method='Powell', 
                            options={'maxiter': 100, 'ftol': 0.01})
            
            optimal_offset = R.from_rotvec(result.x)
            
            # Store offset and identity alignment
            self.calibration.offsets[seg_name] = optimal_offset
            self.calibration.alignments[seg_name] = R.identity()
            
            print(f"[SinglePoseCalib] {seg_name}: error={result.fun:.2f}°")
        
        # Mark as calibrated
        self.calibration.is_static_calibrated = True
        
        print(f"[SinglePoseCalib] {pose_type} calibration complete!")
    
    # ==================== Functional Calibration V2 Methods ====================
    
    def _on_functional_calibration_v2(self):
        """Start functional calibration v2 sequence."""
        print("[App] Starting functional calibration v2")
        self.func_calib_v2.reset()
        self.func_calib_btn.setEnabled(False)
        self.restart_func_btn.setEnabled(False)
        self.func_status_label.setText("In progress...")
        
        # Start first movement
        self._start_next_func_calib_movement()
    
    def _on_restart_functional_calibration_v2(self):
        """Restart functional calibration from beginning."""
        self._on_functional_calibration_v2()
    
    def _start_next_func_calib_movement(self):
        """Start the next movement in the functional calibration sequence."""
        movement = self.func_calib_v2.get_current_movement()
        
        if not movement:
            # All movements complete!
            self._on_func_calib_v2_complete()
            return
        
        # Show movement instruction
        movement_num = self.func_calib_v2.current_movement_index + 1
        total_movements = len(self.func_calib_v2.movements)
        
        self.movement_instruction_label.setText(
            f"MOVEMENT {movement_num}/{total_movements}: {movement.description}\n\n{movement.instruction}"
        )
        self.movement_instruction_label.setVisible(True)
        
        self.movement_progress_label.setText(
            f"Progress: {movement_num-1}/{total_movements} complete"
        )
        self.movement_progress_label.setVisible(True)
        
        self.func_status_label.setText(f"Get ready for: {movement.description}")
        
        # Countdown before starting
        self._func_calib_countdown = 3
        self._func_calib_countdown_timer()
    
    def _func_calib_countdown_timer(self):
        """Countdown timer before starting movement."""
        if self._func_calib_countdown > 0:
            self.status_label.setText(f"Get ready... {self._func_calib_countdown}")
            self._func_calib_countdown -= 1
            QtCore.QTimer.singleShot(1000, self._func_calib_countdown_timer)
        else:
            # Start recording
            self._start_func_calib_recording()
    
    def _start_func_calib_recording(self):
        """Start recording samples for current movement."""
        movement = self.func_calib_v2.get_current_movement()
        if not movement:
            return
        
        self.func_calib_v2.start_recording()
        self.status_label.setText(f"GO! Perform: {movement.description}")
        
        # Stop after duration
        duration_ms = int(movement.duration * 1000)
        QtCore.QTimer.singleShot(duration_ms, self._stop_func_calib_recording)
    
    def _stop_func_calib_recording(self):
        """Stop recording and process current movement."""
        success, message = self.func_calib_v2.stop_recording()
        
        if success:
            # Movement succeeded
            self.status_label.setText(f"✓ {message}")
            
            # Update progress
            completed = self.func_calib_v2.current_movement_index
            total = len(self.func_calib_v2.movements)
            self.movement_progress_label.setText(f"Progress: {completed}/{total} complete")
            
            # Pause before next movement
            movement = self.func_calib_v2.movements[completed - 1]
            pause_ms = int(movement.pause_after * 1000)
            QtCore.QTimer.singleShot(pause_ms, self._start_next_func_calib_movement)
        else:
            # Movement failed - offer to repeat
            self.status_label.setText(f"✗ {message}")
            self.movement_instruction_label.setText(
                f"Movement quality was insufficient.\n\n{message}\n\nClick 'Repeat Movement' to try again."
            )
            
            # Show repeat button
            self._show_repeat_movement_dialog(message)
    
    def _show_repeat_movement_dialog(self, error_msg):
        """Show dialog to repeat failed movement."""
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Warning)
        msg_box.setWindowTitle("Movement Quality Issue")
        msg_box.setText(f"The movement did not meet quality requirements:\n\n{error_msg}")
        msg_box.setInformativeText("Would you like to repeat this movement?")
        
        repeat_btn = msg_box.addButton("Repeat Movement", QtWidgets.QMessageBox.AcceptRole)
        skip_btn = msg_box.addButton("Skip (Use Anyway)", QtWidgets.QMessageBox.RejectRole)
        cancel_btn = msg_box.addButton("Cancel Calibration", QtWidgets.QMessageBox.DestructiveRole)
        
        msg_box.exec_()
        
        if msg_box.clickedButton() == repeat_btn:
            # Repeat the same movement
            self._start_next_func_calib_movement()
        elif msg_box.clickedButton() == skip_btn:
            # Accept poor quality and move on
            self.func_calib_v2.current_movement_index += 1
            self._start_next_func_calib_movement()
        else:
            # Cancel calibration
            self._cancel_func_calib_v2()
    
    def _cancel_func_calib_v2(self):
        """Cancel functional calibration."""
        self.func_calib_v2.reset()
        self.movement_instruction_label.setVisible(False)
        self.movement_progress_label.setVisible(False)
        self.func_calib_btn.setEnabled(True)
        self.restart_func_btn.setEnabled(False)
        self.func_status_label.setText("Cancelled")
        self.status_label.setText("Functional calibration cancelled")
    
    def _on_func_calib_v2_complete(self):
        """Functional calibration v2 sequence complete."""
        print("[App] Functional calibration v2 complete!")
        
        # Apply computed alignments to calibration manager
        for seg, alignment in self.func_calib_v2.alignments.items():
            self.calibration.alignments[seg] = alignment
        
        # Hide movement UI
        self.movement_instruction_label.setVisible(False)
        self.movement_progress_label.setVisible(False)
        
        # Update status
        self.func_status_label.setText("✓ Complete")
        self.status_label.setText("Functional calibration complete!")
        self.instruction_label.setText(
            "Functional calibration successful! Now perform final static calibration (3. Final Static) "
            "to apply the computed alignments."
        )
        
        # Enable post-static calibration
        self.post_static_calib_btn.setEnabled(True)
        self.post_static_calib_btn.setVisible(True)
        
        # Enable restart button
        self.restart_func_btn.setEnabled(True)
        self.func_calib_btn.setEnabled(False)
        
        print("[App] Alignments applied. Proceed to post-static calibration.")
    
    # ==================== End Functional Calibration V2 ====================
    
    def closeEvent(self, event):
        """Clean shutdown"""
        if hasattr(self.imu_stream, 'stop'):
            self.imu_stream.stop()
        super().closeEvent(event)


def main():
    """Main application entry point."""
    app = QtWidgets.QApplication(sys.argv)
    
    # Show welcome/setup screen
    welcome = WelcomeDialog()
    result = welcome.exec_()
    
    if result != QtWidgets.QDialog.Accepted:
        print("[App] Setup cancelled by user")
        sys.exit(0)
    
    # Get user selections
    selected_channel = welcome.get_channel()
    
    print(f"[App] Starting with radio channel: {selected_channel}")
    print(f"[App] Body configuration will be set in session setup")
    
    # Launch main application with configuration
    window = BiomechApp(radio_channel=selected_channel)
    # Show window maximized
    window.showMaximized()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
