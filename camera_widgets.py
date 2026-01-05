"""
Camera preview widgets for BushBiomech UI.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui


class CameraPreviewWidget(QtWidgets.QLabel):
    """Widget to display camera preview."""
    
    def __init__(self, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.preview_size = (320, 240)  # Default size
        
        # Don't set fixed size - let parent control it
        self.setMinimumSize(320, 240)
        self.setScaledContents(True)
        
        # Styling
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #3498db;
                border-radius: 5px;
                background-color: #2c3e50;
            }
        """)
        
        # Default "no signal" image
        self.set_no_signal()
    
    def set_no_signal(self):
        """Display 'No Signal' placeholder."""
        # Use actual widget size
        w = max(self.width(), 320)
        h = max(self.height(), 240)
        
        pixmap = QtGui.QPixmap(w, h)
        pixmap.fill(QtGui.QColor("#2c3e50"))
        
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QColor("#95a5a6"))
        painter.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        painter.drawText(
            pixmap.rect(),
            QtCore.Qt.AlignCenter,
            f"Camera {self.camera_id}\nNo Signal"
        )
        painter.end()
        
        self.setPixmap(pixmap)
    
    def update_frame(self, frame: np.ndarray):
        """
        Update preview with new frame.
        
        Args:
            frame: OpenCV frame (BGR format)
        """
        if frame is None:
            self.set_no_signal()
            return
        
        try:
            # Convert BGR to RGB
            rgb_frame = frame[:, :, ::-1].copy()
            
            # Resize to widget size
            target_w = max(self.width(), 320)
            target_h = max(self.height(), 240)
            
            h, w = rgb_frame.shape[:2]
            
            # Calculate aspect ratio preserving resize
            aspect = w / h
            target_aspect = target_w / target_h
            
            if aspect > target_aspect:
                # Width limited
                new_w = target_w
                new_h = int(target_w / aspect)
            else:
                # Height limited
                new_h = target_h
                new_w = int(target_h * aspect)
            
            import cv2
            resized = cv2.resize(rgb_frame, (new_w, new_h))
            
            # Convert to QImage
            height, width, channels = resized.shape
            bytes_per_line = channels * width
            q_image = QtGui.QImage(
                resized.data,
                width,
                height,
                bytes_per_line,
                QtGui.QImage.Format_RGB888
            )
            
            # Convert to QPixmap and display
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.setPixmap(pixmap)
            
        except Exception as e:
            print(f"[CameraPreview] Error updating frame: {e}")


class CameraGridWidget(QtWidgets.QWidget):
    """Grid of camera previews with controls."""
    
    def __init__(self, max_cameras: int = 4, parent=None):
        super().__init__(parent)
        self.max_cameras = max_cameras
        self.camera_previews = {}
        self.camera_checkboxes = {}
        
        self._init_ui()
    
    def _init_ui(self):
        """Build camera grid UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QtWidgets.QLabel("📹 Camera Feeds")
        title.setStyleSheet("font-size: 11pt; font-weight: bold; padding: 5px;")
        title.setAlignment(QtCore.Qt.AlignLeft)
        layout.addWidget(title)
        
        # Camera grid - horizontal if max_cameras <= 3, else 2x2
        grid_widget = QtWidgets.QWidget()
        
        if self.max_cameras <= 3:
            # Horizontal layout for 3 cameras
            grid_layout = QtWidgets.QHBoxLayout(grid_widget)
            grid_layout.setSpacing(10)
            
            for i in range(self.max_cameras):
                # Container for camera + checkbox
                cam_container = QtWidgets.QWidget()
                cam_layout = QtWidgets.QVBoxLayout(cam_container)
                cam_layout.setSpacing(3)
                cam_layout.setContentsMargins(0, 0, 0, 0)
                
                # Preview widget
                preview = CameraPreviewWidget(camera_id=i)
                self.camera_previews[i] = preview
                cam_layout.addWidget(preview)
                
                # Enable checkbox
                checkbox = QtWidgets.QCheckBox(f"Enable Camera {i}")
                checkbox.setStyleSheet("font-size: 9pt;")
                checkbox.stateChanged.connect(
                    lambda state, cam_id=i: self._on_camera_toggled(cam_id, state)
                )
                self.camera_checkboxes[i] = checkbox
                cam_layout.addWidget(checkbox)
                
                # Add to horizontal layout
                grid_layout.addWidget(cam_container)
        else:
            # 2x2 grid for 4 cameras
            grid_layout = QtWidgets.QGridLayout(grid_widget)
            grid_layout.setSpacing(10)
            
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
            
            for i in range(self.max_cameras):
                # Container for camera + checkbox
                cam_container = QtWidgets.QWidget()
                cam_layout = QtWidgets.QVBoxLayout(cam_container)
                cam_layout.setSpacing(3)
                cam_layout.setContentsMargins(0, 0, 0, 0)
                
                # Preview widget
                preview = CameraPreviewWidget(camera_id=i)
                self.camera_previews[i] = preview
                cam_layout.addWidget(preview)
                
                # Enable checkbox
                checkbox = QtWidgets.QCheckBox(f"Enable Camera {i}")
                checkbox.setStyleSheet("font-size: 9pt;")
                checkbox.stateChanged.connect(
                    lambda state, cam_id=i: self._on_camera_toggled(cam_id, state)
                )
                self.camera_checkboxes[i] = checkbox
                cam_layout.addWidget(checkbox)
                
                # Add to grid
                row, col = positions[i]
                grid_layout.addWidget(cam_container, row, col)
        
        layout.addWidget(grid_widget)
        
        # Recording indicator
        self.recording_indicator = QtWidgets.QLabel("⚫ Not Recording")
        self.recording_indicator.setStyleSheet(
            "font-size: 9pt; color: #7f8c8d; padding: 5px; "
            "background-color: #ecf0f1; border-radius: 3px;"
        )
        self.recording_indicator.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.recording_indicator)
    
    def _on_camera_toggled(self, camera_id: int, state: int):
        """Handle camera enable/disable."""
        enabled = (state == QtCore.Qt.Checked)
        print(f"[CameraGrid] Camera {camera_id} {'enabled' if enabled else 'disabled'}")
        
        # Emit signal for main app to handle
        if hasattr(self, 'camera_toggled'):
            self.camera_toggled(camera_id, enabled)
    
    def update_preview(self, camera_id: int, frame: np.ndarray):
        """Update a camera preview."""
        if camera_id in self.camera_previews:
            self.camera_previews[camera_id].update_frame(frame)
    
    def set_recording_state(self, is_recording: bool):
        """Update recording indicator."""
        if is_recording:
            self.recording_indicator.setText("🔴 Recording Video")
            self.recording_indicator.setStyleSheet(
                "font-size: 9pt; color: white; padding: 5px; "
                "background-color: #e74c3c; border-radius: 3px; font-weight: bold;"
            )
        else:
            self.recording_indicator.setText("⚫ Not Recording")
            self.recording_indicator.setStyleSheet(
                "font-size: 9pt; color: #7f8c8d; padding: 5px; "
                "background-color: #ecf0f1; border-radius: 3px;"
            )
    
    def set_camera_available(self, camera_id: int, available: bool):
        """Enable/disable camera checkbox based on availability."""
        if camera_id in self.camera_checkboxes:
            checkbox = self.camera_checkboxes[camera_id]
            checkbox.setEnabled(available)
            if not available:
                checkbox.setChecked(False)
                checkbox.setText(f"Camera {camera_id} (Not Available)")
            else:
                checkbox.setText(f"Enable Camera {camera_id}")
