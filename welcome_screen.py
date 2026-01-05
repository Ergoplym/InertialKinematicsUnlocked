"""
Welcome screen for BushBiomech IMU Motion Capture System.
Provides setup instructions and channel selection before launching main app.
"""

from PyQt5 import QtWidgets, QtCore, QtGui


class WelcomeDialog(QtWidgets.QDialog):
    """Welcome and setup dialog shown before main application."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IKU - Sensor Setup")
        self.setModal(True)
        
        # Get screen size and set dialog to fit
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        dialog_width = min(700, int(screen.width() * 0.5))
        dialog_height = min(700, int(screen.height() * 0.85))  # 85% of screen height
        self.resize(dialog_width, dialog_height)
        
        # Center on screen
        self.move(
            (screen.width() - dialog_width) // 2,
            (screen.height() - dialog_height) // 2
        )
        
        # User selections
        self.selected_channel = 11
        self.setup_accepted = False
        
        self._init_ui()
    
    def _init_ui(self):
        """Build the UI with scrolling."""
        # Main layout for dialog
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for content
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # Container widget for scroll area
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setSpacing(15)
        
        # Logo at top
        logo_label = QtWidgets.QLabel()
        logo_pixmap = QtGui.QPixmap("IKU_logo.png")
        if not logo_pixmap.isNull():
            # Scale logo to fit width while maintaining aspect ratio
            # Use smaller width for smaller screens
            max_logo_width = min(600, self.width() - 40)
            scaled_logo = logo_pixmap.scaledToWidth(max_logo_width, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
            logo_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(logo_label)
        
        # Title
        title = QtWidgets.QLabel("Inertial Kinematics Unlocked")
        title.setStyleSheet("font-size: 20pt; font-weight: bold; color: #2c3e50;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(10)
        
        # Instructions group
        instructions_group = QtWidgets.QGroupBox("Setup Instructions")
        instructions_group.setStyleSheet("""
            QGroupBox {
                font-size: 11pt;
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        instructions_layout = QtWidgets.QVBoxLayout()
        instructions_layout.setSpacing(10)
        
        # Step-by-step instructions
        steps = [
            ("1. Open MT Manager", 
             "Launch Xsens MT Manager software"),
            
            ("2. Select Radio Channel", 
             "In MT Manager, ensure <b>Channel 11</b> is selected (or choose another below)"),
            
            ("3. Pair Sensors", 
             "Click 'Wireless Configuration' in MT Manager, select 60Hz and 'Enable Wireless Master' wait for all IMUs to connect, then click 'Start Measurement'.<br>"
             "You should see all 15 sensors listed as connected."),
            
            ("4. Close MT Manager", 
             "Once all sensors are paired and connected, <b>close MT Manager completely</b>."),
            
            ("5. Start Application", 
             "Click <b>'Next'</b> below to start BushBiomech and begin streaming data.")
        ]
        
        for step_title, step_desc in steps:
            # Step container
            step_widget = QtWidgets.QWidget()
            step_layout = QtWidgets.QVBoxLayout(step_widget)
            step_layout.setContentsMargins(10, 5, 10, 5)
            step_layout.setSpacing(3)
            
            # Step title
            title_label = QtWidgets.QLabel(step_title)
            title_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #2980b9;")
            step_layout.addWidget(title_label)
            
            # Step description
            desc_label = QtWidgets.QLabel(step_desc)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("font-size: 9pt; color: #34495e; padding-left: 15px;")
            step_layout.addWidget(desc_label)
            
            instructions_layout.addWidget(step_widget)
        
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)
        
        # Channel selection
        channel_group = QtWidgets.QGroupBox("Radio Channel")
        channel_group.setStyleSheet("""
            QGroupBox {
                font-size: 10pt;
                font-weight: bold;
                border: 2px solid #95a5a6;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        channel_layout = QtWidgets.QHBoxLayout()
        
        channel_label = QtWidgets.QLabel("Awinda Channel (11-25):")
        channel_label.setStyleSheet("font-size: 9pt;")
        channel_layout.addWidget(channel_label)
        
        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setRange(11, 25)
        self.channel_spin.setValue(11)
        self.channel_spin.setStyleSheet("font-size: 10pt; padding: 5px;")
        self.channel_spin.setToolTip("Select the radio channel (must match MT Manager setting)")
        self.channel_spin.valueChanged.connect(self._on_channel_changed)
        channel_layout.addWidget(self.channel_spin)
        
        channel_info = QtWidgets.QLabel("ℹ️ Default is Channel 11")
        channel_info.setStyleSheet("font-size: 8pt; color: #7f8c8d; font-style: italic;")
        channel_layout.addWidget(channel_info)
        
        channel_layout.addStretch()
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Important note
        note_widget = QtWidgets.QFrame()
        note_widget.setStyleSheet("""
            QFrame {
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        note_layout = QtWidgets.QHBoxLayout(note_widget)
        
        note_icon = QtWidgets.QLabel("⚠️")
        note_icon.setStyleSheet("font-size: 16pt;")
        note_layout.addWidget(note_icon)
        
        note_text = QtWidgets.QLabel(
            "<b>Important:</b> Make sure MT Manager is completely closed before clicking 'Next'. "
            "Only one application can connect to the Awinda master at a time."
        )
        note_text.setWordWrap(True)
        note_text.setStyleSheet("font-size: 9pt; color: #856404;")
        note_layout.addWidget(note_text)
        
        layout.addWidget(note_widget)
        
        layout.addStretch()
        
        # Set container widget for scroll area
        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        
        # Buttons at bottom (outside scroll area, always visible)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        button_layout.addStretch()
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setFixedSize(120, 40)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        next_btn = QtWidgets.QPushButton("Next ➜")
        next_btn.setFixedSize(120, 40)
        next_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 11pt;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        next_btn.clicked.connect(self._on_next)
        button_layout.addWidget(next_btn)
        
        main_layout.addLayout(button_layout)
    
    def _on_channel_changed(self, value):
        """Update selected channel."""
        self.selected_channel = value
        print(f"[Welcome] Channel selected: {value}")
    
    def _on_next(self):
        """User clicked Next - proceed to main application."""
        self.setup_accepted = True
        print(f"[Welcome] Setup complete. Channel: {self.selected_channel}")
        self.accept()
    
    def get_channel(self):
        """Get the selected radio channel."""
        return self.selected_channel
    
    def was_accepted(self):
        """Check if user completed setup (vs cancelled)."""
        return self.setup_accepted


# Demo/test function
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    dialog = WelcomeDialog()
    result = dialog.exec_()
    
    if result == QtWidgets.QDialog.Accepted:
        print(f"Setup accepted! Selected channel: {dialog.get_channel()}")
    else:
        print("Setup cancelled by user")
    
    sys.exit(0)
