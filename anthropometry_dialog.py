"""
Anthropometry Configuration Dialog for IKU.
Provides UI for simple (height/weight) and advanced (detailed measurements) modes.
"""

import os
from PyQt5 import QtWidgets, QtCore, QtGui
from anthropometry import AnthropometryProfile, AnthropometryManager


class AnthropometryDialog(QtWidgets.QDialog):
    """Dialog for configuring subject anthropometry."""
    
    def __init__(self, current_profile: AnthropometryProfile = None, parent=None):
        super().__init__(parent)
        
        self.manager = AnthropometryManager()
        self.profile = current_profile or AnthropometryProfile()
        self.modified = False
        
        self.setWindowTitle("Anthropometry Configuration")
        self.setModal(True)
        self.resize(600, 700)
        
        self._init_ui()
        self._load_profile_to_ui()
    
    def _init_ui(self):
        """Build the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QtWidgets.QLabel("Subject Anthropometry")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        # Profile name and load/save
        profile_group = QtWidgets.QGroupBox("Profile")
        profile_layout = QtWidgets.QVBoxLayout()
        
        name_layout = QtWidgets.QHBoxLayout()
        name_label = QtWidgets.QLabel("Profile Name:")
        name_label.setStyleSheet("font-size: 10pt;")
        name_layout.addWidget(name_label)
        
        self.name_input = QtWidgets.QLineEdit(self.profile.name)
        self.name_input.setPlaceholderText("e.g., John Doe, Subject001")
        self.name_input.textChanged.connect(self._on_modified)
        name_layout.addWidget(self.name_input, 1)
        profile_layout.addLayout(name_layout)
        
        # Load/Save buttons
        file_layout = QtWidgets.QHBoxLayout()
        
        load_btn = QtWidgets.QPushButton("📂 Load Profile")
        load_btn.clicked.connect(self._load_profile)
        file_layout.addWidget(load_btn)
        
        save_btn = QtWidgets.QPushButton("💾 Save Profile")
        save_btn.clicked.connect(self._save_profile)
        file_layout.addWidget(save_btn)
        
        file_layout.addStretch()
        profile_layout.addLayout(file_layout)
        
        profile_group.setLayout(profile_layout)
        layout.addWidget(profile_group)
        
        # Mode selection
        mode_group = QtWidgets.QGroupBox("Measurement Mode")
        mode_layout = QtWidgets.QVBoxLayout()
        
        self.simple_radio = QtWidgets.QRadioButton("Simple (Height & Weight)")
        self.simple_radio.setStyleSheet("font-size: 10pt; font-weight: bold;")
        self.simple_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.simple_radio)
        
        simple_desc = QtWidgets.QLabel("   Quick setup using standard anthropometric ratios")
        simple_desc.setStyleSheet("font-size: 9pt; color: #666; margin-left: 20px;")
        mode_layout.addWidget(simple_desc)
        
        self.advanced_radio = QtWidgets.QRadioButton("Advanced (Detailed Measurements)")
        self.advanced_radio.setStyleSheet("font-size: 10pt; font-weight: bold;")
        self.advanced_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.advanced_radio)
        
        advanced_desc = QtWidgets.QLabel("   Precise measurements for accurate skeleton scaling")
        advanced_desc.setStyleSheet("font-size: 9pt; color: #666; margin-left: 20px;")
        mode_layout.addWidget(advanced_desc)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Stacked widget for simple/advanced inputs
        self.input_stack = QtWidgets.QStackedWidget()
        
        # Simple mode inputs
        self.simple_widget = self._create_simple_inputs()
        self.input_stack.addWidget(self.simple_widget)
        
        # Advanced mode inputs
        self.advanced_widget = self._create_advanced_inputs()
        self.input_stack.addWidget(self.advanced_widget)
        
        layout.addWidget(self.input_stack)
        
        # Preview of computed segment lengths
        preview_group = QtWidgets.QGroupBox("Computed Segment Lengths")
        preview_layout = QtWidgets.QVBoxLayout()
        
        self.preview_text = QtWidgets.QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        self.preview_text.setStyleSheet(
            "font-family: 'Courier New'; font-size: 9pt; "
            "background-color: #f5f5f5; padding: 5px;"
        )
        preview_layout.addWidget(self.preview_text)
        
        compute_btn = QtWidgets.QPushButton("🔄 Compute Segment Lengths")
        compute_btn.setStyleSheet("padding: 8px; font-weight: bold;")
        compute_btn.clicked.connect(self._compute_and_preview)
        preview_layout.addWidget(compute_btn)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        defaults_btn = QtWidgets.QPushButton("Load Defaults")
        defaults_btn.clicked.connect(self._load_defaults)
        button_layout.addWidget(defaults_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setFixedSize(100, 35)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setFixedSize(100, 35)
        apply_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        apply_btn.clicked.connect(self._on_apply)
        button_layout.addWidget(apply_btn)
        
        layout.addLayout(button_layout)
    
    def _create_simple_inputs(self) -> QtWidgets.QWidget:
        """Create simple mode input form."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setSpacing(15)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        
        # Height
        height_layout = QtWidgets.QHBoxLayout()
        self.height_input = QtWidgets.QDoubleSpinBox()
        self.height_input.setRange(1.0, 2.5)
        self.height_input.setSingleStep(0.01)
        self.height_input.setDecimals(2)
        self.height_input.setValue(self.profile.height_m)
        self.height_input.setSuffix(" m")
        self.height_input.valueChanged.connect(self._on_modified)
        height_layout.addWidget(self.height_input)
        
        # Height in cm helper
        height_cm = QtWidgets.QLabel(f"({self.profile.height_m * 100:.0f} cm)")
        height_cm.setStyleSheet("color: #666; font-size: 9pt;")
        self.height_cm_label = height_cm
        height_layout.addWidget(height_cm)
        self.height_input.valueChanged.connect(
            lambda v: self.height_cm_label.setText(f"({v * 100:.0f} cm)")
        )
        
        height_layout.addStretch()
        layout.addRow("Height:", height_layout)
        
        # Weight
        weight_layout = QtWidgets.QHBoxLayout()
        self.weight_input = QtWidgets.QDoubleSpinBox()
        self.weight_input.setRange(30.0, 200.0)
        self.weight_input.setSingleStep(0.5)
        self.weight_input.setDecimals(1)
        self.weight_input.setValue(self.profile.weight_kg)
        self.weight_input.setSuffix(" kg")
        self.weight_input.valueChanged.connect(self._on_modified)
        weight_layout.addWidget(self.weight_input)
        weight_layout.addStretch()
        layout.addRow("Weight:", weight_layout)
        
        return widget
    
    def _create_advanced_inputs(self) -> QtWidgets.QWidget:
        """Create advanced mode input form."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setSpacing(10)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        
        # Helper function to create input
        def create_input(value, suffix=" m", min_val=0.1, max_val=3.0):
            inp = QtWidgets.QDoubleSpinBox()
            inp.setRange(min_val, max_val)
            inp.setSingleStep(0.01)
            inp.setDecimals(3)
            inp.setValue(value)
            inp.setSuffix(suffix)
            inp.valueChanged.connect(self._on_modified)
            
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.addWidget(inp)
            h_layout.addStretch()
            return h_layout, inp
        
        # Overall height
        h_layout, self.adv_height_input = create_input(self.profile.overall_height)
        layout.addRow("Overall Height:", h_layout)
        
        # Leg length
        leg_layout, self.leg_length_input = create_input(self.profile.leg_length)
        layout.addRow("Leg Length (floor to hip):", leg_layout)
        
        # Foot length
        foot_layout, self.foot_length_input = create_input(self.profile.foot_length, max_val=0.5)
        layout.addRow("Foot Length:", foot_layout)
        
        # Shoulder width
        shoulder_layout, self.shoulder_width_input = create_input(self.profile.shoulder_width, max_val=1.0)
        layout.addRow("Shoulder Width:", shoulder_layout)
        
        # Wingspan
        wingspan_layout, self.wingspan_input = create_input(self.profile.wingspan)
        layout.addRow("Wingspan (fingertip to fingertip):", wingspan_layout)
        
        # Hip width
        hip_layout, self.hip_width_input = create_input(self.profile.hip_width, max_val=1.0)
        layout.addRow("Hip Width:", hip_layout)
        
        # C7 to head top
        c7_layout, self.c7_to_head_input = create_input(self.profile.c7_to_head_top, max_val=0.5)
        layout.addRow("C7 to Head Top:", c7_layout)
        
        # Add measurement guide
        guide_label = QtWidgets.QLabel(
            "💡 Tip: Measure in standing position.\n"
            "Leg length: floor to greater trochanter (hip joint)\n"
            "Wingspan: arms extended horizontally"
        )
        guide_label.setStyleSheet(
            "font-size: 9pt; color: #666; padding: 10px; "
            "background-color: #e8f4f8; border-radius: 3px; margin-top: 10px;"
        )
        guide_label.setWordWrap(True)
        layout.addRow(guide_label)
        
        return widget
    
    def _on_mode_changed(self):
        """Handle mode radio button change."""
        if self.simple_radio.isChecked():
            self.input_stack.setCurrentIndex(0)
            self.profile.mode = "simple"
        else:
            self.input_stack.setCurrentIndex(1)
            self.profile.mode = "advanced"
        self._on_modified()
    
    def _on_modified(self):
        """Mark profile as modified."""
        self.modified = True
    
    def _load_profile_to_ui(self):
        """Load current profile data into UI widgets."""
        # Set mode
        if self.profile.mode == "simple":
            self.simple_radio.setChecked(True)
            self.input_stack.setCurrentIndex(0)
        else:
            self.advanced_radio.setChecked(True)
            self.input_stack.setCurrentIndex(1)
        
        # Set values (already done in widget creation, but update if changed)
        self.name_input.setText(self.profile.name)
        self.height_input.setValue(self.profile.height_m)
        self.weight_input.setValue(self.profile.weight_kg)
        
        if hasattr(self, 'adv_height_input'):
            self.adv_height_input.setValue(self.profile.overall_height)
            self.leg_length_input.setValue(self.profile.leg_length)
            self.foot_length_input.setValue(self.profile.foot_length)
            self.shoulder_width_input.setValue(self.profile.shoulder_width)
            self.wingspan_input.setValue(self.profile.wingspan)
            self.hip_width_input.setValue(self.profile.hip_width)
            self.c7_to_head_input.setValue(self.profile.c7_to_head_top)
    
    def _save_ui_to_profile(self):
        """Save UI values to profile object."""
        self.profile.name = self.name_input.text().strip() or "Unnamed"
        
        if self.profile.mode == "simple":
            self.profile.height_m = self.height_input.value()
            self.profile.weight_kg = self.weight_input.value()
        else:
            self.profile.overall_height = self.adv_height_input.value()
            self.profile.leg_length = self.leg_length_input.value()
            self.profile.foot_length = self.foot_length_input.value()
            self.profile.shoulder_width = self.shoulder_width_input.value()
            self.profile.wingspan = self.wingspan_input.value()
            self.profile.hip_width = self.hip_width_input.value()
            self.profile.c7_to_head_top = self.c7_to_head_input.value()
    
    def _compute_and_preview(self):
        """Compute segment lengths and show preview."""
        self._save_ui_to_profile()
        self.profile.compute_segment_lengths()
        
        # Build preview text
        preview = f"Profile: {self.profile.name} ({self.profile.mode} mode)\n"
        preview += "=" * 50 + "\n\n"
        
        preview += "LOWER BODY:\n"
        preview += f"  Upper Leg:  {self.profile.upper_leg_length*100:.1f} cm\n"
        preview += f"  Lower Leg:  {self.profile.lower_leg_length*100:.1f} cm\n"
        preview += f"  Foot:       {self.profile.foot_length_computed*100:.1f} cm\n"
        preview += f"  Total Leg:  {(self.profile.upper_leg_length + self.profile.lower_leg_length)*100:.1f} cm\n\n"
        
        preview += "UPPER BODY:\n"
        preview += f"  Trunk:      {self.profile.trunk_length*100:.1f} cm\n"
        preview += f"  Head:       {self.profile.head_length*100:.1f} cm\n\n"
        
        preview += "ARMS:\n"
        preview += f"  Upper Arm:  {self.profile.upper_arm_length*100:.1f} cm\n"
        preview += f"  Forearm:    {self.profile.forearm_length*100:.1f} cm\n"
        preview += f"  Hand:       {self.profile.hand_length*100:.1f} cm\n"
        preview += f"  Total Arm:  {(self.profile.upper_arm_length + self.profile.forearm_length + self.profile.hand_length)*100:.1f} cm\n\n"
        
        total_height = (self.profile.upper_leg_length + self.profile.lower_leg_length + 
                       self.profile.trunk_length + self.profile.head_length)
        preview += f"TOTAL HEIGHT: {total_height*100:.1f} cm ({total_height:.3f} m)\n"
        
        self.preview_text.setPlainText(preview)
    
    def _save_profile(self):
        """Save current profile to file."""
        self._save_ui_to_profile()
        self.profile.compute_segment_lengths()
        
        # Suggest filename
        suggested = f"{self.profile.name.replace(' ', '_')}.json"
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Anthropometry Profile",
            os.path.join(self.manager.profiles_dir, suggested),
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                self.profile.save(filename)
                QtWidgets.QMessageBox.information(
                    self,
                    "Profile Saved",
                    f"Anthropometry profile saved to:\n{filename}"
                )
                self.modified = False
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save profile:\n{str(e)}"
                )
    
    def _load_profile(self):
        """Load profile from file."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Anthropometry Profile",
            self.manager.profiles_dir,
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                self.profile = AnthropometryProfile.load(filename)
                self._load_profile_to_ui()
                self._compute_and_preview()
                self.modified = False
                
                QtWidgets.QMessageBox.information(
                    self,
                    "Profile Loaded",
                    f"Loaded profile: {self.profile.name}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Failed to load profile:\n{str(e)}"
                )
    
    def _load_defaults(self):
        """Show dialog to load default profiles."""
        defaults = self.manager.get_default_profiles()
        
        items = list(defaults.keys())
        item, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Load Default Profile",
            "Select a default profile:",
            items,
            0,
            False
        )
        
        if ok and item:
            self.profile = defaults[item]
            self._load_profile_to_ui()
            self._compute_and_preview()
            self.modified = True
    
    def _on_apply(self):
        """Apply changes and close dialog."""
        self._save_ui_to_profile()
        self.profile.compute_segment_lengths()
        self.accept()
    
    def get_profile(self) -> AnthropometryProfile:
        """Get the configured profile."""
        return self.profile
