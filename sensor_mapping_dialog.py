"""
Sensor-Segment Mapping Configurator for IKU.
Allows editing of the mapping.csv file to reassign sensors to segments.
"""

import os
import csv
from typing import Dict, List, Tuple
from PyQt5 import QtWidgets, QtCore, QtGui
from body_config import BODY_CONFIGS, get_body_config


class SensorMappingDialog(QtWidgets.QDialog):
    """Dialog for editing sensor-to-segment mappings."""
    
    def __init__(self, mapping_file: str = "mapping.csv", parent=None):
        super().__init__(parent)
        self.mapping_file = mapping_file
        self.mappings = {}  # {segment_name: sensor_id}
        self.modified = False
        
        self.setWindowTitle("Sensor-Segment Mapping Configuration")
        self.setModal(True)
        self.resize(700, 600)
        
        self._load_mappings()
        self._init_ui()
    
    def _load_mappings(self):
        """Load current mappings from CSV file."""
        if not os.path.exists(self.mapping_file):
            print(f"[SensorMapping] Warning: {self.mapping_file} not found")
            return
        
        try:
            with open(self.mapping_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sensor_id = row.get('IMU_ID', '').strip()
                    segment = row.get('Segment_Name', '').strip()
                    
                    # Skip empty rows and comments
                    if sensor_id and segment and not sensor_id.startswith('#'):
                        self.mappings[segment] = sensor_id
            
            print(f"[SensorMapping] Loaded {len(self.mappings)} mappings from {self.mapping_file}")
        except Exception as e:
            print(f"[SensorMapping] Error loading mappings: {e}")
    
    def _save_mappings(self):
        """Save mappings back to CSV file."""
        try:
            with open(self.mapping_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['IMU_ID', 'Segment_Name'])
                
                # Write mappings in a consistent order
                for segment in sorted(self.mappings.keys()):
                    sensor_id = self.mappings[segment]
                    if sensor_id:  # Only write non-empty mappings
                        writer.writerow([sensor_id, segment])
            
            print(f"[SensorMapping] Saved {len(self.mappings)} mappings to {self.mapping_file}")
            return True
        except Exception as e:
            print(f"[SensorMapping] Error saving mappings: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save mappings:\n{str(e)}"
            )
            return False
    
    def _init_ui(self):
        """Build the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title and instructions
        title = QtWidgets.QLabel("Sensor-Segment Mapping Configuration")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        instructions = QtWidgets.QLabel(
            "Map each segment to its corresponding sensor ID (8-digit hex).\n"
            "Use this when replacing faulty or low-battery sensors."
        )
        instructions.setStyleSheet("font-size: 10pt; color: #666; padding: 5px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Warning banner
        warning = QtWidgets.QLabel("⚠️ Changes require application restart to take effect")
        warning.setStyleSheet(
            "font-size: 10pt; font-weight: bold; color: #e67e22; "
            "background-color: #fef5e7; padding: 8px; border-radius: 3px; "
            "border: 1px solid #f39c12;"
        )
        warning.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(warning)
        
        # Mapping table
        table_group = QtWidgets.QGroupBox("Sensor Mappings")
        table_layout = QtWidgets.QVBoxLayout()
        
        # Create table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Segment Name', 'Sensor ID (Hex)', 'Status'])
        
        # Configure table
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                font-size: 10pt;
                gridline-color: #ddd;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)
        
        # Populate table with all segments from BODY_MODEL
        all_segments = ['pelvis', 'trunk', 'head',
                       'upper_arm_right', 'forearm_right', 'hand_right',
                       'upper_arm_left', 'forearm_left', 'hand_left',
                       'upper_leg_right', 'lower_leg_right', 'foot_right',
                       'upper_leg_left', 'lower_leg_left', 'foot_left']
        
        self.table.setRowCount(len(all_segments))
        self.segment_inputs = {}
        
        for row, segment in enumerate(all_segments):
            # Segment name (read-only)
            segment_item = QtWidgets.QTableWidgetItem(segment)
            segment_item.setFlags(segment_item.flags() & ~QtCore.Qt.ItemIsEditable)
            segment_item.setBackground(QtGui.QColor("#f5f5f5"))
            self.table.setItem(row, 0, segment_item)
            
            # Sensor ID input
            sensor_id = self.mappings.get(segment, '')
            sensor_input = QtWidgets.QLineEdit(sensor_id)
            sensor_input.setPlaceholderText("00000000")
            sensor_input.setMaxLength(8)
            sensor_input.setFont(QtGui.QFont("Courier New", 10))
            sensor_input.textChanged.connect(self._on_mapping_changed)
            self.table.setCellWidget(row, 1, sensor_input)
            self.segment_inputs[segment] = sensor_input
            
            # Status indicator
            status_item = QtWidgets.QTableWidgetItem()
            status_item.setFlags(status_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 2, status_item)
            self._update_status(row, sensor_id)
        
        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # Quick actions
        actions_layout = QtWidgets.QHBoxLayout()
        
        clear_btn = QtWidgets.QPushButton("Clear All")
        clear_btn.setToolTip("Clear all sensor mappings")
        clear_btn.clicked.connect(self._clear_all)
        actions_layout.addWidget(clear_btn)
        
        validate_btn = QtWidgets.QPushButton("Validate IDs")
        validate_btn.setToolTip("Check for duplicate sensor IDs")
        validate_btn.clicked.connect(self._validate_mappings)
        actions_layout.addWidget(validate_btn)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
        
        # Modified indicator
        self.modified_label = QtWidgets.QLabel("")
        self.modified_label.setStyleSheet("font-size: 9pt; color: #e67e22; font-style: italic;")
        layout.addWidget(self.modified_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setFixedSize(100, 35)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.setFixedSize(100, 35)
        save_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        save_btn.clicked.connect(self._on_save)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
    
    def _update_status(self, row: int, sensor_id: str):
        """Update status indicator for a row."""
        status_item = self.table.item(row, 2)
        
        if not sensor_id:
            status_item.setText("❌ Empty")
            status_item.setForeground(QtGui.QColor("#95a5a6"))
        elif len(sensor_id) != 8:
            status_item.setText("⚠️ Invalid")
            status_item.setForeground(QtGui.QColor("#e67e22"))
        elif not all(c in '0123456789ABCDEFabcdef' for c in sensor_id):
            status_item.setText("⚠️ Not Hex")
            status_item.setForeground(QtGui.QColor("#e67e22"))
        else:
            status_item.setText("✓ Valid")
            status_item.setForeground(QtGui.QColor("#27ae60"))
    
    def _on_mapping_changed(self):
        """Handle mapping change."""
        self.modified = True
        self.modified_label.setText("⚠️ Unsaved changes")
        
        # Update all status indicators
        for row, segment in enumerate(self.segment_inputs.keys()):
            sensor_input = self.segment_inputs[segment]
            sensor_id = sensor_input.text().strip().upper()
            self._update_status(row, sensor_id)
    
    def _clear_all(self):
        """Clear all sensor mappings."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear All Mappings",
            "Are you sure you want to clear all sensor mappings?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            for sensor_input in self.segment_inputs.values():
                sensor_input.clear()
            self.modified = True
            self.modified_label.setText("⚠️ Unsaved changes - All cleared")
    
    def _validate_mappings(self):
        """Validate mappings for duplicates and format."""
        sensor_ids = {}
        duplicates = []
        invalid = []
        
        for segment, sensor_input in self.segment_inputs.items():
            sensor_id = sensor_input.text().strip().upper()
            
            if not sensor_id:
                continue
            
            # Check format
            if len(sensor_id) != 8:
                invalid.append(f"{segment}: '{sensor_id}' (wrong length)")
                continue
            
            if not all(c in '0123456789ABCDEF' for c in sensor_id):
                invalid.append(f"{segment}: '{sensor_id}' (not hexadecimal)")
                continue
            
            # Check for duplicates
            if sensor_id in sensor_ids:
                duplicates.append(f"{sensor_id}: {sensor_ids[sensor_id]} and {segment}")
            else:
                sensor_ids[sensor_id] = segment
        
        # Show results
        if duplicates or invalid:
            message = ""
            if duplicates:
                message += "**Duplicate Sensor IDs:**\n" + "\n".join(duplicates) + "\n\n"
            if invalid:
                message += "**Invalid IDs:**\n" + "\n".join(invalid)
            
            QtWidgets.QMessageBox.warning(self, "Validation Issues", message)
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Validation Successful",
                f"All {len(sensor_ids)} sensor IDs are valid and unique!"
            )
    
    def _on_save(self):
        """Save mappings to file."""
        # Collect mappings
        new_mappings = {}
        for segment, sensor_input in self.segment_inputs.items():
            sensor_id = sensor_input.text().strip().upper()
            if sensor_id:
                new_mappings[segment] = sensor_id
        
        # Validate before saving
        sensor_ids = set()
        for segment, sensor_id in new_mappings.items():
            if len(sensor_id) != 8:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Sensor ID",
                    f"Sensor ID for '{segment}' must be 8 characters.\n"
                    f"Current: '{sensor_id}'"
                )
                return
            
            if sensor_id in sensor_ids:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Duplicate Sensor ID",
                    f"Sensor ID '{sensor_id}' is assigned to multiple segments.\n"
                    f"Each sensor can only be assigned to one segment."
                )
                return
            
            sensor_ids.add(sensor_id)
        
        # Update and save
        self.mappings = new_mappings
        if self._save_mappings():
            self.modified = False
            QtWidgets.QMessageBox.information(
                self,
                "Mappings Saved",
                f"Successfully saved {len(self.mappings)} sensor mappings.\n\n"
                "⚠️ Please restart the application for changes to take effect."
            )
            self.accept()
