"""Visualization components for 3D skeleton and angle table - OPTIMIZED"""

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional
from config import BODY_MODEL


class Skeleton3DCanvas(FigureCanvasQTAgg):
    """3D skeleton visualization using matplotlib - OPTIMIZED"""

    CONNECTIONS = [
        ("pelvis", "trunk"),
        ("trunk", "head"),
        ("head", "head_tip"),
        ("trunk_tip_left", "trunk_tip_right"),
        ("trunk_tip_right", "upper_arm_right"),
        ("upper_arm_right", "forearm_right"),
        ("forearm_right", "hand_right"),
        ("trunk_tip_left", "upper_arm_left"),
        ("upper_arm_left", "forearm_left"),
        ("forearm_left", "hand_left"),
        ("hip_left", "hip_right"),
        ("hip_right", "upper_leg_right"),
        ("upper_leg_right", "lower_leg_right"),
        ("lower_leg_right", "foot_right"),
        ("hip_left", "upper_leg_left"),
        ("upper_leg_left", "lower_leg_left"),
        ("lower_leg_left", "foot_left"),
    ]

    def __init__(self, parent=None):
        # Rectangular figure optimized for vertical skeleton (portrait orientation)
        self.fig = Figure(figsize=(4, 7), dpi=90)  # 4 wide, 7 tall
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111, projection="3d")
        
        # Tighter axis limits to fit skeleton better (less empty space)
        self.ax.set_xlim([-0.8, 0.8])   # Narrower horizontal
        self.ax.set_ylim([-0.8, 0.8])   # Narrower depth
        self.ax.set_zlim([-0.1, 2.1])   # Taller vertical (includes feet at 0)
        
        self.ax.set_xlabel("X (Forward)")
        self.ax.set_ylabel("Z (Right/Left)")
        self.ax.set_zlabel("Y (Up)")
        self.ax.view_init(elev=10, azim=-60)

        # Reduce visual clutter for performance
        self.ax.grid(False)
        
        self.world_correction = R.identity()
        
        # Track pelvis rotation for pelvis-relative views
        self.pelvis_rotation = R.identity()
        self.pelvis_yaw = 0.0  # in degrees

        # Persistent artists (created once, updated in-place)
        self._bone_lines = {}
        for a, b in self.CONNECTIONS:
            (ln,) = self.ax.plot([0, 0], [0, 0], [0, 0], 'b-', linewidth=2)
            self._bone_lines[(a, b)] = ln

        # One scatter for all joints
        self._joint_scatter = self.ax.scatter([0], [0], [0], c='r', s=30, marker='o')

        self.fig.tight_layout()

    def set_heading_lock(self, pelvis_rot: R, enabled: bool = True):
        """Remove pelvis yaw to keep subject facing forward."""
        if not enabled or pelvis_rot is None:
            self.world_correction = R.identity()
            return

        fwd = pelvis_rot.apply([1.0, 0.0, 0.0])
        heading = np.arctan2(fwd[2], fwd[0])
        self.world_correction = R.from_rotvec([0.0, -heading, 0.0])

    def view_front(self):
        """View from front - body-relative (heading lock keeps body facing forward)"""
        self.ax.view_init(elev=10, azim=180)
        self.draw_idle()

    def view_back(self):
        """View from back - body-relative"""
        self.ax.view_init(elev=10, azim=0)
        self.draw_idle()

    def view_left(self):
        """View from left - body-relative"""
        self.ax.view_init(elev=10, azim=-90)
        self.draw_idle()

    def view_right(self):
        """View from right - body-relative"""
        self.ax.view_init(elev=10, azim=90)
        self.draw_idle()

    def view_top(self):
        """View from top - body-relative"""
        self.ax.view_init(elev=90, azim=180)
        self.draw_idle()
    
    def view_default(self):
        """Default oblique view"""
        self.ax.view_init(elev=10, azim=-60)
        self.draw_idle()
    
    def update_pelvis_orientation(self, pelvis_rot: R):
        """Update pelvis orientation for pelvis-relative views."""
        self.pelvis_rotation = pelvis_rot
        
        # Extract yaw angle from pelvis rotation
        # Forward direction in world frame
        forward = pelvis_rot.apply([1.0, 0.0, 0.0])
        
        # Yaw angle (rotation around Y/vertical axis)
        self.pelvis_yaw = np.arctan2(forward[2], forward[0]) * 180 / np.pi

    def update_skeleton(self, segment_positions: Dict[str, np.ndarray]):
        """Update skeleton - OPTIMIZED with minimal conversions"""
        if not segment_positions:
            return
        
        # Use cached conversion function
        def _to_vis(p: np.ndarray) -> tuple:
            return float(p[0]), float(-p[2]), float(p[1])

        # Update bones (in-place updates, no new artists)
        for (a, b), ln in self._bone_lines.items():
            if a in segment_positions and b in segment_positions:
                p1 = self.world_correction.apply(segment_positions[a])
                p2 = self.world_correction.apply(segment_positions[b])
                x1, y1, z1 = _to_vis(p1)
                x2, y2, z2 = _to_vis(p2)
                ln.set_data([x1, x2], [y1, y2])
                ln.set_3d_properties([z1, z2])

        # Update joints (single batch update)
        corrected_positions = [
            self.world_correction.apply(pos) 
            for pos in segment_positions.values()
        ]
        vis_positions = [_to_vis(p) for p in corrected_positions]
        xs, ys, zs = zip(*vis_positions)
        self._joint_scatter._offsets3d = (list(xs), list(ys), list(zs))

        # Use draw_idle() instead of draw() - only redraws when idle
        self.draw_idle()


class AngleTableWidget(QtWidgets.QWidget):
    """Real-time display showing all joints with labeled axes - OPTIMIZED"""
    
    def __init__(self, parent=None, use_quaternions: bool = True):
        super().__init__(parent)
        self.use_quaternions = use_quaternions
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # Header
        header = QtWidgets.QLabel("All Joint Angles" if not use_quaternions else "All Joint Rotations")
        header.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 5px;")
        main_layout.addWidget(header)
        
        # Scroll area for joint boxes
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # Container for all joint boxes
        container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(container)
        self.grid_layout.setSpacing(8)
        
        # Create boxes for each joint
        self.joint_boxes = {}
        row, col = 0, 0
        
        for seg_name, seg_def in BODY_MODEL.items():
            if seg_def.parent is None:
                continue
            
            joint_name = f"{seg_def.parent}_{seg_name}"
            
            # Create a group box for this joint (smaller for all joints)
            group_box = QtWidgets.QGroupBox()
            group_box.setStyleSheet("""
                QGroupBox {
                    border: 1px solid #aaaaaa;
                    border-radius: 4px;
                    margin-top: 8px;
                    font-weight: bold;
                    padding-top: 8px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 8px;
                    padding: 0 3px;
                }
            """)
            
            # Joint title
            display_name = seg_name.replace('_', ' ').title()
            group_box.setTitle(display_name)
            
            box_layout = QtWidgets.QVBoxLayout(group_box)
            box_layout.setSpacing(3)
            
            # Create labels for each axis
            if use_quaternions:
                labels = ['w', 'x', 'y', 'z']
                num_components = 4
            else:
                labels = seg_def.axis_labels
                num_components = 3
            
            # Store value labels for updating
            value_labels = []
            
            for i, axis_name in enumerate(labels):
                # Create a horizontal layout for each axis
                axis_layout = QtWidgets.QHBoxLayout()
                
                # Axis name label (smaller font for all joints view)
                name_label = QtWidgets.QLabel(f"{axis_name}:")
                name_label.setMinimumWidth(70)
                name_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
                axis_layout.addWidget(name_label)
                
                # Value label
                value_label = QtWidgets.QLabel("--")
                value_label.setMinimumWidth(70)
                value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                value_label.setStyleSheet("font-family: 'Courier New'; font-size: 9pt; padding: 2px;")
                axis_layout.addWidget(value_label)
                
                value_labels.append(value_label)
                box_layout.addLayout(axis_layout)
            
            # Store references
            self.joint_boxes[joint_name] = {
                'box': group_box,
                'labels': value_labels,
                'num_components': num_components
            }
            
            # Add to grid (3 columns for all joints)
            self.grid_layout.addWidget(group_box, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        
        # Update counter (more decimation for all joints)
        self._update_counter = 0
        self._update_decimation = 2  # Update every 2 frames
    
    def update_angles(self, joint_data: Dict[str, tuple]):
        """Update all joint boxes with current values"""
        self._update_counter += 1
        if self._update_counter < self._update_decimation:
            return
        self._update_counter = 0
        
        for joint_name, box_info in self.joint_boxes.items():
            if joint_name not in joint_data:
                continue
            
            quat, euler = joint_data[joint_name]
            values = quat if self.use_quaternions else euler
            
            for i, value_label in enumerate(box_info['labels']):
                if i >= len(values):
                    continue
                
                if self.use_quaternions:
                    value_label.setText(f"{values[i]:6.3f}")
                    
                    # Color coding for quaternions
                    if abs(values[i]) > 0.7:
                        value_label.setStyleSheet(
                            "font-family: 'Courier New'; font-size: 9pt; "
                            "padding: 2px; background-color: #ffe6e6; border-radius: 2px;"
                        )
                    else:
                        value_label.setStyleSheet(
                            "font-family: 'Courier New'; font-size: 9pt; "
                            "padding: 2px; background-color: #f5f5f5; border-radius: 2px;"
                        )
                else:
                    value_label.setText(f"{values[i]:5.1f}°")
                    
                    # Color coding for angles
                    abs_val = abs(values[i])
                    if abs_val > 60:
                        bg_color = "#ffe6e6"
                    elif abs_val > 30:
                        bg_color = "#fff9e6"
                    else:
                        bg_color = "#e6f7e6"
                    
                    value_label.setStyleSheet(
                        f"font-family: 'Courier New'; font-size: 9pt; "
                        f"padding: 2px; background-color: {bg_color}; border-radius: 2px;"
                    )



class CompactAngleTableWidget(QtWidgets.QWidget):
    """Display showing all joints in three anatomical columns: single/left/right"""
    
    # Mapping of joint technical names to anatomical names
    JOINT_DISPLAY_NAMES = {
        'pelvis_trunk': 'Lumbar',
        'trunk_head': 'Neck',
        'trunk_upper_arm_right': 'R Shoulder',
        'upper_arm_right_forearm_right': 'R Elbow',
        'forearm_right_hand_right': 'R Wrist',
        'trunk_upper_arm_left': 'L Shoulder',
        'upper_arm_left_forearm_left': 'L Elbow',
        'forearm_left_hand_left': 'L Wrist',
        'pelvis_upper_leg_right': 'R Hip',
        'upper_leg_right_lower_leg_right': 'R Knee',
        'lower_leg_right_foot_right': 'R Ankle',
        'pelvis_upper_leg_left': 'L Hip',
        'upper_leg_left_lower_leg_left': 'L Knee',
        'lower_leg_left_foot_left': 'L Ankle',
    }
    
    # Anatomical organization: head to toe, single joints | left side | right side
    JOINT_LAYOUT = {
        'single': [  # Midline joints (column 0)
            'trunk_head',           # Neck
            'pelvis_trunk',         # Lumbar spine
        ],
        'left': [  # Left side (column 1)
            'trunk_upper_arm_left',              # L Shoulder
            'upper_arm_left_forearm_left',       # L Elbow
            'forearm_left_hand_left',            # L Wrist
            'pelvis_upper_leg_left',             # L Hip
            'upper_leg_left_lower_leg_left',     # L Knee
            'lower_leg_left_foot_left',          # L Ankle
        ],
        'right': [  # Right side (column 2)
            'trunk_upper_arm_right',              # R Shoulder
            'upper_arm_right_forearm_right',      # R Elbow
            'forearm_right_hand_right',           # R Wrist
            'pelvis_upper_leg_right',             # R Hip
            'upper_leg_right_lower_leg_right',    # R Knee
            'lower_leg_right_foot_right',         # R Ankle
        ],
    }
    
    def __init__(self, parent=None, use_quaternions: bool = True):
        super().__init__(parent)
        self.use_quaternions = use_quaternions
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # ==================== TOP ROW: Single Joints ====================
        single_row = QtWidgets.QHBoxLayout()
        single_row.setSpacing(10)
        
        # Single joints header
        single_header = QtWidgets.QLabel("Single Joints (Midline)")
        single_header.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 5px;")
        single_header.setAlignment(QtCore.Qt.AlignCenter)
        
        single_container = QtWidgets.QWidget()
        single_layout = QtWidgets.QVBoxLayout(single_container)
        single_layout.setSpacing(5)
        single_layout.addWidget(single_header)
        
        # Row for single joints
        single_joints_row = QtWidgets.QHBoxLayout()
        single_joints_row.setSpacing(10)
        
        self.joint_boxes = {}
        
        # Add single joints horizontally
        for joint_name in self.JOINT_LAYOUT['single']:
            display_name = self.JOINT_DISPLAY_NAMES.get(joint_name, 
                                                        joint_name.replace('_', ' ').title())
            
            # Create larger joint box
            group_box = QtWidgets.QGroupBox(display_name)
            group_box.setStyleSheet("""
                QGroupBox {
                    border: 2px solid #3498db;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                    font-size: 11pt;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
            """)
            
            box_layout = QtWidgets.QVBoxLayout()
            box_layout.setSpacing(3)
            box_layout.setContentsMargins(8, 8, 8, 8)
            
            # Get axis labels for this joint
            seg_name = joint_name.split('_')[-1]
            axis_labels = None
            for model_seg_name, seg_def in BODY_MODEL.items():
                if model_seg_name == seg_name or model_seg_name in joint_name:
                    parent_seg = seg_def.parent
                    check_joint = f"{parent_seg}_{model_seg_name}"
                    if check_joint == joint_name:
                        axis_labels = seg_def.axis_labels
                        break
            
            if axis_labels is None:
                axis_labels = ("Angle 0", "Angle 1", "Angle 2")
            
            # Create value labels for 3 angles/components
            value_labels = []
            for i, axis_label in enumerate(axis_labels):
                row_layout = QtWidgets.QHBoxLayout()
                
                # Axis label
                label = QtWidgets.QLabel(f"{axis_label}:")
                label.setStyleSheet("font-size: 10pt; color: #333; font-weight: bold;")
                label.setMinimumWidth(80)
                row_layout.addWidget(label)
                
                # Value label (larger for single joints)
                value_label = QtWidgets.QLabel("--")
                value_label.setAlignment(QtCore.Qt.AlignRight)
                value_label.setStyleSheet(
                    "font-family: 'Courier New'; font-size: 11pt; font-weight: bold; "
                    "padding: 4px; background-color: #ecf0f1; border-radius: 3px;"
                )
                value_label.setMinimumWidth(70)
                row_layout.addWidget(value_label)
                
                value_labels.append(value_label)
                box_layout.addLayout(row_layout)
            
            group_box.setLayout(box_layout)
            single_joints_row.addWidget(group_box)
            
            # Store reference
            self.joint_boxes[joint_name] = value_labels
        
        single_layout.addLayout(single_joints_row)
        main_layout.addWidget(single_container)
        
        # ==================== BOTTOM ROW: Left and Right Columns ====================
        columns_layout = QtWidgets.QHBoxLayout()
        columns_layout.setSpacing(10)
        
        # Create left and right columns
        for col_name, joints_key in [('Left Side', 'left'), ('Right Side', 'right')]:
            # Column container
            col_widget = QtWidgets.QWidget()
            col_layout = QtWidgets.QVBoxLayout(col_widget)
            col_layout.setSpacing(8)
            col_layout.setContentsMargins(5, 5, 5, 5)
            
            # Column header
            header = QtWidgets.QLabel(col_name)
            header.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 5px; "
                               "background-color: #3498db; color: white; border-radius: 3px;")
            header.setAlignment(QtCore.Qt.AlignCenter)
            col_layout.addWidget(header)
            
            # Add joints for this column
            for joint_name in self.JOINT_LAYOUT[joints_key]:
                # Get anatomical display name
                display_name = self.JOINT_DISPLAY_NAMES.get(joint_name, 
                                                            joint_name.replace('_', ' ').title())
                
                # Create joint box (larger than before)
                group_box = QtWidgets.QGroupBox(display_name)
                group_box.setStyleSheet("""
                    QGroupBox {
                        border: 2px solid #aaaaaa;
                        border-radius: 4px;
                        margin-top: 10px;
                        font-weight: bold;
                        font-size: 10pt;
                        padding-top: 10px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 8px;
                        padding: 0 4px;
                    }
                """)
                
                box_layout = QtWidgets.QVBoxLayout()
                box_layout.setSpacing(3)
                box_layout.setContentsMargins(6, 6, 6, 6)
                
                # Get axis labels for this joint
                seg_name = joint_name.split('_')[-1]
                axis_labels = None
                for model_seg_name, seg_def in BODY_MODEL.items():
                    if model_seg_name == seg_name or model_seg_name in joint_name:
                        parent_seg = seg_def.parent
                        check_joint = f"{parent_seg}_{model_seg_name}"
                        if check_joint == joint_name:
                            axis_labels = seg_def.axis_labels
                            break
                
                if axis_labels is None:
                    axis_labels = ("Angle 0", "Angle 1", "Angle 2")
                
                # Create value labels for 3 angles/components
                value_labels = []
                for i, axis_label in enumerate(axis_labels):
                    row_layout = QtWidgets.QHBoxLayout()
                    
                    # Axis label (larger font)
                    label = QtWidgets.QLabel(f"{axis_label}:")
                    label.setStyleSheet("font-size: 9pt; color: #444; font-weight: 500;")
                    label.setMinimumWidth(70)
                    row_layout.addWidget(label)
                    
                    # Value label (larger)
                    value_label = QtWidgets.QLabel("--")
                    value_label.setAlignment(QtCore.Qt.AlignRight)
                    value_label.setStyleSheet(
                        "font-family: 'Courier New'; font-size: 10pt; "
                        "padding: 3px; background-color: #f5f5f5; border-radius: 2px;"
                    )
                    value_label.setMinimumWidth(60)
                    row_layout.addWidget(value_label)
                    
                    value_labels.append(value_label)
                    box_layout.addLayout(row_layout)
                
                group_box.setLayout(box_layout)
                col_layout.addWidget(group_box)
                
                # Store reference
                self.joint_boxes[joint_name] = value_labels
            
            col_layout.addStretch()
            columns_layout.addWidget(col_widget)
        
        main_layout.addLayout(columns_layout)
    
    def update_angles(self, joint_data: Dict[str, np.ndarray]):
        """Update displayed angles"""
        for joint_name, value_labels in self.joint_boxes.items():
            if joint_name not in joint_data:
                continue
            
            values = joint_data[joint_name]
            
            # Ensure values is a 1D array
            if isinstance(values, np.ndarray) and values.ndim > 1:
                values = values.flatten()
            
            for i, value_label in enumerate(value_labels):
                if i >= len(values):
                    continue
                
                # Extract scalar value
                try:
                    val = float(values[i])
                except (TypeError, IndexError):
                    value_label.setText("--")
                    continue
                
                if self.use_quaternions:
                    # Display quaternion components
                    value_label.setText(f"{val:+.3f}")
                    
                    # Subtle color coding
                    if abs(val) > 0.7:
                        value_label.setStyleSheet(
                            "font-family: 'Courier New'; font-size: 9pt; "
                            "padding: 2px; background-color: #e6f7ff; border-radius: 2px;"
                        )
                    else:
                        value_label.setStyleSheet(
                            "font-family: 'Courier New'; font-size: 9pt; "
                            "padding: 2px; background-color: #f5f5f5; border-radius: 2px;"
                        )
                else:
                    value_label.setText(f"{val:5.1f}°")
                    
                    # Color coding for angles
                    abs_val = abs(val)
                    if abs_val > 60:
                        bg_color = "#ffe6e6"
                    elif abs_val > 30:
                        bg_color = "#fff9e6"
                    else:
                        bg_color = "#e6f7e6"
                    
                    value_label.setStyleSheet(
                        f"font-family: 'Courier New'; font-size: 9pt; "
                        f"padding: 2px; background-color: {bg_color}; border-radius: 2px;"
                    )
