"""
Anthropometry system for IKU.
Manages body segment dimensions with simple (height/weight) and advanced (detailed measurements) modes.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np


@dataclass
class AnthropometryProfile:
    """Complete anthropometry profile for a subject."""
    
    # Metadata
    name: str = "Default"
    mode: str = "simple"  # 'simple' or 'advanced'
    
    # Simple mode inputs
    height_m: float = 1.75  # meters
    weight_kg: float = 75.0  # kilograms
    
    # Advanced mode inputs (meters)
    overall_height: float = 1.75
    leg_length: float = 0.90  # Floor to hip joint
    foot_length: float = 0.26
    shoulder_width: float = 0.40  # Between shoulder joints
    wingspan: float = 1.75  # Fingertip to fingertip
    hip_width: float = 0.30  # Between hip joints
    c7_to_head_top: float = 0.25  # C7 vertebra to top of head
    
    # Computed segment lengths (set by compute_segment_lengths)
    pelvis_height: float = 0.10
    trunk_length: float = 0.50
    head_length: float = 0.25
    upper_arm_length: float = 0.30
    forearm_length: float = 0.25
    hand_length: float = 0.18
    upper_leg_length: float = 0.40
    lower_leg_length: float = 0.40
    foot_length_computed: float = 0.25
    
    def compute_segment_lengths(self):
        """
        Compute individual segment lengths based on mode and inputs.
        Uses anthropometric scaling relationships from literature.
        """
        if self.mode == "simple":
            self._compute_from_height_weight()
        else:  # advanced
            self._compute_from_measurements()
    
    def _compute_from_height_weight(self):
        """
        Estimate segment lengths from height and weight using standard ratios.
        Based on de Leva (1996) and Winter (2009).
        
        IMPORTANT: These ratios represent joint-to-joint distances (anatomical segment lengths),
        not standing height. The skeleton model will appear ~6% taller than standing height
        because it includes:
        - Skull height above C7 (not compressed)
        - Joint capsule thicknesses
        - Foot at ankle level (not on ground)
        
        This is biomechanically correct for motion analysis.
        """
        h = self.height_m
        
        # Standard anthropometric ratios (joint center to joint center)
        # Source: de Leva (1996), Winter (2009)
        self.upper_leg_length = h * 0.245  # Hip to knee: 24.5% of height
        self.lower_leg_length = h * 0.246  # Knee to ankle: 24.6% of height
        self.foot_length_computed = h * 0.152  # Ankle to toe: 15.2% of height
        
        self.trunk_length = h * 0.288  # Pelvis to C7: 28.8% of height
        self.head_length = h * 0.130  # C7 to vertex: 13.0% of height
        
        self.upper_arm_length = h * 0.186  # Shoulder to elbow: 18.6% of height
        self.forearm_length = h * 0.146  # Elbow to wrist: 14.6% of height
        self.hand_length = h * 0.108  # Wrist to fingertip: 10.8% of height
        
        # Pelvis (structural reference)
        self.pelvis_height = h * 0.10
        
        # Calculate skeletal height (joint center chain)
        skeletal_height = (self.lower_leg_length + self.upper_leg_length + 
                          self.trunk_length + self.head_length)
        
        print(f"[Anthropometry] Computed from height {h:.2f}m, weight {self.weight_kg:.1f}kg")
        print(f"[Anthropometry] Standing height: {h:.3f}m")
        print(f"[Anthropometry] Skeletal height: {skeletal_height:.3f}m ({skeletal_height/h*100:.1f}% of standing)")
        print(f"[Anthropometry] Note: ~6% difference is normal (soft tissue, skull, foot contact)")
    
    def _compute_from_measurements(self):
        """Compute segment lengths from detailed measurements."""
        # Use direct measurements where available
        
        # Leg segments from leg length
        total_leg = self.leg_length
        self.upper_leg_length = total_leg * 0.530  # ~53% of leg length is thigh
        self.lower_leg_length = total_leg * 0.470  # ~47% is shin
        self.foot_length_computed = self.foot_length
        
        # Trunk and head from overall height
        # Note: overall_height is standing height (floor to head top)
        # leg_length is floor to hip joint (GT)
        # So: overall_height - leg_length = trunk + head
        # Foot is separate (extends forward from ankle, not part of vertical height)
        
        trunk_and_head = self.overall_height - self.leg_length
        self.head_length = self.c7_to_head_top
        self.trunk_length = trunk_and_head - self.head_length
        
        # Arm segments from wingspan
        # Wingspan ≈ shoulder_width + 2*(upper_arm + forearm + hand)
        arm_span_one_side = (self.wingspan - self.shoulder_width) / 2.0
        
        self.upper_arm_length = arm_span_one_side * 0.436  # ~43.6% of arm
        self.forearm_length = arm_span_one_side * 0.346  # ~34.6% of arm
        self.hand_length = arm_span_one_side * 0.218  # ~21.8% of arm
        
        self.pelvis_height = 0.10  # Small fixed value
        
        # Calculate vertical skeletal height (ankle to head, foot not included in vertical)
        skeletal_height = (self.upper_leg_length + self.lower_leg_length + 
                          self.trunk_length + self.head_length)
        
        print(f"[Anthropometry] Computed from detailed measurements")
        print(f"[Anthropometry] Standing height: {self.overall_height:.3f}m")
        print(f"[Anthropometry] Leg (floor to hip): {self.leg_length:.3f}m")
        print(f"[Anthropometry] Segments - Upper leg: {self.upper_leg_length:.3f}m, Lower leg: {self.lower_leg_length:.3f}m")
        print(f"[Anthropometry] Segments - Trunk: {self.trunk_length:.3f}m, Head: {self.head_length:.3f}m")
        print(f"[Anthropometry] Vertical skeletal: {skeletal_height:.3f}m ({skeletal_height/self.overall_height*100:.1f}%)")
        print(f"[Anthropometry] Foot: {self.foot_length_computed:.3f}m (extends forward from ankle)")
    
    def get_segment_length(self, segment_name: str) -> float:
        """
        Get length for a specific segment.
        
        Args:
            segment_name: Name of segment (e.g., 'upper_arm_left', 'trunk')
            
        Returns:
            Length in meters
        """
        # Map segment names to computed lengths
        mapping = {
            'pelvis': self.pelvis_height,
            'trunk': self.trunk_length,
            'head': self.head_length,
            'upper_arm_right': self.upper_arm_length,
            'upper_arm_left': self.upper_arm_length,
            'forearm_right': self.forearm_length,
            'forearm_left': self.forearm_length,
            'hand_right': self.hand_length,
            'hand_left': self.hand_length,
            'upper_leg_right': self.upper_leg_length,
            'upper_leg_left': self.upper_leg_length,
            'lower_leg_right': self.lower_leg_length,
            'lower_leg_left': self.lower_leg_length,
            'foot_right': self.foot_length_computed,
            'foot_left': self.foot_length_computed,
        }
        
        return mapping.get(segment_name, 0.30)  # Default 30cm if not found
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary for saving."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnthropometryProfile':
        """Create profile from dictionary."""
        return cls(**data)
    
    def save(self, filepath: str):
        """Save profile to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Anthropometry] Saved profile '{self.name}' to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AnthropometryProfile':
        """Load profile from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"[Anthropometry] Loaded profile '{data.get('name', 'Unknown')}' from {filepath}")
        return cls.from_dict(data)


class AnthropometryManager:
    """Manages anthropometry profiles and provides default profiles."""
    
    def __init__(self, profiles_dir: str = "anthropometry_profiles"):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)
        
        self.current_profile = AnthropometryProfile()
        self.current_profile.compute_segment_lengths()
    
    def set_profile(self, profile: AnthropometryProfile):
        """Set the current active profile."""
        self.current_profile = profile
        self.current_profile.compute_segment_lengths()
        print(f"[Anthropometry] Active profile: {profile.name} ({profile.mode} mode)")
    
    def get_segment_length(self, segment_name: str) -> float:
        """Get length for a segment from current profile."""
        return self.current_profile.get_segment_length(segment_name)
    
    def save_profile(self, profile: AnthropometryProfile, filename: str = None):
        """Save a profile to the profiles directory."""
        if filename is None:
            # Generate filename from profile name
            filename = f"{profile.name.replace(' ', '_')}.json"
        
        filepath = os.path.join(self.profiles_dir, filename)
        profile.save(filepath)
    
    def load_profile(self, filename: str) -> AnthropometryProfile:
        """Load a profile from the profiles directory."""
        filepath = os.path.join(self.profiles_dir, filename)
        return AnthropometryProfile.load(filepath)
    
    def list_profiles(self) -> list:
        """List all available profile files."""
        if not os.path.exists(self.profiles_dir):
            return []
        
        profiles = []
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                profiles.append(filename)
        return sorted(profiles)
    
    def get_default_profiles(self) -> Dict[str, AnthropometryProfile]:
        """Get set of default profiles for common body types."""
        profiles = {}
        
        # Average adult male
        male = AnthropometryProfile(
            name="Average Adult Male",
            mode="simple",
            height_m=1.75,
            weight_kg=80.0
        )
        male.compute_segment_lengths()
        profiles["male"] = male
        
        # Average adult female
        female = AnthropometryProfile(
            name="Average Adult Female",
            mode="simple",
            height_m=1.62,
            weight_kg=65.0
        )
        female.compute_segment_lengths()
        profiles["female"] = female
        
        # Tall male
        tall_male = AnthropometryProfile(
            name="Tall Male",
            mode="simple",
            height_m=1.90,
            weight_kg=90.0
        )
        tall_male.compute_segment_lengths()
        profiles["tall_male"] = tall_male
        
        # Tall female
        tall_female = AnthropometryProfile(
            name="Tall Female",
            mode="simple",
            height_m=1.75,
            weight_kg=70.0
        )
        tall_female.compute_segment_lengths()
        profiles["tall_female"] = tall_female
        
        # Short male
        short_male = AnthropometryProfile(
            name="Short Male",
            mode="simple",
            height_m=1.65,
            weight_kg=70.0
        )
        short_male.compute_segment_lengths()
        profiles["short_male"] = short_male
        
        # Short female
        short_female = AnthropometryProfile(
            name="Short Female",
            mode="simple",
            height_m=1.55,
            weight_kg=55.0
        )
        short_female.compute_segment_lengths()
        profiles["short_female"] = short_female
        
        return profiles


# Anthropometric scaling equations from literature
def estimate_segment_mass(segment_name: str, total_mass_kg: float) -> float:
    """
    Estimate segment mass as percentage of total body mass.
    Based on de Leva (1996) adjustments to Zatsiorsky et al. (1990).
    
    Args:
        segment_name: Name of segment
        total_mass_kg: Total body mass in kg
        
    Returns:
        Segment mass in kg
    """
    # Mass ratios (as fraction of total body mass)
    mass_ratios = {
        'head': 0.0694,
        'trunk': 0.4346,  # Trunk + pelvis combined
        'pelvis': 0.1117,
        'upper_arm': 0.0271,
        'forearm': 0.0162,
        'hand': 0.0061,
        'upper_leg': 0.1416,
        'lower_leg': 0.0433,
        'foot': 0.0137,
    }
    
    # Simplify segment name
    base_name = segment_name.replace('_left', '').replace('_right', '')
    
    ratio = mass_ratios.get(base_name, 0.05)
    return total_mass_kg * ratio


# Example usage:
"""
# Create manager
anthro = AnthropometryManager()

# Simple mode (height/weight)
simple_profile = AnthropometryProfile(
    name="John Doe",
    mode="simple",
    height_m=1.80,
    weight_kg=85.0
)
simple_profile.compute_segment_lengths()
anthro.set_profile(simple_profile)

# Advanced mode (detailed measurements)
advanced_profile = AnthropometryProfile(
    name="Jane Smith",
    mode="advanced",
    overall_height=1.68,
    leg_length=0.88,
    foot_length=0.24,
    shoulder_width=0.38,
    wingspan=1.65,
    hip_width=0.28,
    c7_to_head_top=0.23
)
advanced_profile.compute_segment_lengths()
anthro.set_profile(advanced_profile)

# Get segment length
trunk_length = anthro.get_segment_length('trunk')

# Save profile
anthro.save_profile(simple_profile)

# Load profile
loaded = anthro.load_profile('John_Doe.json')
anthro.set_profile(loaded)
"""
