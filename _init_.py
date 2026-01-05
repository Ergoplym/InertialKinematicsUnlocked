"""Biomechanics System - Modular IMU-based biomechanics with functional calibration"""

__version__ = "1.0.0"
__author__ = "Biomech Team"

from .config import *
from .calibration import CalibrationManager
from .kinematics import KinematicsEngine
from .visualization import Skeleton3DCanvas, PlotManager
from .imu_stream import FakeIMUStream, IMUStreamFactory
from .main import BiomechApp

__all__ = [
    'BODY_MODEL',
    'SEGMENT_NAMES',
    'CalibrationManager',
    'KinematicsEngine',
    'Skeleton3DCanvas',
    'PlotManager',
    'FakeIMUStream',
    'IMUStreamFactory',
    'BiomechApp',
]