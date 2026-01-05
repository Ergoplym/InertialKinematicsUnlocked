"""
Webcam recording system for BushBiomech.
Handles multi-camera recording synchronized with IMU data.
"""

import cv2
import numpy as np
import threading
import time
import json
import os
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CameraInfo:
    """Information about a connected camera."""
    camera_id: int
    name: str
    resolution: Tuple[int, int]
    fps: float
    is_available: bool = True


class WebcamRecorder:
    """Manages multi-camera recording synchronized with IMU timestamps."""
    
    def __init__(self, timestamp_manager=None):
        """
        Args:
            timestamp_manager: Object with get_timestamp() method for sync.
                              If None, creates internal timestamp source.
        """
        self.timestamp_manager = timestamp_manager or self._create_timestamp_manager()
        
        # Camera management
        self.cameras: Dict[int, cv2.VideoCapture] = {}
        self.camera_info: Dict[int, CameraInfo] = {}
        self.preview_frames: Dict[int, np.ndarray] = {}
        
        # Recording state
        self.is_recording = False
        self.video_writers: Dict[int, cv2.VideoWriter] = {}
        self.timestamp_files: Dict[int, object] = {}
        self.frame_counts: Dict[int, int] = {}
        
        # Threading
        self.capture_threads: Dict[int, threading.Thread] = {}
        self.stop_flags: Dict[int, threading.Event] = {}
        self.frame_locks: Dict[int, threading.Lock] = {}
        
        # Recording session info
        self.session_folder = None
        self.session_metadata = {}
    
    def _create_timestamp_manager(self):
        """Create internal timestamp manager if none provided."""
        class SimpleTimestampManager:
            def __init__(self):
                self.start_time = time.monotonic()
            
            def get_timestamp(self):
                return time.monotonic() - self.start_time
        
        return SimpleTimestampManager()
    
    def detect_cameras(self, max_cameras: int = 4) -> List[CameraInfo]:
        """
        Detect available cameras.
        
        Args:
            max_cameras: Maximum number of cameras to detect
            
        Returns:
            List of CameraInfo objects for available cameras
        """
        available_cameras = []
        
        print("[Webcam] Detecting cameras...")
        for camera_id in range(max_cameras):
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    # Try to read a frame to verify it works
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Some cameras report 0 FPS, default to 30
                        if fps == 0:
                            fps = 30.0
                        
                        info = CameraInfo(
                            camera_id=camera_id,
                            name=f"Camera {camera_id}",
                            resolution=(width, height),
                            fps=fps,
                            is_available=True
                        )
                        available_cameras.append(info)
                        print(f"[Webcam] Found: {info.name} @ {width}x{height}, {fps} fps")
                    
                    cap.release()
            except Exception as e:
                print(f"[Webcam] Error checking camera {camera_id}: {e}")
                continue
        
        print(f"[Webcam] Detected {len(available_cameras)} camera(s)")
        return available_cameras
    
    def open_camera(self, camera_id: int, target_fps: int = 60) -> bool:
        """
        Open a camera for preview and recording.
        
        Args:
            camera_id: Camera index (0, 1, 2, ...)
            target_fps: Desired FPS (will try to set, may not be supported)
            
        Returns:
            True if successful
        """
        if camera_id in self.cameras:
            print(f"[Webcam] Camera {camera_id} already open")
            return True
        
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print(f"[Webcam] Failed to open camera {camera_id}")
                return False
            
            # Try to set target FPS (may not work on all cameras)
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            
            # Get actual properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if actual_fps == 0:
                actual_fps = 30.0  # Default fallback
            
            # Store camera info
            self.cameras[camera_id] = cap
            self.camera_info[camera_id] = CameraInfo(
                camera_id=camera_id,
                name=f"Camera {camera_id}",
                resolution=(width, height),
                fps=actual_fps
            )
            self.frame_locks[camera_id] = threading.Lock()
            self.preview_frames[camera_id] = None
            
            print(f"[Webcam] Opened camera {camera_id}: {width}x{height} @ {actual_fps} fps")
            
            # Start capture thread
            self._start_capture_thread(camera_id)
            
            return True
            
        except Exception as e:
            print(f"[Webcam] Error opening camera {camera_id}: {e}")
            return False
    
    def _start_capture_thread(self, camera_id: int):
        """Start background thread to capture frames from camera."""
        if camera_id in self.capture_threads and self.capture_threads[camera_id].is_alive():
            return
        
        self.stop_flags[camera_id] = threading.Event()
        thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_id,),
            daemon=True
        )
        self.capture_threads[camera_id] = thread
        thread.start()
        print(f"[Webcam] Started capture thread for camera {camera_id}")
    
    def _capture_loop(self, camera_id: int):
        """Background thread loop to continuously capture frames."""
        cap = self.cameras[camera_id]
        stop_flag = self.stop_flags[camera_id]
        frame_lock = self.frame_locks[camera_id]
        
        last_frame_time = time.monotonic()
        frame_interval = 1.0 / 60.0  # Target 60 fps
        
        while not stop_flag.is_set():
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"[Webcam] Failed to read from camera {camera_id}")
                    time.sleep(0.01)
                    continue
                
                # Get timestamp
                timestamp = self.timestamp_manager.get_timestamp()
                
                # Update preview frame
                with frame_lock:
                    self.preview_frames[camera_id] = frame.copy()
                
                # If recording, write frame
                if self.is_recording and camera_id in self.video_writers:
                    try:
                        writer = self.video_writers[camera_id]
                        writer.write(frame)
                        
                        # Write timestamp
                        ts_file = self.timestamp_files[camera_id]
                        ts_file.write(f"{self.frame_counts[camera_id]},{timestamp:.6f}\n")
                        
                        self.frame_counts[camera_id] += 1
                        
                        # Check for frame drops
                        current_time = time.monotonic()
                        if current_time - last_frame_time > frame_interval * 2:
                            print(f"[Webcam] Warning: Possible frame drop on camera {camera_id}")
                        last_frame_time = current_time
                        
                    except Exception as e:
                        print(f"[Webcam] Error writing frame from camera {camera_id}: {e}")
                
                # Limit capture rate
                time.sleep(0.001)  # Small sleep to prevent CPU hogging
                
            except Exception as e:
                print(f"[Webcam] Error in capture loop for camera {camera_id}: {e}")
                time.sleep(0.1)
    
    def get_preview_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """
        Get latest preview frame from camera.
        
        Args:
            camera_id: Camera index
            
        Returns:
            Frame as numpy array or None
        """
        if camera_id not in self.frame_locks:
            return None
        
        with self.frame_locks[camera_id]:
            if self.preview_frames[camera_id] is not None:
                return self.preview_frames[camera_id].copy()
        
        return None
    
    def start_recording(self, session_folder: str, target_fps: float = None, 
                       preferred_codec: str = None, quality: str = 'high',
                       filename_prefix: str = None):
        """
        Start recording from all open cameras.
        
        Args:
            session_folder: Folder to save recordings
            target_fps: Target recording FPS (default: use camera's native FPS)
            preferred_codec: Force specific codec ('H.265', 'H.264', 'MPEG-4', 'MJPEG')
            quality: Compression quality - 'high', 'medium', 'low'
            filename_prefix: Prefix for video files (e.g., "P001_Baseline_20260103_143022")
                           If None, uses default "camera_X.mp4"
        """
        if self.is_recording:
            print("[Webcam] Already recording")
            return
        
        if not self.cameras:
            print("[Webcam] No cameras open")
            return
        
        # Create session folder
        self.session_folder = session_folder
        os.makedirs(session_folder, exist_ok=True)
        
        print(f"[Webcam] Starting recording to: {session_folder}")
        if target_fps:
            print(f"[Webcam] Target FPS: {target_fps}, Quality: {quality}")
        else:
            print(f"[Webcam] Using camera native FPS, Quality: {quality}")
        if preferred_codec:
            print(f"[Webcam] Preferred codec: {preferred_codec}")
        
        # Determine codec order based on preference
        if preferred_codec == 'H.265':
            # H.265 software encoders (bypass Windows Media Foundation)
            codec_options = [
                ('X265', 'H.265/x265'),     # x265 software encoder - best compression
                ('mp4v', 'MPEG-4'),         # Fallback
                ('MJPG', 'MJPEG'),          # Last resort
            ]
        elif preferred_codec == 'H.264':
            # H.264 software encoders
            codec_options = [
                ('X264', 'H.264/x264'),     # x264 software encoder
                ('mp4v', 'MPEG-4'),         # Fallback
                ('MJPG', 'MJPEG'),          # Last resort
            ]
        elif preferred_codec == 'MPEG-4':
            codec_options = [
                ('mp4v', 'MPEG-4'),
                ('MJPG', 'MJPEG'),
            ]
        elif preferred_codec == 'MJPEG':
            codec_options = [
                ('MJPG', 'MJPEG'),
            ]
        else:
            # Default: Try software encoders first (avoid Windows MF threading issues)
            codec_options = [
                ('X265', 'H.265/x265'),     # Best compression, software
                ('X264', 'H.264/x264'),     # Good compression, software
                ('mp4v', 'MPEG-4'),         # Moderate compression
                ('MJPG', 'MJPEG'),          # Largest files, most compatible
            ]
        
        # Initialize writers for each camera
        for camera_id, info in self.camera_info.items():
            try:
                # Use camera's actual FPS if target_fps not specified
                recording_fps = target_fps if target_fps else info.fps
                
                # Video file with custom naming (auto-increment if exists)
                if filename_prefix:
                    base_filename = f"{filename_prefix}_camera_{camera_id}"
                else:
                    base_filename = f"camera_{camera_id}"
                
                # Check if file exists and increment if needed
                video_path = os.path.join(session_folder, f"{base_filename}.mp4")
                ts_path = os.path.join(session_folder, f"{base_filename}_timestamps.txt")
                
                counter = 1
                while os.path.exists(video_path) or os.path.exists(ts_path):
                    video_path = os.path.join(session_folder, f"{base_filename}_{counter}.mp4")
                    ts_path = os.path.join(session_folder, f"{base_filename}_{counter}_timestamps.txt")
                    counter += 1
                    
                    # Safety limit
                    if counter > 100:
                        raise RuntimeError(f"Too many video files for camera {camera_id} (>100)")
                
                if counter > 1:
                    print(f"[Webcam] File exists, using _{counter-1} suffix for camera {camera_id}")
                
                
                writer = None
                used_codec = None
                tried_codecs = []
                
                for fourcc_str, codec_name in codec_options:
                    tried_codecs.append(codec_name)
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                        test_writer = cv2.VideoWriter(
                            video_path,
                            fourcc,
                            recording_fps,  # Use actual camera FPS
                            info.resolution
                        )
                        
                        # Extra check: try to write a test frame
                        if test_writer.isOpened():
                            # Create dummy frame
                            test_frame = np.zeros((info.resolution[1], info.resolution[0], 3), dtype=np.uint8)
                            test_writer.write(test_frame)
                            
                            # If we got here, codec works!
                            writer = test_writer
                            used_codec = codec_name
                            print(f"[Webcam] Camera {camera_id}: ✓ Using {codec_name} ({fourcc_str}) @ {recording_fps:.1f} FPS")
                            break
                        else:
                            test_writer.release()
                    except Exception as e:
                        # Codec failed, try next
                        continue
                
                if writer is None or not writer.isOpened():
                    print(f"[Webcam] ✗ ERROR: No working codec for camera {camera_id}")
                    print(f"[Webcam] Tried: {', '.join(tried_codecs)}")
                    print(f"[Webcam] SOLUTION: Install ffmpeg with x264/x265 support")
                    print(f"[Webcam]   Download: https://github.com/GyanD/codecs/releases")
                    continue
                
                self.video_writers[camera_id] = writer
                
                # Store codec name for metadata
                if not hasattr(self, 'used_codecs'):
                    self.used_codecs = {}
                self.used_codecs[camera_id] = used_codec
                
                # Open timestamp file (path already determined above with auto-increment)
                ts_file = open(ts_path, 'w')
                ts_file.write("frame_number,timestamp\n")
                self.timestamp_files[camera_id] = ts_file
                
                # Frame counter
                self.frame_counts[camera_id] = 0
                
                print(f"[Webcam] Recording camera {camera_id} to {video_path}")
                
            except Exception as e:
                print(f"[Webcam] Error starting recording for camera {camera_id}: {e}")
        
        self.is_recording = True
        
        # Save metadata
        self._save_metadata(quality)
    
    def stop_recording(self):
        """Stop recording from all cameras."""
        if not self.is_recording:
            return
        
        print("[Webcam] Stopping recording...")
        self.is_recording = False
        
        # Close video writers and timestamp files
        for camera_id in list(self.video_writers.keys()):
            try:
                self.video_writers[camera_id].release()
                self.timestamp_files[camera_id].close()
                print(f"[Webcam] Saved {self.frame_counts[camera_id]} frames from camera {camera_id}")
            except Exception as e:
                print(f"[Webcam] Error stopping recording for camera {camera_id}: {e}")
        
        self.video_writers.clear()
        self.timestamp_files.clear()
        self.frame_counts.clear()
        
        print("[Webcam] Recording stopped")
    
    def _save_metadata(self, quality: str):
        """Save session metadata to JSON file."""
        metadata = {
            "session_folder": self.session_folder,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quality_setting": quality,
            "cameras": []
        }
        
        for camera_id, info in self.camera_info.items():
            codec_name = self.used_codecs.get(camera_id, "Unknown") if hasattr(self, 'used_codecs') else "Unknown"
            metadata["cameras"].append({
                "id": camera_id,
                "name": info.name,
                "resolution": list(info.resolution),
                "fps": info.fps,
                "codec": codec_name,
                "quality": quality
            })
        
        metadata_path = os.path.join(self.session_folder, "camera_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[Webcam] Metadata saved to {metadata_path}")
    
    def close_camera(self, camera_id: int):
        """Close a specific camera."""
        # Stop capture thread
        if camera_id in self.stop_flags:
            self.stop_flags[camera_id].set()
        
        if camera_id in self.capture_threads:
            self.capture_threads[camera_id].join(timeout=1.0)
        
        # Release camera
        if camera_id in self.cameras:
            self.cameras[camera_id].release()
            del self.cameras[camera_id]
        
        # Clean up
        if camera_id in self.camera_info:
            del self.camera_info[camera_id]
        if camera_id in self.preview_frames:
            del self.preview_frames[camera_id]
        if camera_id in self.frame_locks:
            del self.frame_locks[camera_id]
        
        print(f"[Webcam] Closed camera {camera_id}")
    
    def close_all(self):
        """Close all cameras and stop recording."""
        if self.is_recording:
            self.stop_recording()
        
        for camera_id in list(self.cameras.keys()):
            self.close_camera(camera_id)
        
        print("[Webcam] All cameras closed")
