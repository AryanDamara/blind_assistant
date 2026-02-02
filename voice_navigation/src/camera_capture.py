"""
Camera Capture Module
---------------------
Handles video capture from webcam or video file.
Runs as a background thread, pushing frames to an internal queue.

Features:
- Supports webcam (int) or video file (str path)
- Internal queue with configurable buffer size
- Timestamps on each frame
- Auto-restart on failure
- Warmup frame handling
"""

import cv2
import time
import queue
import threading
import yaml
import os
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np


@dataclass
class FramePacket:
    """Container for a captured frame with metadata."""
    frame: np.ndarray          # The actual frame (BGR format)
    timestamp: float           # Capture time (time.time())
    frame_id: int              # Sequential frame counter
    width: int
    height: int


class CameraCapture:
    """
    Thread-safe camera capture with internal queue.
    
    Usage:
        camera = CameraCapture(config_path="config/settings.yaml")
        camera.start()
        
        while running:
            packet = camera.get_frame(timeout=1.0)
            if packet:
                process(packet.frame)
        
        camera.stop()
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize camera capture with configuration."""
        # Load configuration
        self._load_config(config_path)
        
        # Internal state
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self._buffer_size)
        self._frame_counter: int = 0
        
        # Thread control
        self._capture_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._stop_event: threading.Event = threading.Event()
        
        # Error tracking
        self._consecutive_errors: int = 0
        self._last_error: Optional[str] = None
        
    def _load_config(self, config_path: str) -> None:
        """Load camera settings from YAML config."""
        # Defaults
        self._source: Union[int, str] = 0
        self._width: int = 640
        self._height: int = 480
        self._fps: int = 30
        self._buffer_size: int = 2
        self._frame_skip: int = 3
        self._auto_restart: bool = True
        self._restart_delay: float = 2.0
        self._warmup_frames: int = 10
        
        # Load from file if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                cam_cfg = config.get('camera', {})
                
                self._source = cam_cfg.get('source', 0)
                self._width = cam_cfg.get('width', 640)
                self._height = cam_cfg.get('height', 480)
                self._fps = cam_cfg.get('fps', 30)
                self._buffer_size = cam_cfg.get('buffer_size', 2)
                self._frame_skip = cam_cfg.get('frame_skip', 3)
                self._auto_restart = cam_cfg.get('auto_restart', True)
                self._restart_delay = cam_cfg.get('restart_delay_sec', 2.0)
                self._warmup_frames = cam_cfg.get('warmup_frames', 10)
    
    def _init_capture(self) -> bool:
        """Initialize the video capture device."""
        try:
            # Release existing capture if any
            if self._cap is not None:
                self._cap.release()
            
            # Open capture source
            self._cap = cv2.VideoCapture(self._source)
            
            if not self._cap.isOpened():
                self._last_error = f"Failed to open camera source: {self._source}"
                print(f"[CameraCapture] ERROR: {self._last_error}")
                return False
            
            # Configure camera properties (only for webcam, not video files)
            if isinstance(self._source, int):
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._cap.set(cv2.CAP_PROP_FPS, self._fps)
                
                # Reduce buffer to minimize latency
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Perform warmup (skip first N frames for camera exposure adjustment)
            print(f"[CameraCapture] Warming up ({self._warmup_frames} frames)...")
            for i in range(self._warmup_frames):
                ret, _ = self._cap.read()
                if not ret:
                    print(f"[CameraCapture] WARNING: Warmup frame {i} failed")
            
            # Get actual resolution (may differ from requested)
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
            
            print(f"[CameraCapture] Initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            print(f"[CameraCapture] Source: {self._source}")
            
            self._consecutive_errors = 0
            return True
            
        except Exception as e:
            self._last_error = str(e)
            print(f"[CameraCapture] ERROR initializing: {e}")
            return False
    
    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        print("[CameraCapture] Capture thread started")
        
        while not self._stop_event.is_set():
            # Check if capture is valid
            if self._cap is None or not self._cap.isOpened():
                if self._auto_restart:
                    print(f"[CameraCapture] Attempting restart in {self._restart_delay}s...")
                    time.sleep(self._restart_delay)
                    if not self._init_capture():
                        self._consecutive_errors += 1
                        continue
                else:
                    print("[CameraCapture] Camera not available, stopping")
                    break
            
            # Read frame
            ret, frame = self._cap.read()
            timestamp = time.time()
            
            if not ret or frame is None:
                self._consecutive_errors += 1
                self._last_error = "Failed to read frame"
                
                # Check if video file ended
                if isinstance(self._source, str):
                    print("[CameraCapture] Video file ended")
                    # Optionally loop video
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                if self._consecutive_errors > 10:
                    print("[CameraCapture] Too many consecutive errors")
                    if self._auto_restart:
                        self._cap.release()
                        self._cap = None
                continue
            
            # Reset error counter on success
            self._consecutive_errors = 0
            
            # Create frame packet
            self._frame_counter += 1
            packet = FramePacket(
                frame=frame,
                timestamp=timestamp,
                frame_id=self._frame_counter,
                width=frame.shape[1],
                height=frame.shape[0]
            )
            
            # Add to queue (non-blocking, drop old frames if full)
            try:
                # If queue is full, remove oldest frame
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self._frame_queue.put_nowait(packet)
                
            except queue.Full:
                # This shouldn't happen after the above check, but just in case
                pass
        
        # Cleanup
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        print("[CameraCapture] Capture thread stopped")
    
    def start(self) -> bool:
        """
        Start the camera capture thread.
        
        Returns:
            bool: True if started successfully
        """
        if self._running:
            print("[CameraCapture] Already running")
            return True
        
        # Initialize capture
        if not self._init_capture():
            return False
        
        # Start capture thread
        self._stop_event.clear()
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureThread",
            daemon=True
        )
        self._capture_thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop the camera capture thread."""
        if not self._running:
            return
        
        print("[CameraCapture] Stopping...")
        self._stop_event.set()
        self._running = False
        
        # Wait for thread to finish
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None
        
        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        print("[CameraCapture] Stopped")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """
        Get the next frame from the queue.
        
        Args:
            timeout: Max seconds to wait for a frame
            
        Returns:
            FramePacket or None if no frame available
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest_frame(self) -> Optional[FramePacket]:
        """
        Get the most recent frame, discarding any older ones.
        
        Returns:
            FramePacket or None if no frame available
        """
        latest = None
        while True:
            try:
                latest = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        return latest
    
    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running
    
    @property
    def frame_skip(self) -> int:
        """Get configured frame skip value (for main loop to use)."""
        return self._frame_skip
    
    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error
    
    @property
    def queue_size(self) -> int:
        """Get current number of frames in queue."""
        return self._frame_queue.qsize()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# ============================================
# Testing / Demo
# ============================================
if __name__ == "__main__":
    """Test camera capture module standalone."""
    
    print("=" * 50)
    print("Camera Capture Module Test")
    print("=" * 50)
    
    # Use context manager for automatic cleanup
    with CameraCapture(config_path="config/settings.yaml") as camera:
        print(f"\nFrame skip setting: {camera.frame_skip}")
        print("Press 'q' to quit\n")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Get frame
            packet = camera.get_frame(timeout=1.0)
            
            if packet is None:
                print("No frame received")
                continue
            
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Display frame info
            cv2.putText(
                packet.frame,
                f"Frame: {packet.frame_id} | FPS: {fps:.1f} | Queue: {camera.queue_size}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow("Camera Test", packet.frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print("\nTest complete!")
