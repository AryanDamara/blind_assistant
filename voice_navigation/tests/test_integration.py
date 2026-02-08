#!/usr/bin/env python3
"""
Integration Tests
-----------------
Test full navigation pipeline with known scenarios.

Usage:
    pytest tests/test_integration.py -v
    python -m pytest tests/test_integration.py -v
"""

import pytest
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestModuleImports:
    """Test that all modules import correctly."""
    
    def test_camera_capture_import(self):
        from camera_capture import CameraCapture
        assert CameraCapture is not None
    
    def test_object_detector_import(self):
        from object_detector import ObjectDetector
        assert ObjectDetector is not None
    
    def test_safety_manager_import(self):
        from safety_manager import SafetyManager
        assert SafetyManager is not None
    
    def test_scene_analyzer_import(self):
        from scene_analyzer import SceneAnalyzer
        assert SceneAnalyzer is not None
    
    def test_audio_feedback_import(self):
        from audio_feedback import AudioFeedback
        assert AudioFeedback is not None
    
    def test_ai_assistant_import(self):
        from ai_assistant import AIAssistant
        assert AIAssistant is not None
    
    def test_telemetry_import(self):
        from telemetry import TelemetryLogger, LatencyMetrics
        assert TelemetryLogger is not None
        assert LatencyMetrics is not None


class TestConfigValidation:
    """Test configuration file validity."""
    
    @pytest.fixture
    def config(self):
        import yaml
        config_path = "config/settings.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_required_sections_exist(self, config):
        required = ['camera', 'yolo', 'safety', 'audio']
        for section in required:
            assert section in config, f"Missing section: {section}"
    
    def test_confidence_threshold_valid(self, config):
        conf = config.get('yolo', {}).get('confidence_threshold', 0)
        assert 0 < conf < 1, f"Invalid confidence: {conf}"
    
    def test_min_distance_positive(self, config):
        # Check nested danger_levels.critical.distance_m
        safety = config.get('safety', {})
        danger_levels = safety.get('danger_levels', {})
        critical = danger_levels.get('critical', {})
        min_dist = critical.get('distance_m', 0)
        assert min_dist > 0, f"Invalid critical distance: {min_dist}"


class TestPipelineIntegration:
    """Test full pipeline integration (requires camera)."""
    
    @pytest.fixture
    def pipeline(self):
        """Setup test pipeline."""
        from camera_capture import CameraCapture
        from object_detector import ObjectDetector
        from safety_manager import SafetyManager
        
        camera = CameraCapture(config_path="config/settings.yaml")
        detector = ObjectDetector(config_path="config/settings.yaml")
        safety = SafetyManager(config_path="config/settings.yaml")
        
        yield {
            'camera': camera,
            'detector': detector,
            'safety': safety
        }
        
        # Cleanup
        if camera._running:
            camera.stop()
    
    @pytest.mark.skipif(
        os.environ.get('CI') == 'true',
        reason="Camera not available in CI"
    )
    def test_camera_starts(self, pipeline):
        """Test camera initialization."""
        assert pipeline['camera'].start() == True
        pipeline['camera'].stop()
    
    @pytest.mark.skipif(
        os.environ.get('CI') == 'true',
        reason="Camera not available in CI"
    )
    def test_detection_pipeline(self, pipeline):
        """Test detection on a frame."""
        camera = pipeline['camera']
        detector = pipeline['detector']
        
        if not camera.start():
            pytest.skip("Camera not available")
        
        frame = camera.get_frame(timeout=2.0)
        assert frame is not None, "No frame received"
        
        result = detector.detect(frame.data, frame_id=1)
        assert result is not None
        assert hasattr(result, 'detections')
        assert hasattr(result, 'count')
        
        camera.stop()
    
    @pytest.mark.skipif(
        os.environ.get('CI') == 'true',
        reason="Camera not available in CI"
    )
    def test_latency_under_threshold(self, pipeline):
        """Verify end-to-end latency < 500ms."""
        camera = pipeline['camera']
        detector = pipeline['detector']
        safety = pipeline['safety']
        
        if not camera.start():
            pytest.skip("Camera not available")
        
        frame = camera.get_frame(timeout=2.0)
        if frame is None:
            camera.stop()
            pytest.skip("No frame available")
        
        start = time.time()
        
        det_result = detector.detect(frame.data, frame_id=1)
        safety_result = safety.analyze(
            det_result.detections,
            frame_width=frame.width,
            frame_id=1
        )
        
        latency_ms = (time.time() - start) * 1000
        camera.stop()
        
        assert latency_ms < 500, f"Latency {latency_ms:.0f}ms exceeds 500ms"


class TestSafetyManager:
    """Test SafetyManager functionality."""
    
    @pytest.fixture
    def safety(self):
        from safety_manager import SafetyManager
        return SafetyManager(config_path="config/settings.yaml")
    
    def test_empty_detections(self, safety):
        """Handle empty detection list."""
        result = safety.analyze([], frame_width=640, frame_id=1)
        assert result is not None
        assert len(result.alerts) == 0
    
    def test_stats_tracking(self, safety):
        """Verify stats are tracked."""
        safety.analyze([], frame_width=640, frame_id=1)
        stats = safety.get_stats()
        assert 'total_alerts' in stats
        assert 'frames_analyzed' in stats


class TestTelemetry:
    """Test telemetry module."""
    
    def test_latency_metrics_creation(self):
        from telemetry import LatencyMetrics
        
        metrics = LatencyMetrics(
            frame_id=1,
            timestamp=time.time(),
            detection_ms=50.0,
            safety_ms=5.0
        )
        
        assert metrics.frame_id == 1
        assert metrics.detection_ms == 50.0
    
    def test_latency_total_calculation(self):
        from telemetry import LatencyMetrics
        
        metrics = LatencyMetrics(
            frame_id=1,
            timestamp=time.time(),
            camera_ms=10.0,
            detection_ms=50.0,
            safety_ms=5.0,
            scene_ms=10.0,
            audio_ms=5.0
        )
        
        total = metrics.calculate_total()
        assert total == 80.0
    
    def test_session_manager(self):
        from telemetry import TelemetryLogger
        import shutil
        
        # Use temp directory
        logger = TelemetryLogger(config_path="config/settings.yaml")
        
        # Override log directory
        logger._log_directory = "data/logs/test_session"
        logger._session._base_log_dir = "data/logs/test_session"
        
        assert logger.start() == True
        
        from telemetry import LatencyMetrics
        metrics = LatencyMetrics(
            frame_id=1,
            timestamp=time.time(),
            detection_ms=50.0
        )
        logger.log_latency(metrics)
        
        stats = logger.stop()
        assert stats.total_frames == 1
        
        # Cleanup
        if os.path.exists("data/logs/test_session"):
            shutil.rmtree("data/logs/test_session")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
