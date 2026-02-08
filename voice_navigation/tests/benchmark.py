#!/usr/bin/env python3
"""
Benchmark Module
----------------
Performance testing and latency measurement for the navigation system.

Features:
- Module latency benchmarking
- FPS measurement with statistics
- Memory usage tracking
- Automated test reports (JSON/HTML)
- Stress testing support

Usage:
    python tests/benchmark.py                    # Quick benchmark (30 seconds)
    python tests/benchmark.py --duration 300     # 5-minute stress test
    python tests/benchmark.py --scenario crowded # Use specific video
    python tests/benchmark.py --report html      # Generate HTML report
"""

import os
import sys
import time
import json
import argparse
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[Benchmark] psutil not installed, memory tracking disabled")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    timestamp: str
    duration_sec: float
    
    # Frame statistics
    total_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    
    # FPS statistics
    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    
    # Latency statistics (ms)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    
    # Component latencies (ms)
    avg_camera_ms: float = 0.0
    avg_detection_ms: float = 0.0
    avg_safety_ms: float = 0.0
    avg_scene_ms: float = 0.0
    
    # Memory statistics (MB)
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0
    
    # Detection statistics
    total_detections: int = 0
    avg_detections_per_frame: float = 0.0
    
    # Errors
    error_count: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def passes_target(self, target_latency_ms: float = 500.0, 
                     target_fps: float = 10.0) -> bool:
        """Check if benchmark passes performance targets."""
        return (self.avg_latency_ms < target_latency_ms and 
                self.avg_fps >= target_fps and
                self.error_count == 0)


class ModuleBenchmark:
    """Benchmark individual modules."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self._latencies: deque = deque(maxlen=10000)
        self._fps_samples: deque = deque(maxlen=1000)
        self._memory_samples: deque = deque(maxlen=1000)
        
        self._component_times: Dict[str, deque] = {
            'camera': deque(maxlen=5000),
            'detection': deque(maxlen=5000),
            'safety': deque(maxlen=5000),
            'scene': deque(maxlen=5000)
        }
        
        self._total_detections: int = 0
        self._error_count: int = 0
        self._process = psutil.Process() if HAS_PSUTIL else None
    
    def benchmark_camera(self, duration_sec: float = 10.0) -> Dict:
        """Benchmark camera capture module."""
        from camera_capture import CameraCapture
        
        print(f"\n[Benchmark] Testing CameraCapture ({duration_sec}s)...")
        
        camera = CameraCapture(config_path=self.config_path)
        if not camera.start():
            return {'error': 'Failed to start camera'}
        
        frame_count = 0
        latencies = []
        start_time = time.time()
        
        while time.time() - start_time < duration_sec:
            frame = camera.get_frame(timeout=1.0)
            if frame is not None:
                frame_count += 1
                latencies.append(frame.get_age())
        
        camera.stop()
        
        return {
            'module': 'CameraCapture',
            'frames': frame_count,
            'fps': frame_count / duration_sec,
            'avg_latency_ms': statistics.mean(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0
        }
    
    def benchmark_detector(self, duration_sec: float = 10.0) -> Dict:
        """Benchmark object detection module."""
        from camera_capture import CameraCapture
        from object_detector import ObjectDetector
        
        print(f"\n[Benchmark] Testing ObjectDetector ({duration_sec}s)...")
        
        camera = CameraCapture(config_path=self.config_path)
        detector = ObjectDetector(config_path=self.config_path)
        
        if not camera.start():
            return {'error': 'Failed to start camera'}
        
        frame_count = 0
        detection_times = []
        total_detections = 0
        start_time = time.time()
        
        while time.time() - start_time < duration_sec:
            frame = camera.get_frame(timeout=1.0)
            if frame is not None:
                det_start = time.time()
                result = detector.detect(frame.data, frame_id=frame_count)
                det_time = (time.time() - det_start) * 1000
                
                frame_count += 1
                detection_times.append(det_time)
                total_detections += result.count
        
        camera.stop()
        
        return {
            'module': 'ObjectDetector',
            'frames': frame_count,
            'fps': frame_count / duration_sec,
            'avg_detection_ms': statistics.mean(detection_times) if detection_times else 0,
            'max_detection_ms': max(detection_times) if detection_times else 0,
            'p95_detection_ms': sorted(detection_times)[int(len(detection_times) * 0.95)] if detection_times else 0,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / max(1, frame_count)
        }
    
    def benchmark_full_pipeline(self, duration_sec: float = 30.0,
                                video_source: str = None) -> BenchmarkResult:
        """Benchmark the complete navigation pipeline."""
        from camera_capture import CameraCapture
        from object_detector import ObjectDetector
        from safety_manager import SafetyManager
        from scene_analyzer import SceneAnalyzer
        
        print(f"\n[Benchmark] Testing full pipeline ({duration_sec}s)...")
        print("=" * 60)
        
        # Initialize modules
        camera = CameraCapture(config_path=self.config_path)
        detector = ObjectDetector(config_path=self.config_path)
        safety = SafetyManager(config_path=self.config_path)
        scene = SceneAnalyzer(config_path=self.config_path)
        
        if not camera.start():
            result = BenchmarkResult(
                name="full_pipeline",
                timestamp=datetime.now().isoformat(),
                duration_sec=0
            )
            result.error_count = 1
            return result
        
        # Tracking variables
        frame_count = 0
        processed_count = 0
        frame_skip = camera.frame_skip
        
        latencies = []
        fps_times = []
        memory_samples = []
        last_fps_time = time.time()
        fps_frame_count = 0
        
        component_times = {
            'camera': [],
            'detection': [],
            'safety': [],
            'scene': []
        }
        
        total_detections = 0
        initial_memory = self._get_memory_mb()
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration_sec:
                frame_start = time.time()
                
                # Get frame
                cam_start = time.time()
                frame = camera.get_frame(timeout=1.0)
                cam_time = (time.time() - cam_start) * 1000
                
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Skip frames as configured
                if frame_count % (frame_skip + 1) != 0:
                    continue
                
                processed_count += 1
                component_times['camera'].append(frame.get_age())
                
                # Detection
                det_start = time.time()
                det_result = detector.detect(frame.data, frame_id=frame_count)
                det_time = (time.time() - det_start) * 1000
                component_times['detection'].append(det_time)
                total_detections += det_result.count
                
                # Safety analysis
                safety_start = time.time()
                safety_result = safety.analyze(
                    det_result.detections,
                    frame_width=frame.width,
                    frame_id=frame_count
                )
                safety_time = (time.time() - safety_start) * 1000
                component_times['safety'].append(safety_time)
                
                # Scene analysis
                scene_start = time.time()
                scene.analyze(
                    det_result.detections,
                    frame_id=frame_count,
                    frame_width=frame.width,
                    alerts=safety_result.alerts
                )
                scene_time = (time.time() - scene_start) * 1000
                component_times['scene'].append(scene_time)
                
                # Total latency
                total_latency = (time.time() - frame_start) * 1000
                latencies.append(total_latency)
                
                # FPS calculation (every second)
                fps_frame_count += 1
                if time.time() - last_fps_time >= 1.0:
                    fps_times.append(fps_frame_count / (time.time() - last_fps_time))
                    fps_frame_count = 0
                    last_fps_time = time.time()
                    
                    # Memory sample
                    memory_samples.append(self._get_memory_mb())
                
                # Progress update
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    print(f"[Benchmark] {elapsed:.0f}s / {duration_sec}s - "
                          f"Frames: {processed_count}, "
                          f"Avg latency: {statistics.mean(latencies[-100:]):.1f}ms")
        
        except Exception as e:
            print(f"[Benchmark] ERROR: {e}")
            self._error_count += 1
        
        finally:
            camera.stop()
        
        # Calculate results
        result = BenchmarkResult(
            name="full_pipeline",
            timestamp=datetime.now().isoformat(),
            duration_sec=time.time() - start_time
        )
        
        result.total_frames = frame_count
        result.processed_frames = processed_count
        result.dropped_frames = frame_count - processed_count
        
        if latencies:
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            
            result.avg_latency_ms = statistics.mean(latencies)
            result.min_latency_ms = min(latencies)
            result.max_latency_ms = max(latencies)
            result.std_latency_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
            result.p50_latency_ms = sorted_latencies[int(n * 0.50)]
            result.p95_latency_ms = sorted_latencies[int(n * 0.95)]
            result.p99_latency_ms = sorted_latencies[min(int(n * 0.99), n-1)]
        
        if fps_times:
            result.avg_fps = statistics.mean(fps_times)
            result.min_fps = min(fps_times)
            result.max_fps = max(fps_times)
        
        # Component averages
        for name, times in component_times.items():
            if times:
                setattr(result, f'avg_{name}_ms', statistics.mean(times))
        
        # Memory statistics
        if memory_samples:
            result.avg_memory_mb = statistics.mean(memory_samples)
            result.peak_memory_mb = max(memory_samples)
            result.memory_growth_mb = memory_samples[-1] - initial_memory
        
        result.total_detections = total_detections
        result.avg_detections_per_frame = total_detections / max(1, processed_count)
        result.error_count = self._error_count
        
        return result
    
    def _get_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        if self._process:
            return self._process.memory_info().rss / (1024 * 1024)
        return 0.0


def generate_report(result: BenchmarkResult, format: str = 'json',
                   output_path: str = None) -> str:
    """Generate benchmark report."""
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/logs/benchmark_{timestamp}.{format}"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    elif format == 'html':
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report - {result.timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
        .stat-card {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>🎯 Navigation System Benchmark Report</h1>
    <p><strong>Timestamp:</strong> {result.timestamp}</p>
    <p><strong>Duration:</strong> {result.duration_sec:.1f} seconds</p>
    <p><strong>Status:</strong> <span class="{'pass' if result.passes_target() else 'fail'}">
        {"✅ PASS" if result.passes_target() else "❌ FAIL"}
    </span></p>
    
    <h2>📊 Performance Summary</h2>
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{result.avg_fps:.1f}</div>
            <div class="stat-label">Average FPS</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{result.avg_latency_ms:.0f}ms</div>
            <div class="stat-label">Average Latency</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{result.p99_latency_ms:.0f}ms</div>
            <div class="stat-label">P99 Latency</div>
        </div>
    </div>
    
    <h2>🔧 Component Latencies</h2>
    <table>
        <tr>
            <th>Component</th>
            <th>Average (ms)</th>
        </tr>
        <tr><td>Camera</td><td>{result.avg_camera_ms:.1f}</td></tr>
        <tr><td>Detection</td><td>{result.avg_detection_ms:.1f}</td></tr>
        <tr><td>Safety</td><td>{result.avg_safety_ms:.1f}</td></tr>
        <tr><td>Scene</td><td>{result.avg_scene_ms:.1f}</td></tr>
    </table>
    
    <h2>💾 Memory Usage</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr><td>Average</td><td>{result.avg_memory_mb:.1f} MB</td></tr>
        <tr><td>Peak</td><td>{result.peak_memory_mb:.1f} MB</td></tr>
        <tr><td>Growth</td><td>{result.memory_growth_mb:+.1f} MB</td></tr>
    </table>
    
    <h2>📈 Frame Statistics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr><td>Total Frames</td><td>{result.total_frames}</td></tr>
        <tr><td>Processed Frames</td><td>{result.processed_frames}</td></tr>
        <tr><td>Detections</td><td>{result.total_detections}</td></tr>
        <tr><td>Avg Detections/Frame</td><td>{result.avg_detections_per_frame:.1f}</td></tr>
    </table>
</body>
</html>
"""
        with open(output_path, 'w') as f:
            f.write(html)
    
    print(f"\n[Benchmark] Report saved to: {output_path}")
    return output_path


def print_results(result: BenchmarkResult) -> None:
    """Print benchmark results to console."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    status = "✅ PASS" if result.passes_target() else "❌ FAIL"
    print(f"Status: {status}")
    print(f"Duration: {result.duration_sec:.1f}s")
    
    print(f"\n📊 Performance:")
    print(f"  Average FPS: {result.avg_fps:.1f}")
    print(f"  Average Latency: {result.avg_latency_ms:.1f}ms")
    print(f"  P50 Latency: {result.p50_latency_ms:.1f}ms")
    print(f"  P95 Latency: {result.p95_latency_ms:.1f}ms")
    print(f"  P99 Latency: {result.p99_latency_ms:.1f}ms")
    
    print(f"\n🔧 Component Latencies (avg):")
    print(f"  Camera: {result.avg_camera_ms:.1f}ms")
    print(f"  Detection: {result.avg_detection_ms:.1f}ms")
    print(f"  Safety: {result.avg_safety_ms:.1f}ms")
    print(f"  Scene: {result.avg_scene_ms:.1f}ms")
    
    print(f"\n💾 Memory:")
    print(f"  Average: {result.avg_memory_mb:.1f}MB")
    print(f"  Peak: {result.peak_memory_mb:.1f}MB")
    print(f"  Growth: {result.memory_growth_mb:+.1f}MB")
    
    print(f"\n📈 Frames:")
    print(f"  Total: {result.total_frames}")
    print(f"  Processed: {result.processed_frames}")
    print(f"  Detections: {result.total_detections}")
    
    if result.error_count > 0:
        print(f"\n⚠️ Errors: {result.error_count}")
    
    print("=" * 60)


def compare_to_baseline(current: BenchmarkResult, 
                       baseline_path: str = "data/baseline.json") -> Dict:
    """Compare current results to baseline for regression detection."""
    if not os.path.exists(baseline_path):
        print(f"\n[Baseline] No baseline found, saving current as baseline")
        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        with open(baseline_path, 'w') as f:
            json.dump(current.to_dict(), f, indent=2)
        return {'status': 'baseline_created'}
    
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    comparison = {
        'latency_change_ms': current.avg_latency_ms - baseline.get('avg_latency_ms', 0),
        'fps_change': current.avg_fps - baseline.get('avg_fps', 0),
        'memory_change_mb': current.avg_memory_mb - baseline.get('avg_memory_mb', 0),
        'p99_change_ms': current.p99_latency_ms - baseline.get('p99_latency_ms', 0)
    }
    
    print("\n📊 Comparison to Baseline:")
    print(f"  Latency: {comparison['latency_change_ms']:+.0f}ms")
    print(f"  P99 Latency: {comparison['p99_change_ms']:+.0f}ms")
    print(f"  FPS: {comparison['fps_change']:+.1f}")
    print(f"  Memory: {comparison['memory_change_mb']:+.0f}MB")
    
    # Check for regression
    regressions = []
    if comparison['latency_change_ms'] > 50:
        regressions.append(f"Latency increased by {comparison['latency_change_ms']:.0f}ms")
    if comparison['fps_change'] < -2:
        regressions.append(f"FPS dropped by {-comparison['fps_change']:.1f}")
    if comparison['memory_change_mb'] > 100:
        regressions.append(f"Memory increased by {comparison['memory_change_mb']:.0f}MB")
    
    if regressions:
        print("\n⚠️ PERFORMANCE REGRESSIONS DETECTED:")
        for r in regressions:
            print(f"   • {r}")
        comparison['regressions'] = regressions
    else:
        print("\n✅ No significant regressions")
    
    return comparison


def main():
    """Run benchmark from command line."""
    parser = argparse.ArgumentParser(description="Navigation System Benchmark")
    parser.add_argument('--duration', type=int, default=30,
                       help='Benchmark duration in seconds (default: 30)')
    parser.add_argument('--scenario', type=str, default=None,
                       help='Video scenario file to use')
    parser.add_argument('--report', type=str, choices=['json', 'html'], default='json',
                       help='Output report format (default: json)')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                       help='Path to config file')
    parser.add_argument('--module', type=str, choices=['camera', 'detector', 'full'],
                       default='full', help='Module to benchmark')
    parser.add_argument('--baseline', action='store_true',
                       help='Compare results to baseline')
    parser.add_argument('--save-baseline', action='store_true',
                       help='Save current results as new baseline')
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("=" * 60)
    print("Navigation System Benchmark")
    print("=" * 60)
    print(f"Duration: {args.duration}s")
    print(f"Config: {args.config}")
    print(f"Module: {args.module}")
    
    benchmark = ModuleBenchmark(config_path=args.config)
    
    if args.module == 'camera':
        result = benchmark.benchmark_camera(duration_sec=args.duration)
        print(f"\n{json.dumps(result, indent=2)}")
    
    elif args.module == 'detector':
        result = benchmark.benchmark_detector(duration_sec=args.duration)
        print(f"\n{json.dumps(result, indent=2)}")
    
    else:  # full
        result = benchmark.benchmark_full_pipeline(duration_sec=args.duration)
        print_results(result)
        generate_report(result, format=args.report)
        
        # Baseline comparison
        if args.baseline:
            compare_to_baseline(result)
        
        # Save as baseline
        if args.save_baseline:
            baseline_path = "data/baseline.json"
            os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
            with open(baseline_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\n[Baseline] Saved to {baseline_path}")


if __name__ == "__main__":
    main()

