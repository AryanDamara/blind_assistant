"""
Audio Feedback Module
---------------------
Text-to-speech audio output for navigation alerts.

Features:
- pyttsx3 TTS engine
- Priority queue for speech (higher priority speaks first)
- Interrupt on critical alerts
- Voice profiles (urgent/alert/calm)
- Configurable max announcements per cycle
- Non-blocking speech in background thread
"""

import time
import queue
import threading
import yaml
import os
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    import pyttsx3
except ImportError:
    print("[AudioFeedback] ERROR: pyttsx3 not installed. Run: pip install pyttsx3")
    raise


@dataclass
class SpeechItem:
    """Item in the speech queue."""
    text: str
    priority: int           # Higher = more urgent
    voice_profile: str      # 'urgent', 'alert', 'calm'
    timestamp: float        # When queued
    
    def __lt__(self, other):
        """For priority queue comparison (higher priority first)."""
        return self.priority > other.priority


class AudioFeedback:
    """
    Text-to-speech audio feedback with priority queue.
    
    Usage:
        audio = AudioFeedback(config_path="config/settings.yaml")
        audio.start()
        
        audio.speak("Person ahead", priority=10, voice_profile="alert")
        
        # Or use with alerts from SafetyManager:
        audio.announce_alerts(safety_result.alerts)
        
        audio.stop()
    """
    
    # Critical priority threshold for interruption
    CRITICAL_PRIORITY_THRESHOLD = 20
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize audio feedback with configuration."""
        self._load_config(config_path)
        
        # TTS engine (initialized in start())
        self._engine: Optional[pyttsx3.Engine] = None
        
        # Speech queue (priority queue)
        self._speech_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=20)
        
        # Thread control
        self._speech_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._stop_event: threading.Event = threading.Event()
        
        # State
        self._is_speaking: bool = False
        self._current_text: str = ""
        self._should_interrupt: bool = False
        
        # Statistics
        self._total_announcements: int = 0
        self._interrupted_count: int = 0
        self._queue_drops: int = 0
    
    def _load_config(self, config_path: str) -> None:
        """Load audio settings from YAML config."""
        # Defaults
        self._enabled: bool = True
        self._voice_profiles: Dict = {
            'urgent': {'rate': 200, 'volume': 1.0},
            'alert': {'rate': 175, 'volume': 0.9},
            'calm': {'rate': 150, 'volume': 0.8}
        }
        self._verbosity: str = "standard"
        self._interrupt_on_critical: bool = True
        self._max_alerts_per_cycle: int = 3
        self._voice_index: int = 0
        
        # Load from file if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                audio_cfg = config.get('audio', {})
                
                self._enabled = audio_cfg.get('enabled', True)
                self._verbosity = audio_cfg.get('verbosity', 'standard')
                self._interrupt_on_critical = audio_cfg.get('interrupt_on_critical', True)
                self._max_alerts_per_cycle = audio_cfg.get('max_alerts_per_cycle', 3)
                self._voice_index = audio_cfg.get('voice_index', 0)
                
                # Voice profiles
                profiles_cfg = audio_cfg.get('voice_profiles', {})
                for profile in ['urgent', 'alert', 'calm']:
                    if profile in profiles_cfg:
                        self._voice_profiles[profile] = {
                            'rate': profiles_cfg[profile].get('rate', self._voice_profiles[profile]['rate']),
                            'volume': profiles_cfg[profile].get('volume', self._voice_profiles[profile]['volume'])
                        }
        
        print(f"[AudioFeedback] Initialized")
        print(f"[AudioFeedback] Enabled: {self._enabled}")
        print(f"[AudioFeedback] Verbosity: {self._verbosity}")
        print(f"[AudioFeedback] Interrupt on critical: {self._interrupt_on_critical}")
        print(f"[AudioFeedback] Max alerts per cycle: {self._max_alerts_per_cycle}")
    
    def _init_engine(self) -> bool:
        """Initialize pyttsx3 engine."""
        try:
            self._engine = pyttsx3.init()
            
            # Set voice
            voices = self._engine.getProperty('voices')
            if voices and len(voices) > self._voice_index:
                self._engine.setProperty('voice', voices[self._voice_index].id)
                print(f"[AudioFeedback] Voice: {voices[self._voice_index].name}")
            
            # Set default properties
            self._apply_voice_profile('calm')
            
            print("[AudioFeedback] TTS engine initialized")
            return True
            
        except Exception as e:
            print(f"[AudioFeedback] ERROR initializing TTS: {e}")
            return False
    
    def _apply_voice_profile(self, profile: str) -> None:
        """Apply voice profile settings to engine."""
        if self._engine is None:
            return
        
        settings = self._voice_profiles.get(profile, self._voice_profiles['calm'])
        self._engine.setProperty('rate', settings['rate'])
        self._engine.setProperty('volume', settings['volume'])
    
    def _speech_loop(self) -> None:
        """Main speech loop running in background thread."""
        print("[AudioFeedback] Speech thread started")
        
        while not self._stop_event.is_set():
            try:
                # Get next item from queue (with timeout for responsive shutdown)
                try:
                    item: SpeechItem = self._speech_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Check if item is stale (queued > 5 seconds ago)
                age = time.time() - item.timestamp
                if age > 5.0:
                    print(f"[AudioFeedback] Dropping stale speech: {item.text[:30]}...")
                    self._queue_drops += 1
                    continue
                
                # Speak the item
                self._speak_item(item)
                
            except Exception as e:
                print(f"[AudioFeedback] ERROR in speech loop: {e}")
        
        print("[AudioFeedback] Speech thread stopped")
    
    def _speak_item(self, item: SpeechItem) -> None:
        """Speak a single item."""
        if self._engine is None or not self._enabled:
            return
        
        try:
            self._is_speaking = True
            self._current_text = item.text
            
            # Apply voice profile
            self._apply_voice_profile(item.voice_profile)
            
            # Speak
            self._engine.say(item.text)
            self._engine.runAndWait()
            
            self._total_announcements += 1
            self._is_speaking = False
            self._current_text = ""
            
        except Exception as e:
            print(f"[AudioFeedback] ERROR speaking: {e}")
            self._is_speaking = False
    
    def start(self) -> bool:
        """
        Start the audio feedback thread.
        
        Returns:
            bool: True if started successfully
        """
        if self._running:
            print("[AudioFeedback] Already running")
            return True
        
        if not self._enabled:
            print("[AudioFeedback] Audio disabled in config")
            return False
        
        # Initialize TTS engine
        if not self._init_engine():
            return False
        
        # Start speech thread
        self._stop_event.clear()
        self._running = True
        self._speech_thread = threading.Thread(
            target=self._speech_loop,
            name="AudioFeedbackThread",
            daemon=True
        )
        self._speech_thread.start()
        
        # Speak startup message
        self.speak("Navigation system ready", priority=1, voice_profile="calm")
        
        return True
    
    def stop(self) -> None:
        """Stop the audio feedback thread."""
        if not self._running:
            return
        
        print("[AudioFeedback] Stopping...")
        
        # Speak shutdown message
        self.speak("Navigation system stopping", priority=1, voice_profile="calm", bypass_queue=True)
        
        self._stop_event.set()
        self._running = False
        
        # Wait for thread
        if self._speech_thread is not None:
            self._speech_thread.join(timeout=2.0)
            self._speech_thread = None
        
        # Cleanup engine
        if self._engine is not None:
            try:
                self._engine.stop()
            except:
                pass
            self._engine = None
        
        # Print statistics
        stats = self.get_stats()
        print(f"""
[AudioFeedback] Session Statistics:
  Total announcements: {stats['total_announcements']}
  Interrupted: {stats['interrupted_count']}
  Queue drops: {stats['queue_drops']}
        """)
        
        print("[AudioFeedback] Stopped")
    
    def speak(self, text: str, priority: int = 1, voice_profile: str = "calm",
              bypass_queue: bool = False) -> bool:
        """
        Queue text for speech.
        
        Args:
            text: Text to speak
            priority: Priority level (higher = more urgent)
            voice_profile: 'urgent', 'alert', or 'calm'
            bypass_queue: If True, speak immediately (blocking)
            
        Returns:
            bool: True if queued/spoken successfully
        """
        if not self._enabled:
            return False
        
        # Handle critical interruption
        if self._interrupt_on_critical and priority >= self.CRITICAL_PRIORITY_THRESHOLD:
            if self._is_speaking:
                self._interrupt_current()
        
        if bypass_queue:
            # Speak immediately (blocking)
            item = SpeechItem(text=text, priority=priority, 
                            voice_profile=voice_profile, timestamp=time.time())
            self._speak_item(item)
            return True
        
        # Add to priority queue
        try:
            item = SpeechItem(
                text=text,
                priority=priority,
                voice_profile=voice_profile,
                timestamp=time.time()
            )
            self._speech_queue.put_nowait(item)
            return True
        except queue.Full:
            print(f"[AudioFeedback] Queue full, dropping: {text[:30]}...")
            self._queue_drops += 1
            return False
    
    def _interrupt_current(self) -> None:
        """Interrupt current speech for critical alert."""
        if self._engine is not None:
            try:
                self._engine.stop()
                self._interrupted_count += 1
                print(f"[AudioFeedback] ⚠️  Interrupted for critical alert")
            except:
                pass
    
    def announce_alerts(self, alerts: List) -> int:
        """
        Announce alerts from SafetyManager.
        
        Args:
            alerts: List of Alert objects (already sorted by priority)
            
        Returns:
            Number of alerts announced
        """
        if not self._enabled or not alerts:
            return 0
        
        # Take top N alerts
        alerts_to_announce = alerts[:self._max_alerts_per_cycle]
        announced = 0
        
        for alert in alerts_to_announce:
            # Get announcement text
            if hasattr(alert, 'get_announcement'):
                text = alert.get_announcement(self._verbosity)
            else:
                # Dict fallback
                text = f"{alert.get('class_name', 'object')} {alert.get('zone', 'ahead')}"
            
            # Get priority, voice profile, and danger level
            if hasattr(alert, 'priority'):
                priority = alert.priority
                voice_profile = alert.voice_profile
                danger_level = getattr(alert, 'danger_level', None)
            else:
                priority = alert.get('priority', 1)
                voice_profile = alert.get('voice_profile', 'calm')
                danger_level = alert.get('danger_level', None)
            
            # Check danger_level for critical interruption
            if danger_level == 'critical' and self._interrupt_on_critical:
                # Use very high priority to ensure interruption
                priority = 100
                text = f"Caution! {text}"  # Add emphasis
            
            # Queue the announcement
            if self.speak(text, priority=priority, voice_profile=voice_profile):
                announced += 1
        
        return announced
    
    def announce_critical(self, text: str) -> None:
        """
        Announce a critical alert immediately with interruption.
        
        Args:
            text: Critical message to announce
        """
        # Add "STOP" prefix for critical alerts
        critical_text = f"STOP! {text}"
        self.speak(critical_text, priority=100, voice_profile="urgent")
    
    def get_stats(self) -> dict:
        """Get audio feedback statistics."""
        return {
            'total_announcements': self._total_announcements,
            'interrupted_count': self._interrupted_count,
            'queue_drops': self._queue_drops,
            'queue_size': self._speech_queue.qsize(),
            'is_speaking': self._is_speaking
        }
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking
    
    @property
    def is_running(self) -> bool:
        """Check if audio is running."""
        return self._running
    
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
    """Test audio feedback module standalone."""
    
    print("=" * 50)
    print("Audio Feedback Module Test")
    print("=" * 50)
    
    # Create mock alerts for testing
    class MockAlert:
        def __init__(self, class_name, zone, distance, danger, priority, voice_profile):
            self.class_name = class_name
            self.zone = zone
            self.distance_m = distance
            self.danger_level = danger
            self.priority = priority
            self.voice_profile = voice_profile
        
        def get_announcement(self, verbosity="standard"):
            return f"{self.class_name} {self.distance_m} meters on your {self.zone}"
    
    with AudioFeedback(config_path="config/settings.yaml") as audio:
        print("\nTesting individual announcements...")
        
        # Test different voice profiles
        audio.speak("Testing calm voice", priority=1, voice_profile="calm")
        time.sleep(2)
        
        audio.speak("Testing alert voice", priority=5, voice_profile="alert")
        time.sleep(2)
        
        audio.speak("Testing urgent voice", priority=10, voice_profile="urgent")
        time.sleep(2)
        
        print("\nTesting alert announcements...")
        
        # Create test alerts
        test_alerts = [
            MockAlert("person", "center", 0.8, "critical", 40, "urgent"),
            MockAlert("car", "left", 2.5, "warning", 25, "alert"),
            MockAlert("chair", "right", 3.2, "info", 10, "calm"),
            MockAlert("bicycle", "center", 4.0, "info", 5, "calm"),
        ]
        
        announced = audio.announce_alerts(test_alerts)
        print(f"Announced {announced} alerts")
        
        time.sleep(5)
        
        print("\nTesting critical interrupt...")
        audio.speak("This is a long message that should be interrupted by a critical alert", 
                   priority=1, voice_profile="calm")
        time.sleep(0.5)
        audio.announce_critical("Person directly ahead!")
        
        time.sleep(3)
        
        # Print final stats
        stats = audio.get_stats()
        print(f"\n[AudioFeedback] Final Statistics:")
        print(f"  Total announcements: {stats['total_announcements']}")
        print(f"  Interrupted: {stats['interrupted_count']}")
        print(f"  Queue drops: {stats['queue_drops']}")
    
    print("\nTest complete!")
