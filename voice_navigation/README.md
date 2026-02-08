# Voice Navigation System

Real-time navigation assistance for visually impaired users using computer vision and voice interaction.

## Features

- **Real-time Object Detection**: YOLOv8 detects people, obstacles, and hazards
- **Distance Estimation**: Calculates distance to detected objects
- **Zone-based Alerts**: Audio alerts based on object position (left/center/right)
- **Voice Commands**: Natural language queries about surroundings
- **AI Assistance**: LLM-powered responses via Ollama

## Quick Start

### 1. Install Dependencies

```bash
cd blind_assistant/voice_navigation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download YOLO Model

Download YOLOv8n from: https://github.com/ultralytics/assets/releases

Place in:
```
voice_navigation/models/yolo/yolov8n.pt
```

Or auto-download on first run:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Run System Check

```bash
python tools/system_check.py
```

### 4. Start Navigation

```bash
python src/main.py
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5 / M1 | Intel i7 / M2 |
| RAM | 4 GB | 8 GB |
| Camera | 720p webcam | 1080p webcam |
| Microphone | Built-in | External |
| Audio | Earbuds | Bone conduction |

## Usage

### Voice Commands

| Command | Action |
|---------|--------|
| "clear" | Check if path is clear |
| "ahead" | What's directly ahead |
| "status" | Everything around you |
| "help" | Get assistance |

### Keyboard Controls

| Key | Action |
|-----|--------|
| SPACE | Speak a command |
| Q | Quit |

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
camera:
  source: 0           # Camera index or video file
  fps: 30

yolo:
  confidence: 0.45    # Detection confidence

safety:
  min_distance: 1.0   # Critical distance (meters)
```

## Documentation

- [User Guide](docs/user_guide.md) - For blind users
- [Calibration Guide](docs/calibration_guide.md) - Distance calibration
- [Privacy Policy](docs/privacy_policy.md) - Data handling

## AI Assistant (Optional)

For enhanced voice responses, install Ollama:

```bash
# macOS
brew install ollama

# Start and pull model
ollama run llama3.2:3b
```

## Testing

```bash
# Run benchmark
python tests/benchmark.py --duration 30

# Generate HTML report
python tests/benchmark.py --report html
```

## Project Structure

```
voice_navigation/
├── src/
│   ├── main.py              # Main orchestrator
│   ├── camera_capture.py    # Video input
│   ├── object_detector.py   # YOLO detection
│   ├── safety_manager.py    # Distance/zone analysis
│   ├── scene_analyzer.py    # Object tracking
│   ├── audio_feedback.py    # Text-to-speech
│   ├── ai_assistant.py      # LLM integration
│   ├── conversation_handler.py  # Voice input
│   ├── telemetry.py         # Logging/metrics
│   └── calibration_tool.py  # Distance calibration
├── config/
│   └── settings.yaml        # Configuration
├── tests/
│   ├── benchmark.py         # Performance tests
│   └── user_testing_protocol.md
└── docs/
    ├── user_guide.md
    └── calibration_guide.md
```

## Troubleshooting

### Camera not working
```bash
python src/camera_capture.py  # Test camera directly
```

### Audio not playing
```bash
python src/audio_feedback.py  # Test audio
```

### Distance estimates wrong
```bash
python src/calibration_tool.py  # Run calibration
```

## License

MIT License - See LICENSE file

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Ollama](https://ollama.ai)
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3)
