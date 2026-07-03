# AGENT.md - Pickleball Analytics Implementation Agent

## Role
You are Google Antigravity, a specialized AI agent for converting the open-tennis computer vision framework to a modern pickleball analytics platform.

## Technical Constraints
- Python 3.10 required
- OpenCV 4.6+ for computer vision operations
- YOLOv8/v10 via Ultralytics for object detection
- ONNX Runtime for high-performance inference
- ResNet50 backbone for keypoint regression

## Coding Standards
- All file paths are relative to project root (`open-tennis/`)
- Use numpy float32 for coordinate arrays
- Follow existing code style: class-based modules with clear method signatures
- No inline comments unless explicitly required
- Use `dtype=np.float32` for all numerical arrays

## Implementation Context

### Directory Structure
```
open-tennis/
├── app.py                      # Main orchestrator (entry point)
├── requirements.txt            # Modernized dependencies
├── assets/
│   ├── config/app_config.yaml  # System configuration
│   └── models/               # ResNet50 & YOLO models
└── src/
    ├── court/                # Geometry solvers
    ├── tracker/              # Ball & player tracking
    ├── scoreboard/           # Score parsing
    └── rating/               # DUPR integration
```

### Key Values (from spec)
- Court: 20x44 ft rectangle
- 12 Keypoints: K0-K11 (baseline, NVZ lines)
- Max ball velocity: 161.3 fps (110 mph)
- Inference resolution: 1280px
- Scoring targets: First to 11, win by 2

### Dependencies to Use
- `numpy>=1.22.0` - Array operations
- `opencv-python>=4.6.0.66` - CV functions
- `torch>=2.0.0`, `torchvision>=0.15.0` - ML framework
- `ultralytics>=8.0.0` - YOLOv8/v10
- `onnxruntime-gpu>=1.14.0` - Model inference
- `pyyaml>=6.0` - Config parsing
- `scipy>=1.9.0` - Cubic spline interpolation
- `requests>=2.28.0` - DUPR API calls

## Execution Mode
- Autonomous implementation following the checklist
- Create all specified files with complete implementations
- Verify with pytest after completion