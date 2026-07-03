# AGENT.md

## Role
You are a specialized AI agent for implementing and maintaining the Open Tennis pickleball video analytics platform. Your work spans computer vision, kinematics, scoring logic, and API integration.

## Technical Stack
- Python 3.10+
- OpenCV 4.6+
- PyTorch 2.0+ / torchvision 0.15+
- Ultralytics YOLOv8/v10
- ONNX Runtime GPU
- scipy, numpy, requests, pyyaml

## Coding Standards
- All file paths are relative to project root
- Use `dtype=np.float32` for coordinate arrays
- Follow existing class-based module structure
- No inline comments unless explicitly required
- Use type hints where practical
- Keep functions focused and under 50 lines where possible

## Project Structure
```
open-tennis/
├── app.py                        # Entry point
├── requirements.txt              # Dependencies
├── test_physics_engine.py        # Validation tests
├── assets/
│   ├── configs/
│   │   ├── app_config.yaml       # System configuration
│   │   └── detector_config.yaml  # YOLO parameters
│   └── models/                   # Model weights
├── src/
│   ├── court/
│   │   └── geometry_solver.py    # Homography and coordinate mapping
│   ├── tracker/
│   │   ├── ball_tracker.py       # Physics-constrained ball tracking
│   │   ├── player_tracker.py     # Player track filtering
│   │   └── motion_ball.py        # Frame-differencing fallback
│   ├── scoreboard/
│   │   └── parser.py             # Side-out scoring state machine
│   ├── rating/
│   │   └── dupr_client.py        # DUPR API integration
│   ├── controllers/
│   │   ├── open_tennis.py        # Legacy orchestrator
│   │   ├── model_manager.py      # TensorRT / ONNX runtime
│   │   └── detector/
│   │       └── score_detector.py # YOLO inference wrapper
│   └── utils/
│       ├── daos.py               # Data transfer objects
│       ├── math_utils.py         # Geometry utilities
│       └── renderer.py           # Visualization
└── .kilo/
    └── plans/                    # Implementation plans
```

## Key Domain Values
- Court dimensions: 20 ft x 44 ft
- 12 canonical keypoints (K0-K11)
- Max ball speed: 161.3 fps (110 mph)
- YOLO inference size: 1280px
- Scoring: first to 11, win by 2
- Starting state: 0-0-2

## Execution Mode
- Implement modules from the conversion plan
- Run `pytest test_physics_engine.py -v` to validate
- Do not commit unless explicitly requested
