<div align="center">

# Open Tennis - Pickleball Video Analytics Platform

A modernized computer vision framework for analyzing pickleball matches using deep keypoint regression, physics-constrained ball tracking, and DUPR API integration.

[Architecture](#architecture) • [Features](#features) • [Installation](#installation) • [Configuration](#configuration) • [Usage](#usage) • [Testing](#testing) • [Roadmap](#roadmap)

</div>

## Architecture

```
[Frame (imgsz=1280)]
        │
        ├─► Ball & Player Ground Coordinates
        │
        └─► 12 Canonical Court Keypoints
                 │
                 ▼
        [Homography Solver] ──► Physical Coordinate Mapping (0-20ft x 0-44ft)
                 │
                 ▼
        [Physics-Constrained Linker] ──► Trajectory & Kinematics
```

The platform replaces legacy Hough line detection with a ResNet50 keypoint regressor and upgrades YOLOv5 to YOLOv8/v10 for high-resolution ball detection. A physics-constrained tracking engine filters false positives and corrects occlusions via motion-differencing fallback.

## Features

- [x] Court boundary mapping via 12 canonical keypoints (ResNet50)
- [x] Homography-based pixel-to-physical coordinate projection
- [x] High-resolution ball detection (1280px inference)
- [x] Physics-constrained ball tracking with velocity limit filtering (161.3 fps / 110 mph max)
- [x] Motion-differencing fallback for missed detections
- [x] Cubic spline interpolation for tracking gap filling
- [x] Player tracking with court boundary validation (spectator filtering)
- [x] Pickleball side-out scoring state machine (0-0-2 starting state)
- [x] Server position parity validation against physical court coordinates
- [x] DUPR API integration for match result synchronization
- [x] Configurable evaluation and video export

## Installation

### Requirements

- Linux / macOS
- Python >= 3.10
- CUDA >= 11.0 (recommended)

### Steps

1. Clone the repository

```bash
git clone https://github.com/your-org/open-tennis.git
cd open-tennis
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Download model assets

Place the following models in `assets/models/`:
- `yolov8x_pickleball.onnx` — YOLOv8/v10 ball and player detector
- `resnet50_keypoints.pth` — Court keypoint regressor

## Configuration

Edit `assets/configs/app_config.yaml` to set paths and parameters:

| Section | Key | Description |
|---------|-----|-------------|
| `paths` | `video_path` | Input video for inference |
| | `output_video_path` | Path to save annotated video |
| `detector` | `inference_image_size` | YOLO inference resolution (default 1280) |
| | `confidence_threshold` | Detection confidence threshold |
| | `iou_threshold` | NMS IoU threshold |
| `court_regressor` | `model_path` | ResNet50 keypoint regressor weights |
| | `num_keypoints` | Number of court keypoints (12) |
| | `iterative_refinement` | Enable white-line residual minimization |
| `ball_tracker` | `max_velocity_fps` | Physical speed limit (161.3) |
| | `max_gap_interpolation_frames` | Max frames to interpolate (5) |
| | `motion_fallback_enabled` | Enable frame-differencing fallback |
| `player_tracker` | `min_track_length_frames` | Minimum frames to keep a track (15) |
| | `proximity_threshold_feet` | Hit detection radius (3.5 ft) |
| `scoring` | `mode` | `traditional` (side-out) or `rally` |
| | `target_score` | Points to win (default 11) |
| | `win_by_two` | Require 2-point lead |
| | `starting_state` | Initial score state (0-0-2) |
| `dupr_api` | `enabled` | Enable DUPR result sync |
| | `api_url` | DUPR API endpoint |
| | `client_key` / `client_secret` | Partner credentials (use env vars) |

## Usage

Run inference on a video:

```bash
python app.py
```

Run the validation test suite:

```bash
pytest test_physics_engine.py -v
```

## Testing

The project includes physics-based validation tests:

- **Homography precision** — Verifies pixel-to-court projection accuracy (net line at y=22 ft)
- **Speed limit filter** — Ensures the tracker discards improbable position jumps
- **Server side parity** — Validates scoring state against physical player positions
- **Canonical keypoints** — Confirms court geometry mapping
- **Scoring state machine** — Verifies side-out state transitions

## Roadmap

- [ ] Train ResNet50 keypoint regressor on pickleball court datasets
- [ ] Integrate YOLOv8/v10 ONNX runtime for production inference
- [ ] Implement velocity-reversal hit detection heuristic
- [ ] Add rally segmentation and shot classification
- [ ] Player activity analysis and heatmaps
- [ ] Multi-camera calibration and synchronization

## Project Structure

```
open-tennis/
├── app.py                        # Main orchestrator
├── requirements.txt              # Modern Python dependencies
├── test_physics_engine.py        # Validation tests
├── assets/
│   ├── configs/
│   │   ├── app_config.yaml       # Main application config
│   │   └── detector_config.yaml  # YOLO detector parameters
│   └── models/                   # Model weights (.onnx, .pth)
├── src/
│   ├── court/
│   │   └── geometry_solver.py    # Homography and coordinate mapping
│   ├── tracker/
│   │   ├── ball_tracker.py       # Physics-constrained ball tracking
│   │   ├── player_tracker.py     # Player track filtering
│   │   └── motion_ball.py        # Frame-differencing fallback
│   ├── scoreboard/
│   │   └── parser.py             # Pickleball side-out scoring engine
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
```

## License

MIT
