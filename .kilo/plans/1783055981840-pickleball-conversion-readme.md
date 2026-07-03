# Pickleball Video Analytics Platform

## Overview
A computer vision platform for analyzing pickleball matches using deep keypoint regression, physics-constrained ball tracking, and DUPR integration.

## Project Structure
```
open-tennis/
├── app.py                      # Main orchestrator
├── requirements.txt            # Dependencies
├── assets/
│   ├── config/app_config.yaml  # Configuration
│   ├── models/               # ML models
│   ├── input_videos/         # Input match videos
│   └── output_videos/        # Annotated output
└── src/
    ├── court/geometry_solver.py    # Homography & coordinate mapping
    ├── tracker/player_tracker.py   # Player filtering & tracking
    ├── tracker/motion_ball.py      # Frame differencing fallback
    ├── tracker/ball_tracker.py     # Physics-constrained ball tracking
    ├── scoreboard/parser.py        # Score parsing & state machine
    └── rating/dupr_client.py      # DUPR API integration
```

## Setup

```bash
# Create conda environment
conda create -n pickleball_watch python=3.10 -y
conda activate pickleball_watch

# Install dependencies
pip install -r requirements.txt
```

## Configuration
Edit `assets/config/app_config.yaml` to set:
- Video input/output paths
- Model paths (YOLOv8 ONNX, ResNet50)
- Tracking thresholds
- DUPR API credentials

## Usage

```bash
# Run analytics pipeline
python app.py

# Run tests
pytest test_physics_engine.py -v
```

## Dependencies
- numpy>=1.22.0
- opencv-python>=4.6.0.66
- torch>=2.0.0, torchvision>=0.15.0
- ultralytics>=8.0.0
- onnxruntime-gpu>=1.14.0
- pyyaml>=6.0
- scipy>=1.9.0
- requests>=2.28.0
- pytest>=7.0.0

## Court Dimensions
- Total: 20x44 feet
- Non-Volley Zone (Kitchen): 7 feet from baseline
- Net: Center at y=22 ft, extends 1.5 ft beyond sidelines