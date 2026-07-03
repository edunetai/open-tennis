# Pickleball Video Analytics Platform - Strategic Overview

## Project Goals
Transform the legacy open-tennis computer vision framework into a modern pickleball analytics platform by:
- Replacing classical Hough line detection with ResNet50 keypoint regression for court boundary mapping
- Upgrading YOLOv5 to YOLOv8/v10 for high-resolution ball detection (1280px inference)
- Implementing physics-constrained ball tracking with motion-differencing fallback
- Adapting scoring logic from tennis (game-set-match) to pickleball (side-out system)
- Integrating with DUPR API for match result synchronization

## Modernized Architecture

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

## Core Logic

### 1. Geometry & Homography
- Map 12 keypoint coordinates (K0-K11) to canonical court positions
- Project pixels to physical coordinates using homography matrix H
- Iteratively refine using white-line pixel extraction and residual minimization

### 2. Ball Tracking
- Constraint: ||b_t - b_{t-1}||_2 <= 161.3 * dt (110 mph max speed)
- Motion fallback triggers after 5 consecutive missed detections
- Cubic spline interpolation fills tracking gaps

### 3. Player Tracking
- Filter spectators using court boundary validation (padding=5.0 ft)
- Minimum track length: 15 frames
- Proximity threshold: 3.5 ft for hit detection

### 4. Scoring State Machine
- Format: "server_score-receiver_score-active_server"
- Initial state: "0-0-2"
- Server side parity rules:
  - Even score: server on right (x >= 10.0 ft)
  - Odd score: server on left (x < 10.0 ft)

### 5. DUPR Integration
- Authenticate via POST /auth/v1.0/token
- Submit matches via POST /v1.0/match with JWT header