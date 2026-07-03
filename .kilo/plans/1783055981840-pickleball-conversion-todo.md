# Pickleball Analytics Platform - Task Backlog

## Phase 1: Foundation Setup
1. ~~Create project directory structure~~
2. ~~Update requirements.txt with modern dependencies~~
3. ~~Create app_config.yaml with all configuration parameters~~

## Phase 2: Geometry Core
4. ~~Implement `src/court/geometry_solver.py`~~
   - ~~12 canonical keypoint array for 20x44 ft court~~
   - ~~`fit_homography()` for matrix computation~~
   - ~~`pixel_to_court()` and `court_to_pixel()` projections~~
   - ~~`calculate_velocity_and_speed()` conversion (fps to mph)~~
   - ~~`is_within_court_polygon()` boundary validation~~

## Phase 3: Tracking Modules
5. ~~Implement `src/tracker/motion_ball.py`~~
   - ~~Frame differencing with Gaussian blur~~
   - ~~Contour filtering for pickleball-sized objects~~

6. ~~Implement `src/tracker/ball_tracker.py`~~
   - ~~Physical constraint validation (161.3 fps max)~~
   - ~~Motion fallback integration~~
   - ~~Gap interpolation~~

7. ~~Implement `src/tracker/player_tracker.py`~~
   - ~~Court boundary filtering~~
   - ~~Track cleanup for short-lived detections~~

## Phase 4: Scoring & Integration
8. ~~Implement `src/scoreboard/parser.py`~~
   - ~~Three-number score parsing (server-receiver-server_num)~~
   - ~~Server side parity validation~~
   - ~~Traditional side-out state machine~~

9. ~~Implement `src/rating/dupr_client.py`~~
   - ~~JWT authentication~~
   - ~~Match result submission~~

## Phase 5: Orchestration
10. ~~Implement `app.py` main pipeline~~
    - ~~Config loading~~
    - ~~Module initialization~~
    - ~~Frame processing loop~~
    - ~~Trajectory interpolation~~
    - ~~DUPR upload~~

## Phase 6: Model Integration
11. ~~Implement `src/court/keypoint_regressor.py` (ResNet50 keypoint regression)~~
12. ~~Implement `src/controllers/detector/yolo_detector.py` (YOLOv8/v10 ONNX inference)~~

## Phase 7: Validation
13. ~~Create `test_physics_engine.py`~~
    - ~~Homography precision test~~
    - ~~Speed limit filter test~~
    - ~~Server side parity test~~
    - ~~Canonical keypoints test~~
    - ~~Scoring state machine test~~
14. ~~Run test suite and verify all tests pass~~
