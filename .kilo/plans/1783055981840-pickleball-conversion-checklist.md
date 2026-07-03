# Pickleball Analytics Platform - Implementation Checklist

## Environment & Dependencies
- [x] Update requirements.txt with modern packages (YOLOv8, onnxruntime-gpu, scipy)
- [x] Create conda environment `pickleball_watch` with Python 3.10
- [x] Install all dependencies via `pip install -r requirements.txt`

## Configuration
- [x] Create `assets/config/app_config.yaml` with all required parameters:
  - [x] paths: video_path, output_video_path, logs_path, keypoints_cache_path
  - [x] detector: yolo_engine_path, execution_provider, confidence_threshold, iou_threshold, inference_image_size=1280
  - [x] court_regressor: model_path, num_keypoints=12, iterative_refinement, max_iterations=15, convergence_threshold=0.2
  - [x] ball_tracker: max_velocity_fps=161.3, max_gap_interpolation_frames=5, motion_fallback_enabled, min_rally_duration_seconds=1.5
  - [x] player_tracker: court_filtering=true, min_track_length_frames=15, proximity_threshold_feet=3.5
  - [x] scoring: mode="traditional", target_score=11, win_by_two=true, starting_state="0-0-2"
  - [x] dupr_api: enabled, api_url, client_key, client_secret, sandbox_mode

## Core Modules

### Geometry Solver (`src/court/geometry_solver.py`)
- [x] Implement `__init__` with 12 canonical points (20x44 ft court)
- [x] Implement `fit_homography(detected_keypoints)` using cv2.findHomography
- [x] Implement `pixel_to_court(pixel_coord)` with homography projection
- [x] Implement `court_to_pixel(court_coord)` inverse projection
- [x] Implement `calculate_velocity_and_speed(coord_t1, coord_t2, fps)`
- [x] Implement `is_within_court_polygon(physical_coord, padding)`

### Player Tracker (`src/tracker/player_tracker.py`)
- [x] Implement `update_player_tracks(frame_id, detected_persons)`
- [x] Map bounding box center-bottom to physical coordinates
- [x] Filter tracks outside court boundaries (padding=5.0)
- [x] Clean up tracks shorter than min_track_length (15 frames)

### Motion Ball Detector (`src/tracker/motion_ball.py`)
- [x] Implement `detect_motion(frame_image)`
- [x] Apply GaussianBlur and frame differencing
- [x] Use threshold (25) and dilation for motion isolation
- [x] Filter contours by area (10-500 pixels)
- [x] Return center point of detected motion

### Ball Tracker (`src/tracker/ball_tracker.py`)
- [x] Implement `process_frame(frame_id, frame_image, yolo_ball_centroid)`
- [x] Map YOLO detections to physical coordinates
- [x] Apply velocity constraint filter (max_step_distance)
- [x] Trigger motion fallback on consecutive misses
- [x] Implement `interpolate_gaps(max_gap_frames)` with cubic spline

### Scoreboard Parser (`src/scoreboard/parser.py`)
- [x] Implement `parse_score(scoreboard_crop_img)` stub
- [x] Implement `validate_position_parity(server_score, active_server, physical_server_pos)`
- [x] Implement `register_point(winning_team)` state machine logic
- [x] Validate server side parity rules

### DUPR Client (`src/rating/dupr_client.py`)
- [x] Implement `authenticate()` for JWT token retrieval
- [x] Implement `upload_match_result(payload)` with bearer token

## Main Pipeline (`app.py`)
- [x] Implement `load_system_config(config_path)`
- [x] Initialize GeometrySolver with reference keypoints
- [x] Initialize PlayerTracker, BallTracker, ScoreboardParser
- [x] Implement frame processing loop with mock detection values
- [x] Call `interpolate_gaps()` post-processing
- [x] Upload results to DUPR if enabled

## Model Integration
- [x] Create `src/court/keypoint_regressor.py` (ResNet50 keypoint regression)
- [x] Create `src/controllers/detector/yolo_detector.py` (YOLOv8/v10 ONNX inference)

## Testing
- [x] Create `test_physics_engine.py`
- [x] Test homography precision (net line at y=22 ft)
- [x] Test speed limit filter (anomalous jumps rejected)
- [x] Test server side parity validation
- [x] Test canonical keypoints
- [x] Test scoring state machine
- [x] Run `pytest test_physics_engine.py -v`

## File Structure Verification
- [x] `open-tennis/app.py`
- [x] `open-tennis/requirements.txt`
- [x] `open-tennis/assets/config/app_config.yaml`
- [x] `open-tennis/src/court/geometry_solver.py`
- [x] `open-tennis/src/court/keypoint_regressor.py`
- [x] `open-tennis/src/controllers/detector/yolo_detector.py`
- [x] `open-tennis/src/tracker/player_tracker.py`
- [x] `open-tennis/src/tracker/motion_ball.py`
- [x] `open-tennis/src/tracker/ball_tracker.py`
- [x] `open-tennis/src/scoreboard/parser.py`
- [x] `open-tennis/src/rating/dupr_client.py`
