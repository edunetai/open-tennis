import cv2
import yaml
import numpy as np
from src.court.geometry_solver import GeometrySolver
from src.tracker.player_tracker import PlayerTracker
from src.tracker.ball_tracker import BallTracker
from src.scoreboard.parser import ScoreboardParser
from src.rating.dupr_client import DUPRClient


def load_system_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_analytics_pipeline(config_path):
    config = load_system_config(config_path)

    solver = GeometrySolver()

    reference_keypoints = np.array([
        [0.0, 44.0], [10.0, 44.0], [20.0, 44.0],
        [0.0, 29.0], [10.0, 29.0], [20.0, 29.0],
        [0.0, 15.0], [10.0, 15.0], [20.0, 15.0],
        [0.0, 0.0], [10.0, 0.0], [20.0, 0.0]
    ], dtype=np.float32)

    solver.fit_homography(reference_keypoints)

    player_tracker = PlayerTracker(solver, min_track_length=config["player_tracker"]["min_track_length_frames"])
    ball_tracker = BallTracker(solver, max_velocity_fps=config["ball_tracker"]["max_velocity_fps"], fps=30)
    score_engine = ScoreboardParser()

    cap = cv2.VideoCapture(config["paths"]["video_path"])
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mock_ball_pixel = (640, 480) if frame_id % 2 == 0 else None
        mock_persons = [[100, 300, 200, 600, 1], [800, 300, 900, 600, 2]]

        active_players = player_tracker.update_player_tracks(frame_id, mock_persons)
        active_ball = ball_tracker.process_frame(frame_id, frame, mock_ball_pixel)

        frame_id += 1

    cap.release()

    ball_tracker.interpolate_gaps(max_gap_frames=config["ball_tracker"]["max_gap_interpolation_frames"])

    if config["dupr_api"]["enabled"]:
        dupr = DUPRClient(
            api_url=config["dupr_api"]["api_url"],
            client_key=config["dupr_api"]["client_key"],
            client_secret=config["dupr_api"]["client_secret"]
        )
        match_payload = {
            "match_source": "COURT_CHECK_CV",
            "match_type": "DOUBLES",
            "scoring_format": "ONE_GAME_TO_11",
            "game_results": [
                {
                    "game_number": 1,
                    "team_1_score": score_engine.current_state["server_score"],
                    "team_2_score": score_engine.current_state["receiver_score"],
                    "winner": "TEAM_1"
                }
            ],
            "team_1": {"player_1_dupr_id": "DPR87261", "player_2_dupr_id": "DPR54329"},
            "team_2": {"player_1_dupr_id": "DPR21098", "player_2_dupr_id": "DPR65431"}
        }
        dupr.upload_match_result(match_payload)


if __name__ == "__main__":
    run_analytics_pipeline("assets/config/app_config.yaml")