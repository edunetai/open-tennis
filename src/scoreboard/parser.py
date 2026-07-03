import logging
import numpy as np


class ScoreboardParser:
    def __init__(self, target_score=11):
        self.target_score = target_score
        self.score_history = []
        self.current_state = {"server_score": 0, "receiver_score": 0, "server_num": 2}
        self.logger = logging.getLogger("ScoreboardParser")

    def parse_score(self, scoreboard_crop_img):
        """
        Parses score details from the scoreboard image region.
        In production, replace this stub with OCR model inference (e.g., CRNN).
        Returns current state when OCR is unavailable.
        """
        return self.current_state

    def validate_position_parity(self, server_score, active_server, physical_server_pos):
        """
        Validates parsed scores against physical player positions.
        - Even server scores = server must be on the right side (x >= 10.0 ft).
        - Odd server scores = server must be on the left side (x < 10.0 ft).
        """
        is_even = (server_score % 2 == 0)
        server_x_coord = physical_server_pos[0] if isinstance(physical_server_pos, (list, tuple, np.ndarray)) else physical_server_pos

        if is_even:
            return server_x_coord >= 10.0
        else:
            return server_x_coord < 10.0

    def register_point(self, winning_team):
        """
        Updates the scoring state machine based on rally winner.
        winning_team: "SERVER" or "RECEIVER"
        """
        if winning_team == "SERVER":
            self.current_state["server_score"] += 1
        else:
            if self.current_state["server_num"] == 1:
                self.current_state["server_num"] = 2
            else:
                self.current_state["server_num"] = 1
                temp = self.current_state["server_score"]
                self.current_state["server_score"] = self.current_state["receiver_score"]
                self.current_state["receiver_score"] = temp

        self.score_history.append(dict(self.current_state))
        return self.current_state

    def get_state_string(self):
        return f"{self.current_state['server_score']}-{self.current_state['receiver_score']}-{self.current_state['server_num']}"

    def reset(self):
        self.current_state = {"server_score": 0, "receiver_score": 0, "server_num": 2}
        self.score_history = []
