import numpy as np


class PlayerTracker:
    def __init__(self, geometry_solver, min_track_length=15):
        self.geometry_solver = geometry_solver
        self.min_track_length = min_track_length
        self.active_tracks = {}

    def update_player_tracks(self, frame_id, detected_persons):
        """
        Filters out spectator noise and tracks active players on the court.
        detected_persons: list of bounding boxes [x1, y1, x2, y2, track_id]
        """
        updated_tracks = {}
        for box in detected_persons:
            x1, y1, x2, y2, track_id = box
            center_x = (x1 + x2) / 2.0
            center_y = y2

            physical_pos = self.geometry_solver.pixel_to_court((center_x, center_y))

            if not self.geometry_solver.is_within_court_polygon(physical_pos, padding=5.0):
                continue

            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = {
                    "history": [],
                    "physical_history": [],
                    "frames": []
                }

            self.active_tracks[track_id]["history"].append([x1, y1, x2, y2])
            self.active_tracks[track_id]["physical_history"].append(physical_pos)
            self.active_tracks[track_id]["frames"].append(frame_id)
            updated_tracks[track_id] = self.active_tracks[track_id]

        cleaned_tracks = {}
        for track_id, data in updated_tracks.items():
            if len(data["frames"]) >= self.min_track_length:
                cleaned_tracks[track_id] = data

        return cleaned_tracks
