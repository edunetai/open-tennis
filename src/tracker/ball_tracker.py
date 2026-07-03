import numpy as np
from scipy.interpolate import CubicSpline

from src.tracker.motion_ball import FrameDifferenceDetector


class BallTracker:
    def __init__(self, geometry_solver, max_velocity_fps=161.3, fps=30):
        self.geometry_solver = geometry_solver
        self.max_velocity_fps = max_velocity_fps
        self.fps = fps
        self.max_step_distance = max_velocity_fps * (1.0 / fps)
        self.trajectory = []
        self.motion_detector = FrameDifferenceDetector()
        self.consecutive_misses = 0

    def process_frame(self, frame_id, frame_image, yolo_ball_centroid):
        """
        Validates and links ball detections across frames.
        yolo_ball_centroid: (x, y) pixel coordinate or None
        Returns physical coordinate (x', y') or None
        """
        physical_candidate = None

        if yolo_ball_centroid is not None:
            physical_candidate = self.geometry_solver.pixel_to_court(yolo_ball_centroid)

        if physical_candidate is not None and len(self.trajectory) > 0:
            last_pos = np.array(self.trajectory[-1][:2], dtype=np.float32)
            dist_moved = np.linalg.norm(physical_candidate - last_pos)

            if dist_moved > self.max_step_distance:
                physical_candidate = None
            else:
                self.consecutive_misses = 0

        if physical_candidate is None and frame_image is not None and len(self.trajectory) > 0:
            last_pos = np.array(self.trajectory[-1][:2], dtype=np.float32)
            search_radius_px = int(self.max_step_distance * 2)
            search_x1 = max(0, int(last_pos[0]) - search_radius_px)
            search_y1 = max(0, int(last_pos[1]) - search_radius_px)
            search_x2 = min(frame_image.shape[1], int(last_pos[0]) + search_radius_px)
            search_y2 = min(frame_image.shape[0], int(last_pos[1]) + search_radius_px)
            search_roi = (search_x1, search_y1, search_x2, search_y2)

            fallback_pixel = self.motion_detector.detect_motion(frame_image, search_roi)
            if fallback_pixel is not None:
                fallback_physical = self.geometry_solver.pixel_to_court(fallback_pixel)
                dist_moved = np.linalg.norm(fallback_physical - last_pos)
                if dist_moved <= self.max_step_distance:
                    physical_candidate = fallback_physical
                    self.consecutive_misses = 0

        if physical_candidate is None and len(self.trajectory) > 0:
            self.consecutive_misses += 1

        if physical_candidate is not None:
            self.trajectory.append((physical_candidate[0], physical_candidate[1], frame_id))
            return physical_candidate

        return None

    def interpolate_gaps(self, max_gap_frames=5):
        """
        Interpolates short tracking gaps using cubic spline interpolation.
        """
        if len(self.trajectory) < 3:
            return self.trajectory

        traj = np.array(self.trajectory)
        frames = traj[:, 2].astype(int)
        xs = traj[:, 0]
        ys = traj[:, 1]

        full_frames = np.arange(frames[0], frames[-1] + 1)
        cs_x = CubicSpline(frames, xs)
        cs_y = CubicSpline(frames, ys)

        interp_x = cs_x(full_frames)
        interp_y = cs_y(full_frames)

        self.trajectory = [(float(x), float(y), int(f)) for x, y, f in zip(interp_x, interp_y, full_frames)]
        return self.trajectory

    def get_trajectory(self):
        return self.trajectory
