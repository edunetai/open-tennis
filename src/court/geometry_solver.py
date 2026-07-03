import cv2
import numpy as np

from src.utils.math_utils import MathUtils


class GeometrySolver(object):
    CANONICAL_KEYPOINTS = np.array(
        [
            [0.0, 44.0],
            [10.0, 44.0],
            [20.0, 44.0],
            [0.0, 29.0],
            [10.0, 29.0],
            [20.0, 29.0],
            [0.0, 15.0],
            [10.0, 15.0],
            [20.0, 15.0],
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, canonical_points=None):
        if canonical_points is not None:
            self.canonical_points = np.array(canonical_points, dtype=np.float32)
        else:
            self.canonical_points = self.CANONICAL_KEYPOINTS.copy()
        self.H = None
        self.H_inv = None

    def fit_homography(self, detected_keypoints):
        detected_keypoints = np.array(detected_keypoints, dtype=np.float32)
        if detected_keypoints.shape[0] != 12 or detected_keypoints.shape[1] != 2:
            raise ValueError(
                "detected_keypoints must have shape (12, 2), got %s"
                % (detected_keypoints.shape,)
            )

        H, mask = cv2.findHomography(
            self.canonical_points, detected_keypoints, method=cv2.RANSAC
        )
        if H is None:
            raise ValueError("Failed to fit homography using cv2.findHomography")

        self.H = H
        self.H_inv = np.linalg.inv(H)

    def pixel_to_court(self, pixel_coord):
        pixel_coord = np.array(pixel_coord, dtype=np.float32)
        court_coord = MathUtils.pixel_to_homogeneous(pixel_coord, self.H_inv)
        return court_coord

    def court_to_pixel(self, court_coord):
        court_coord = np.array(court_coord, dtype=np.float32)
        pixel_coord = MathUtils.pixel_to_homogeneous(court_coord, self.H)
        return pixel_coord

    @staticmethod
    def calculate_velocity_and_speed(coord_t1, coord_t2, fps):
        coord_t1 = np.array(coord_t1, dtype=np.float32)
        coord_t2 = np.array(coord_t2, dtype=np.float32)
        dist_ft = np.linalg.norm(coord_t2 - coord_t1)
        time_sec = 1.0 / fps
        vel_fps = dist_ft / time_sec
        speed_mph = vel_fps * 0.681818
        return vel_fps, speed_mph

    @staticmethod
    def is_within_court_polygon(physical_coord, padding=3.0):
        x, y = float(physical_coord[0]), float(physical_coord[1])
        in_x = -padding <= x <= 20.0 + padding
        in_y = -padding <= y <= 44.0 + padding
        return in_x and in_y

    def iterative_refine(
        self, frame_image, detected_keypoints, max_iterations=15, convergence_threshold=0.2
    ):
        pass
