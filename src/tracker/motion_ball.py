import cv2
import numpy as np


class FrameDifferenceDetector:
    def __init__(self, threshold_value=25):
        self.prev_frame = None
        self.threshold_value = threshold_value

    def detect_motion(self, frame_image, search_roi=None):
        """
        Performs frame subtraction to isolate moving objects when YOLO fails.
        Optionally restricts search to a region of interest (x1, y1, x2, y2).
        Returns (x, y) center of detected motion blob or None.
        """
        gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if search_roi is not None:
            x1, y1, x2, y2 = search_roi
            gray = gray[y1:y2, x1:x2]

        if self.prev_frame is None:
            self.prev_frame = gray
            return None

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold_value, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.prev_frame = gray

        best_contour = None
        best_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 <= area <= 500:
                if area > best_area:
                    best_area = area
                    best_contour = contour

        if best_contour is not None:
            (x, y, w, h) = cv2.boundingRect(best_contour)
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            if search_roi is not None:
                center_x += x1
                center_y += y1
            return (center_x, center_y)

        return None

    def reset(self):
        self.prev_frame = None
