import cv2
import time
from config import MOTION_THRESHOLD, MOTION_POST_DETECTION_TIME # Import necessary constants

class MotionDetector:
    def __init__(self, threshold=MOTION_THRESHOLD):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=25, detectShadows=False
        )
        self.threshold = threshold
        self.last_motion_time = 0

    def detect(self, frame):
        """Detects motion in the frame and updates the last motion time."""
        fgmask = self.fgbg.apply(frame)
        motion_pixels = cv2.countNonZero(fgmask)
        has_motion = motion_pixels > (frame.shape[0] * frame.shape[1] * self.threshold)
        if has_motion:
            self.last_motion_time = time.time()
        return has_motion or (time.time() - self.last_motion_time < MOTION_POST_DETECTION_TIME)
