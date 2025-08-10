import time
import numpy as np
from collections import deque
from config import TRACKER_ALPHA, TRACKER_HISTORY_SIZE, ANALYSIS_TIMEOUT, TRACKER_RETRY_ATTEMPTS, TRACKER_RETRY_COOLDOWN

class DetectionTracker:
    """
    Tracks face detection results over time to provide smoothed and stable outputs.
    Improved for responsiveness to fast movements.
    """
    def __init__(self, alpha=TRACKER_ALPHA, history_size=TRACKER_HISTORY_SIZE):
        self.position_history = deque(maxlen=history_size)
        self.age_history = deque(maxlen=history_size)
        self.gender_history = deque(maxlen=history_size)
        self.last_update_timestamp = 0
        self.alpha = alpha
        self.retry_count = 0
        self.last_retry_attempt_time = 0
        self._has_ever_detected = False

    def update_from_result(self, result_data):
        """
        Update tracker state based on data received from the analysis thread.
        """
        timestamp = result_data.get('timestamp', time.time())
        self.last_update_timestamp = timestamp
        
        face_data = result_data.get('face', {})
        
        if face_data:
            self._has_ever_detected = True
            self.retry_count = 0

            region = face_data.get('region', {})
            if all(k in region for k in ['x', 'y', 'w', 'h']):
                self.position_history.append((
                    region['x'], region['y'],
                    region['w'], region['h']
                ))

            if 'age' in face_data:
                self.age_history.append(face_data['age'])

            if 'gender' in face_data:
                gender_data = face_data['gender']
                if isinstance(gender_data, dict) and gender_data:
                    dominant_gender = max(gender_data, key=gender_data.get)
                    confidence = gender_data[dominant_gender]
                    self.gender_history.append((dominant_gender, confidence))
        else:
            if self._has_ever_detected and self.is_valid(timeout=ANALYSIS_TIMEOUT * 0.5):
                 self.retry_count = min(self.retry_count + 1, TRACKER_RETRY_ATTEMPTS)
                 self.last_retry_attempt_time = timestamp

    def get_smoothed_age(self):
        """Calculates a weighted average age from recent history."""
        if not self.age_history:
            return None

        ages = list(self.age_history)
        if len(ages) == 1:
            return int(round(ages[0]))


        weights = np.exp(np.linspace(-2, 0, len(ages))) 
        weights /= weights.sum()
        smoothed_age = np.average(ages, weights=weights)
        return int(round(smoothed_age))

    def get_smoothed_gender(self):
        """
        Determines the most likely gender and its smoothed confidence.
        """
        if not self.gender_history:
            return None, None

        genders, confidences = zip(*self.gender_history)
        if len(confidences) == 1:
            gender, conf = genders[0], confidences[0]
            return (gender, conf) if conf >= 50.0 else (None, None)

        weights = np.exp(np.linspace(-2, 0, len(confidences)))
        weights /= weights.sum()
        avg_confidence = np.average(confidences, weights=weights)

        unique_genders, counts = np.unique(genders, return_counts=True)
        dominant_gender = unique_genders[np.argmax(counts)]

        return (dominant_gender, avg_confidence) if avg_confidence >= 50.0 else (None, None)

    def get_smoothed_position(self):
        """
        Calculates a weighted average bounding box with higher weight on recent positions.
        """
        if not self.position_history:
            return None, None, None, None

        positions = list(self.position_history)
        if len(positions) == 1:
            return positions[0]

        weights = np.exp(np.linspace(-2, 0, len(positions))) 
        weights /= weights.sum()
        x_vals, y_vals, w_vals, h_vals = zip(*positions)

        x_smooth = int(round(np.average(x_vals, weights=weights)))
        y_smooth = int(round(np.average(y_vals, weights=weights)))
        w_smooth = int(round(np.average(w_vals, weights=weights)))
        h_smooth = int(round(np.average(h_vals, weights=weights)))

        return x_smooth, y_smooth, w_smooth, h_smooth

    def is_valid(self, timeout=ANALYSIS_TIMEOUT):
        """
        Checks if the tracker data is still considered recent and valid.
        """
        return (time.time() - self.last_update_timestamp) < timeout

    def should_retry_detection(self, current_time):
        """
        Determines if a retry should be attempted.
        """
        return (
            self._has_ever_detected and
            self.is_valid() and
            self.retry_count > 0 and
            (current_time - self.last_retry_attempt_time) > TRACKER_RETRY_COOLDOWN
        )

    def get_detections_for_drawing(self):
        """
        Provides the current best estimate of face location for drawing.
        """
        if self.is_valid() and self.position_history:
            x, y, w, h = self.get_smoothed_position()
            if x is not None:
                return [{'region': {'x': x, 'y': y, 'w': w, 'h': h}}]
        return []

    def consume_retry(self):
        """Decrements the retry count."""
        if self.retry_count > 0:
            self.retry_count -= 1
            self.last_retry_attempt_time = time.time()
