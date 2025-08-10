import os
import cv2
import numpy as np
from deepface import DeepFace
import time
import logging
from collections import deque
import threading
import queue

#Configuration 
# Set environment and suppress TensorFlow warnings
os.environ['DEEPFACE_HOME'] = "C:/Users/nemad/deepface_cache"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Configure logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)

#Constants
ROI_MIN_SIZE = 30
ROI_PADDING_RATIO = 0.2
MOTION_THRESHOLD = 0.015
MOTION_POST_DETECTION_TIME = 1.0 # Time to keep detecting after motion stops
ANALYSIS_POST_DETECTION_TIME = 2.0 # Time to keep analyzing after motion stops or tracker becomes invalid
TRACKER_HISTORY_SIZE = 5
TRACKER_ALPHA = 0.4
DETECTION_COOLDOWN = 0.2 # Minimum time between analysis triggers
ANALYSIS_TIMEOUT = 2.0 # Time after which tracker data is considered stale

#DeepFace Model Loading
try:
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    DeepFace.analyze(dummy_img, actions=['age', 'gender'], enforce_detection=False, silent=True, detector_backend='opencv', align=True)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

#Camera Setup 
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

#Performance Tracking and Control Variables
frame_count = 0
start_time = time.time()
fps = 0
last_detection_trigger_time = 0 

#Thread Communication
result_queue = queue.Queue(maxsize=2) 
stop_event = threading.Event()
frame_queue = queue.Queue(maxsize=2)

class DetectionTracker:
    def __init__(self, alpha=TRACKER_ALPHA, history_size=TRACKER_HISTORY_SIZE):
        self.age_history = deque(maxlen=history_size)
        self.gender_history = deque(maxlen=history_size)
        self.position_history = deque(maxlen=history_size)
        self.last_update = 0
        self.alpha = alpha

    def update_from_result(self, result_data):
        """Update tracker based on data received from analysis thread."""
        self.last_update = time.time()
        face = result_data.get('face', {})
        region = face.get('region', {})
        if all(k in region for k in ['x', 'y', 'w', 'h']):
            self.position_history.append((
                region['x'], region['y'],
                region['w'], region['h']
            ))
        if 'age' in face:
            self.age_history.append(face['age'])
        if 'gender' in face:
            gdata = face['gender']
            if isinstance(gdata, dict):
                dominant = max(gdata, key=gdata.get)
                self.gender_history.append((dominant, gdata[dominant]))

    def get_smoothed_age(self):
        if not self.age_history:
            return None
        ages = list(self.age_history)
        if len(ages) == 1:
             return int(ages[0])
        weights = np.exp(np.linspace(-1, 0, len(ages)))
        weights /= weights.sum()
        return int(np.average(ages, weights=weights))

    def get_smoothed_gender(self):
        if not self.gender_history:
            return None, None
        genders = [g for g, _ in self.gender_history]
        confidences = [c for _, c in self.gender_history]
        if len(confidences) == 1:
            return (genders[0], confidences[0]) if confidences[0] >= 50 else (None, None)
        weights = np.exp(np.linspace(-1, 0, len(confidences)))
        weights /= weights.sum()
        avg_conf = np.average(confidences, weights=weights)
        unique_genders, counts = np.unique(genders, return_counts=True)
        dominant = unique_genders[np.argmax(counts)]
        return (dominant, avg_conf) if avg_conf >= 50 else (None, None)

    def get_smoothed_position(self):
        if not self.position_history:
            return None, None, None, None
        positions = list(self.position_history)
        if len(positions) == 1:
            return positions[0]
        weights = np.exp(np.linspace(-1, 0, len(positions)))
        weights /= weights.sum()
        x_vals, y_vals, w_vals, h_vals = zip(*positions)
        x = int(np.average(x_vals, weights=weights))
        y = int(np.average(y_vals, weights=weights))
        w = int(np.average(w_vals, weights=weights))
        h = int(np.average(h_vals, weights=weights))
        return x, y, w, h

    def is_valid(self, timeout=ANALYSIS_TIMEOUT):
        return time.time() - self.last_update < timeout

    def get_detections(self):
        if self.is_valid() and self.position_history:
            pos = self.get_smoothed_position()
            if pos[0] is not None:
                return [{'region': dict(zip(['x', 'y', 'w', 'h'], pos))}]
        return []

class MotionDetector:
    def __init__(self, threshold=MOTION_THRESHOLD):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=25, detectShadows=False
        )
        self.threshold = threshold
        self.last_motion_time = 0

    def detect(self, frame):
        fgmask = self.fgbg.apply(frame)
        motion_pixels = cv2.countNonZero(fgmask)
        has_motion = motion_pixels > (frame.shape[0] * frame.shape[1] * self.threshold)
        if has_motion:
            self.last_motion_time = time.time()
        return has_motion or (time.time() - self.last_motion_time < MOTION_POST_DETECTION_TIME)


def analysis_worker(frame_queue, result_queue, stop_event):
    """Function to run in the separate analysis thread."""
    print("Analysis thread started.")
    last_analysis_time = 0
    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=0.1)
            if item is None: 
                 break

            frame_for_analysis, x_roi_rel, y_roi_rel, w_roi_rel, h_roi_rel, x1_offset_received, y1_offset_received = item
            current_time = time.time()

            try:
                analysis_frame = frame_for_analysis 
                rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)

                result = DeepFace.analyze(
                    rgb,
                    actions=['age', 'gender'],
                    enforce_detection=True, 
                    silent=True,
                    detector_backend='opencv', 
                    align=False 
                )
                faces = result if isinstance(result, list) else [result]

                # Adjust coordinates using offsets received from main thread
                if faces:
                    for face in faces:
                        if 'region' in face:
                            face['region']['x'] += x1_offset_received
                            face['region']['y'] += y1_offset_received

                #Filter valid faces
                valid_faces = []
                for face in faces:
                    region = face.get("region", {})
                    if all(k in region for k in ['x', 'y', 'w', 'h']):
                        if region['w'] > ROI_MIN_SIZE and region['h'] > ROI_MIN_SIZE:
                            valid_faces.append(face)

                result_data = {'face': valid_faces[0] if valid_faces else {}, 'timestamp': current_time}

                try:
                    result_queue.put_nowait(result_data)
                except queue.Full:
                    try:
                        result_queue.get_nowait() #Remove old result
                        result_queue.put_nowait(result_data)
                    except queue.Empty:
                        pass

                last_analysis_time = current_time 

            except Exception as e:
                logging.debug(f"Analysis error (DeepFace): {type(e).__name__}: {e}")
            finally:
                frame_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
             print(f"Unexpected error in analysis thread loop: {type(e).__name__}: {e}")
             logging.error(f"Unexpected error in analysis thread loop: {type(e).__name__}: {e}")
             break

    print("Analysis thread stopped.")


tracker = DetectionTracker(alpha=TRACKER_ALPHA, history_size=TRACKER_HISTORY_SIZE)
motion_detector = MotionDetector(threshold=MOTION_THRESHOLD)
print("Starting camera. Press 'q' to quit.")


for _ in range(5):
    ret, frame = cam.read()
    if ret:
        motion_detector.detect(frame)


analysis_thread = threading.Thread(target=analysis_worker, args=(frame_queue, result_queue, stop_event), daemon=True)
analysis_thread.start()

#Main Loop
try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to read frame.")
            break
        frame_count += 1
        display = frame.copy()
        current_time = time.time()

        has_motion = motion_detector.detect(frame)

        try:
            while True: 
                result_data = result_queue.get_nowait()
                tracker.update_from_result(result_data)
        except queue.Empty:
            pass 

        #Trigger Analysis (Main Thread)
        should_trigger_analysis = (
            (has_motion or current_time - motion_detector.last_motion_time < ANALYSIS_POST_DETECTION_TIME or not tracker.is_valid()) and
            current_time - last_detection_trigger_time > DETECTION_COOLDOWN
        )

        if should_trigger_analysis:
            x_roi, y_roi, w_roi, h_roi = tracker.get_smoothed_position()

            #Extract ROI *in the main thread*
            analysis_frame_roi = None
            x1_offset_main, y1_offset_main = 0, 0
            if all(v is not None for v in [x_roi, y_roi, w_roi, h_roi]) and w_roi > ROI_MIN_SIZE and h_roi > ROI_MIN_SIZE:
                padding = int(ROI_PADDING_RATIO * max(w_roi, h_roi))
                x1_main = max(0, x_roi - padding)
                y1_main = max(0, y_roi - padding)
                x2_main = min(frame.shape[1], x_roi + w_roi + padding)
                y2_main = min(frame.shape[0], y_roi + h_roi + padding)

                if x2_main > x1_main and y2_main > y1_main:
                    analysis_frame_roi = frame[y1_main:y2_main, x1_main:x2_main].copy()
                    x1_offset_main, y1_offset_main = x1_main, y1_main 

            # Send either the full frame ROI or the extracted ROI
            frame_to_send = analysis_frame_roi if analysis_frame_roi is not None else frame.copy()
            if analysis_frame_roi is not None:
                 
                 x_roi_sent, y_roi_sent, w_roi_sent, h_roi_sent = 0, 0, analysis_frame_roi.shape[1], analysis_frame_roi.shape[0]
            else:
                 
                 x_roi_sent, y_roi_sent, w_roi_sent, h_roi_sent = x_roi, y_roi, w_roi, h_roi

            try:
                frame_queue.put_nowait((frame_to_send, x_roi_sent, y_roi_sent, w_roi_sent, h_roi_sent, x1_offset_main, y1_offset_main))
                last_detection_trigger_time = current_time
            except queue.Full:
                try:
                    frame_queue.get_nowait() 
                    frame_queue.put_nowait((frame_to_send, x_roi_sent, y_roi_sent, w_roi_sent, h_roi_sent, x1_offset_main, y1_offset_main))
                    last_detection_trigger_time = current_time
                except queue.Empty:
                    pass 

        #Get current detections for drawing (Main Thread)
        detections = tracker.get_detections()

        #Draw bounding boxes and labels (Main Thread)
        if detections and detections[0]['region']['x'] is not None:
            face = detections[0]
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            age = tracker.get_smoothed_age()
            gender, conf = tracker.get_smoothed_gender()
            if age is not None and gender is not None:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label_age = f"Age: {age}"
                gender_text = f"{gender} ({conf:.0f}%)" if conf else "Unknown"
                la_size = cv2.getTextSize(label_age, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                lg_size = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_width = max(la_size[0], lg_size[0])
                cv2.rectangle(display, (x, max(0, y - 45)),
                              (x + text_width + 10, y), (0, 0, 0), -1)
                cv2.putText(display, label_age, (x + 5, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, gender_text, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        #Calculate and display FPS (Main Thread)
        if frame_count % 30 == 0 and (time.time() - start_time) > 0:
            fps = frame_count / (time.time() - start_time)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Faces: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Age-Gender Detection - DeepFace (Threaded)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            motion_detector = MotionDetector(threshold=MOTION_THRESHOLD)

finally:
    #Cleanup
    print("Stopping threads...")
    stop_event.set() 
    try:
        frame_queue.put_nowait(None)
    except queue.Full:
        pass
    analysis_thread.join(timeout=2) 
    if analysis_thread.is_alive():
        print("Warning: Analysis thread did not stop gracefully.")
    cam.release()
    cv2.destroyAllWindows()
    print("Program ended successfully.")
