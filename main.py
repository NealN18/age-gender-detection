import cv2
import numpy as np
import time
import threading
import queue
import config
from models import dummy_img
from motion_detector import MotionDetector
from detection_tracker import DetectionTracker
from analysis_worker import analysis_worker

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

frame_count = 0
start_time = time.time()
fps = 0
last_detection_trigger_time = 0 

result_queue = queue.Queue(maxsize=2) 
stop_event = threading.Event()
frame_queue = queue.Queue(maxsize=2)

tracker = DetectionTracker(alpha=config.TRACKER_ALPHA, history_size=config.TRACKER_HISTORY_SIZE)
motion_detector = MotionDetector(threshold=config.MOTION_THRESHOLD)

print("Starting camera. Press 'q' to quit.")

for _ in range(5):
    ret, frame = cam.read()
    if ret:
        motion_detector.detect(frame)


analysis_thread = threading.Thread(target=analysis_worker, args=(frame_queue, result_queue, stop_event), daemon=True)
analysis_thread.start()


try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame_count += 1
        display_frame = frame.copy()
        current_time = time.time()

        has_motion = motion_detector.detect(frame)

        latest_result = None
        try:
            while True:
                latest_result = result_queue.get_nowait()
        except queue.Empty:
            pass
        
        if latest_result is not None:
            tracker.update_from_result(latest_result)

        should_trigger_standard = (
            (has_motion or 
             current_time - motion_detector.last_motion_time < config.ANALYSIS_POST_DETECTION_TIME or 
             not tracker.is_valid()) and
            current_time - last_detection_trigger_time > config.DETECTION_COOLDOWN
        )


        should_trigger_retry = tracker.should_retry_detection(current_time) and \
                               (current_time - last_detection_trigger_time > config.TRACKER_RETRY_COOLDOWN)

        should_trigger_analysis = should_trigger_standard or should_trigger_retry

        if should_trigger_analysis:
            use_roi = False
            x_tracked, y_tracked, w_tracked, h_tracked = tracker.get_smoothed_position()
            
            if (tracker.is_valid() and 
                all(coord is not None for coord in [x_tracked, y_tracked, w_tracked, h_tracked]) and
                w_tracked > config.ROI_MIN_SIZE and h_tracked > config.ROI_MIN_SIZE):
                use_roi = True
            
            analysis_frame_to_send = None
            x_offset_sent, y_offset_sent = 0, 0

            if use_roi and not should_trigger_retry:
                padding = int(config.ROI_PADDING_RATIO * max(w_tracked, h_tracked))
                x1_roi = max(0, x_tracked - padding)
                y1_roi = max(0, y_tracked - padding)
                x2_roi = min(frame.shape[1], x_tracked + w_tracked + padding)
                y2_roi = min(frame.shape[0], y_tracked + h_tracked + padding)

                if x2_roi > x1_roi and y2_roi > y1_roi:
                    analysis_frame_to_send = frame[y1_roi:y2_roi, x1_roi:x2_roi].copy()
                    x_offset_sent, y_offset_sent = x1_roi, y1_roi


            if analysis_frame_to_send is None:
                analysis_frame_to_send = frame.copy()
                x_offset_sent, y_offset_sent = 0, 0
                if should_trigger_retry:
                    tracker.consume_retry() 

            x_roi_sent, y_roi_sent = 0, 0 
            w_roi_sent, h_roi_sent = analysis_frame_to_send.shape[1], analysis_frame_to_send.shape[0]

            try:
                frame_queue.put_nowait((
                    analysis_frame_to_send, 
                    x_roi_sent, y_roi_sent, w_roi_sent, h_roi_sent, 
                    x_offset_sent, y_offset_sent
                ))
                last_detection_trigger_time = current_time
            except queue.Full:
                try:
                    frame_queue.get_nowait() 
                    frame_queue.put_nowait((
                        analysis_frame_to_send,
                        x_roi_sent, y_roi_sent, w_roi_sent, h_roi_sent,
                        x_offset_sent, y_offset_sent
                    ))
                    last_detection_trigger_time = current_time
                except queue.Empty:
                    pass 

        detections_to_draw = tracker.get_detections_for_drawing()

        if detections_to_draw:
            face_data = detections_to_draw[0]
            region = face_data['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            age = tracker.get_smoothed_age()
            gender, gender_conf = tracker.get_smoothed_gender()

            if age is not None and gender is not None:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                label_age = f"Age: {age}"
                gender_text = f"{gender} ({gender_conf:.0f}%)"

                la_size = cv2.getTextSize(label_age, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                lg_size = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_width = max(la_size[0], lg_size[0])

                cv2.rectangle(display_frame, (x, max(0, y - 45)),
                              (x + text_width + 10, y), (0, 0, 0), -1)
                
                cv2.putText(display_frame, label_age, (x + 5, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, gender_text, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Faces: {len(detections_to_draw)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Age-Gender Detection - DeepFace (Robust)", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            motion_detector = MotionDetector(threshold=config.MOTION_THRESHOLD)
            print("Motion detector background reset.")

finally:
    print("Stopping threads...")
    stop_event.set() 
    try:
        frame_queue.put_nowait(None)
    except queue.Full:
        pass
    analysis_thread.join(timeout=3)
    if analysis_thread.is_alive():
        print("Warning: Analysis thread did not stop gracefully.")

    cam.release()
    cv2.destroyAllWindows()
    print("Program ended successfully.")
