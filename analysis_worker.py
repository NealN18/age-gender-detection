import cv2
import time
import logging
import queue
from deepface import DeepFace
from config import ROI_MIN_SIZE, DETECTION_COOLDOWN 

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
                if frame_for_analysis is None or frame_for_analysis.size == 0:
                     raise ValueError("Received empty or None frame for analysis.")
                
                rgb = cv2.cvtColor(frame_for_analysis, cv2.COLOR_BGR2RGB)
                
                raw_result = DeepFace.analyze(
                    rgb,
                    actions=['age', 'gender'],
                    enforce_detection=True, 
                    silent=True,
                    detector_backend='opencv',
                    align=True
                )
                
                faces_results = raw_result if isinstance(raw_result, list) else [raw_result]

                if faces_results:
                    for face in faces_results:
                        if 'region' in face:
                            face['region']['x'] += x1_offset_received
                            face['region']['y'] += y1_offset_received
                
                valid_faces = []
                for face in faces_results:
                    region = face.get("region", {})
                    if (all(k in region for k in ['x', 'y', 'w', 'h']) and
                        region['w'] >= ROI_MIN_SIZE and region['h'] >= ROI_MIN_SIZE):
                        valid_faces.append(face)

                result_data = {
                    'face': valid_faces[0] if valid_faces else {},
                    'timestamp': current_time
                }

                try:
                    result_queue.put_nowait(result_data)
                except queue.Full:
                    try:
                        result_queue.get_nowait()
                        result_queue.put_nowait(result_data)
                    except queue.Empty:
                        pass
                last_analysis_time = current_time 
            except Exception as e:
                result_data = {'face': {}, 'timestamp': current_time}
                try:
                    result_queue.put_nowait(result_data)
                except (queue.Full, queue.Empty):
                    pass
                logging.debug(f"Analysis error (DeepFace): {type(e).__name__}: {e}")
            finally:
                frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
             print(f"Unexpected error in analysis thread loop: {type(e).__name__}: {e}")
             logging.error(f"Unexpected error in analysis thread loop: {type(e).__name__}: {e}")

             try:
                 result_queue.put_nowait({'face': {}, 'timestamp': time.time()})
             except:
                 pass
             break
    print("Analysis thread stopped.")
