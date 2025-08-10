# Age and Gender Detection with DeepFace (Threaded)

This project performs real-time age and gender detection using a webcam. It leverages the [DeepFace](https://github.com/serengil/deepface) library for analysis and employs a multi-threaded approach to separate the potentially slow analysis process from the main video capture loop, aiming for smoother video display.

## Features

*   **Real-time Detection:** Processes webcam feed in real-time.
*   **Age & Gender Prediction:** Uses DeepFace to estimate age and gender.
*   **Motion Detection:** Implements background subtraction (`cv2.createBackgroundSubtractorMOG2`) to trigger analysis only when motion is detected or recently occurred, reducing unnecessary computations.
*   **Tracking & Smoothing:** Tracks detected faces using a simple custom tracker (`DetectionTracker`) and applies exponential smoothing to age, gender, and bounding box coordinates for more stable results.
*   **Threading:** Offloads the DeepFace analysis to a separate thread using Python's `threading` and `queue` modules. This keeps the main OpenCV loop responsive for video display and motion detection.
*   **Performance Considerations:** Includes configurable timeouts, cooldowns, and ROI (Region of Interest) extraction to manage performance.

## How It Works

1.  The main thread captures video frames from the default webcam.
2.  A `MotionDetector` checks for significant changes in the scene.
3.  A `DetectionTracker` maintains the last known position and attributes (age, gender) of detected faces.
4.  Based on motion, tracker validity, and cooldown timers, the main thread decides whether to trigger a new analysis.
5.  If triggered, the main thread extracts a Region of Interest (ROI) around the tracked face (or uses the full frame if no tracker is available) and places this frame onto a queue (`frame_queue`).
6.  A separate analysis thread (`analysis_worker`) waits for frames on `frame_queue`.
7.  When a frame arrives, the analysis thread runs `DeepFace.analyze` on it.
8.  The analysis results (age, gender, bounding box) are placed onto another queue (`result_queue`).
9.  The main thread checks `result_queue` for new results each frame and updates the `DetectionTracker`.
10. The tracked information is used to draw bounding boxes and labels on the displayed frame.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/NealN18/age-gender-detection.git
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: DeepFace will automatically download the required models (e.g., for OpenCV face detection, age, gender prediction) on the first run.*

3.  **(Optional) Configure Paths:**
    *   Modify the `os.environ['DEEPFACE_HOME']` path in `config.py` if you want to specify a custom location for DeepFace model caching.

## Usage

1.  Ensure your webcam is connected and accessible.
2.  Run the script:
    ```bash
    python main.py
    ```
3.  A window will open showing the webcam feed with bounding boxes and predicted age/gender labels.
4.  Press `q` to quit the application.
5.  Press `r` to manually reset the motion detector's background model.

## Configuration

Several constants control the behavior and can be adjusted in `config.py`:

*   `ROI_MIN_SIZE`: Minimum size for a detected face region.
*   `ROI_PADDING_RATIO`: Padding added around the tracked ROI when extracting for analysis.
*   `MOTION_THRESHOLD`: Sensitivity of the motion detector.
*   `MOTION_POST_DETECTION_TIME`: Time to continue triggering analysis after motion stops.
*   `ANALYSIS_POST_DETECTION_TIME`: Time to continue analysis if tracker becomes invalid but no recent motion.
*   `TRACKER_HISTORY_SIZE`: Number of recent detections used for smoothing.
*   `TRACKER_ALPHA`: Weight for the most recent value in smoothing (higher = less smoothing).
*   `DETECTION_COOLDOWN`: Minimum time interval between analysis triggers.
*   `ANALYSIS_TIMEOUT`: Time after which tracker data is considered stale.
*   `TRACKER_RETRY_ATTEMPTS`: Number of times to retry analysis if a face is lost.
*   `TRACKER_RETRY_COOLDOWN`: Minimum time interval between retry attempts.
*   DeepFace settings like `detector_backend` ('opencv', 'ssd', etc.) and `align` (True/False) can be adjusted in `analysis_worker.py`.

## Dependencies

See `requirements.txt` for the list of required Python packages.

## Acknowledgements

*   [DeepFace](https://github.com/serengil/deepface) for the powerful face analysis library.
*   [OpenCV](https://opencv.org/) for comprehensive computer vision tools.
