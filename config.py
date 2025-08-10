import os
import logging

#Configuration 
# Set environment and suppress TensorFlow warnings
os.environ['DEEPFACE_HOME'] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Configure logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)
#Constants
ROI_MIN_SIZE = 30
ROI_PADDING_RATIO = 0.3 
MOTION_THRESHOLD = 0.015
MOTION_POST_DETECTION_TIME = 1.0
ANALYSIS_POST_DETECTION_TIME = 2.5 
TRACKER_HISTORY_SIZE = 3 
TRACKER_ALPHA = 0.7 
DETECTION_COOLDOWN = 0.1 
ANALYSIS_TIMEOUT = 1.5 
TRACKER_RETRY_ATTEMPTS = 2
TRACKER_RETRY_COOLDOWN = 0.3 

