import numpy as np
from deepface import DeepFace

try:
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    DeepFace.analyze(dummy_img, actions=['age', 'gender'], enforce_detection=False, silent=True, detector_backend='opencv', align=True)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)
