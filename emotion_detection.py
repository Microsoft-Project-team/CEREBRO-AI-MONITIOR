import os
import sys
import warnings
import cv2
import json
from deepface import DeepFace


def emotion_detect(frame):
    try:
        # img = cv2.imread(frame)
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        dominant_emotion = result[0]['dominant_emotion']
        emotions = result[0]['emotion']
        confidence = round(emotions[dominant_emotion], 2)
        face_detected = False
        if result[0]["face_confidence"] > 0.90:
            face_detected = True

        return {
            "status": "success",
            "face_detected": face_detected,
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
        }

    except Exception as e:
        return {
            "status": "error",
        }
        print(str(e))
    

# frame = r"C:\\Users\\BOBBY\\Pictures\\Screenshots\\Screenshot 2025-03-20 190110.png"

# data = emotion_detect(frame)
# print(json.dumps(data, indent = 2))