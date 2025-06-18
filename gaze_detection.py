import os
import cv2
import json
import numpy as np
import mediapipe as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# EAR: Eye Aspect Ratio for blink detection
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Head-based gaze direction
def get_head_direction(landmarks, w, h):
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
    pitch = round((landmarks[10].y - landmarks[152].y) * 90, 1)
    yaw = round((landmarks[234].x - landmarks[454].x) * 90, 1)
    roll = 0.0
    eye_center = (left_eye + right_eye) / 2
    dx = nose[0] - eye_center[0]
    dy = nose[1] - eye_center[1]

    if abs(dx) > 30:
        direction = "right" if dx > 0 else "left"
    elif abs(dy) > 30:
        direction =  "down" if dy > 0 else "up"
    else:
        direction = "center"
    return direction, {"pitch":pitch, 'yaw':yaw, 'roll':roll}

# Iris-based eye direction
def get_eye_direction(landmarks, w, h):
    left_eye_inner = np.array([landmarks[133].x * w, landmarks[133].y * h])
    left_eye_outer = np.array([landmarks[33].x * w, landmarks[33].y * h])
    left_iris = np.array([landmarks[468].x * w, landmarks[468].y * h])

    right_eye_inner = np.array([landmarks[362].x * w, landmarks[362].y * h])
    right_eye_outer = np.array([landmarks[263].x * w, landmarks[263].y * h])
    right_iris = np.array([landmarks[473].x * w, landmarks[473].y * h])

    def get_pos(iris, inner, outer):
        x_ratio = (iris[0] - inner[0]) / (outer[0] - inner[0] + 1e-6)
        if x_ratio < 0.35:
            return "left"
        elif x_ratio > 0.65:
            return "right"
        else:
            return "center"

    left_pos = get_pos(left_iris, left_eye_inner, left_eye_outer)
    right_pos = get_pos(right_iris, right_eye_inner, right_eye_outer)

    # Combine left and right eyes
    if left_pos == right_pos:
        return left_pos
    else:
        return f"{left_pos}-{right_pos}"


def main(frame):
    try:
        # frame = cv2.imread(img)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        output = {
            "status": "success",
            "gaze_direction": "undetected",
            "eye_direction": "undetected",
            "head_pose": "undetected",
            "blinking": None
        }

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            # Head-based gaze direction
            direction, head_pose = get_head_direction(lm, w, h)
            output["gaze_direction"], output["head_pose"] = direction, head_pose

            # Eye direction from iris position
            output["eye_direction"] = get_eye_direction(lm, w, h) 
            # EAR (blinking) using left eye landmarks
            left_eye_ids = [362, 385, 387, 263, 373, 380]
            left_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in left_eye_ids])
            ear = eye_aspect_ratio(left_eye)
            output["blinking"] = True if ear < 0.20 else False
        return output
    except Exception as e:
        print(str(e))
        return {
            "status": "error",
            "error": str(e)
        }

# output = main(r"C:\\Users\\BOBBY\\Pictures\\Screenshots\\Screenshot 2025-03-20 190110.png")
# print(json.dumps(output, indent=2))
