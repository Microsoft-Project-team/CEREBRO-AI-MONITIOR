
import cv2
import time
import threading
import os
from collections import deque
from datetime import datetime
from emotion_detection import emotion_detect
from gaze_detection import main as gaze_main
from collections import Counter
import screen_brightness_control as sbc
from pygetwindow import getWindowsWithTitle
from playsound import playsound
import pygame
import ctypes
import json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from datetime import timedelta
import logging


logging.basicConfig(
    filename='session_graph.log',        # .log file to store logs
    level=logging.INFO,                  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- CONFIG ---
TARGET_FPS = 20
RECORD_SECONDS = 10
FRAME_COUNT_LIMIT = TARGET_FPS * RECORD_SECONDS
VIDEO_DIR = "temp_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)


# === Frame Analyzer ===
def extract_frames_from_video(video_path, target_fps=10):
    logging.info(f"Started processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / target_fps)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("Finished reading video or encountered end of stream.")
            break

        if count % interval == 0:
            emo = emotion_detect(frame)
            gaze = gaze_main(frame)

            frame_info = {
                "face_detected": emo.get("face_detected", True),
                "dominant_emotion": emo.get("dominant_emotion", "neutral"),
                "confidence": emo.get("confidence", 0),
                "gaze_direction": gaze.get("gaze_direction", "center"),
                "eye_direction": gaze.get("eye_direction", "center"),
                "head_pose": gaze.get("head_pose", None),
                "blinking": gaze.get("blinking", False)
            }

            logging.debug(f"Frame {count}: {frame_info}")
            frames.append(frame_info)

        count += 1

    cap.release()
    logging.info(f"Released video file: {video_path}")
    return frames


def analyze_video_frames(frames, fps=10):
    logging.info("Started analyzing video frames...")
    
    global session_summary
    total_frames = len(frames)
    frame_duration = 1 / fps
    neg_emotions = {"sad", "angry"}

    face_count = 0
    blink_count = 0
    sad_count = 0
    attention_count = 0
    distraction_count = 0
    high_load_frames = 0
    emotion_list = []
    emotion_cnt = {}
    prev_gaze = None
    gaze_change_count = 0
    emotion_confidence = 0

    for i, frame in enumerate(frames):
        logging.debug(f"Processing frame {i+1}/{total_frames}")
        
        if frame["face_detected"]:
            face_count += 1

        emotion = frame["dominant_emotion"].lower()
        emotion_cnt[emotion] = emotion_cnt.get(emotion, 0) + 1
        emotion_list.append(emotion)
        if emotion == "sad":
            sad_count += 1
        emotion_flag = 1 if emotion in neg_emotions else 0
        emotion_confidence += frame['confidence']

        gaze = frame["gaze_direction"]
        if prev_gaze is not None and gaze != prev_gaze:
            gaze_change_count += 1
        prev_gaze = gaze
        gaze_flag = 1 if gaze != "center" else 0

        head_pose_data = frame["head_pose"]

        blink_flag = 1 if frame["blinking"] else 0
        if blink_flag:
            blink_count += 1

        if frame["face_detected"] and gaze == "center":
            attention_count += 1
        else:
            distraction_count += 1

        score = emotion_flag + blink_flag + gaze_flag
        if score >= 2:
            high_load_frames += 1

    sorted_emotion = list(sorted(emotion_cnt.items(), key=lambda item: item[1], reverse=True))
    face_presence = round((face_count / total_frames) * 100)
    face_presence_time = round(face_count * frame_duration, 1)
    dominant_emt = sorted_emotion[0][0]
    emotion_confidence = round((emotion_confidence / (len(frames)*100))*100, 1)
    high_load_duration = round(high_load_frames * frame_duration, 1)
    attention_time = round(attention_count * frame_duration, 1)
    distraction_time = round(distraction_count * frame_duration, 1)
    sad_time = round(sad_count * frame_duration, 1)
    blink_rate = round((blink_count * 60) / 10)
    blink_rate_str = blink_rate
    head_pose = head_pose_data

    cognitive_load = "high" if high_load_frames >= total_frames * 0.5 else \
                     "medium" if high_load_frames >= total_frames * 0.2 else \
                     "low"

    unique_emotions = set(emotion_list)
    if emotion_list[-1] == "sad":
        mood_stability = "Downtrend"
    elif len(unique_emotions) > 3:
        mood_stability = "Volatile"
    else:
        mood_stability = "Stable"

    emotion_counter = Counter(emotion_list)
    current_emotion = emotion_counter.most_common(1)[0][0]

    if blink_rate > 25:
        gaze_estimation = "Drowsy"
    elif face_presence != "100%" or distraction_time > attention_time:
        gaze_estimation = "Distracted"
    else:
        gaze_estimation = "Looking at screen"

    session_summary["attention_time"] = int((session_summary['attention_time']+int(face_presence_time))%100)

    logging.info(f"Analysis complete. Summary: {result}")
    return {
        "face_presence": face_presence,
        "face_presence_time": face_presence_time,
        "dominant_emotion": dominant_emt,
        "emotion_confidence": emotion_confidence,
        "cognitive_load": cognitive_load,
        "high_load_duration": high_load_duration,
        "attention_time": attention_time,
        "distraction_time": distraction_time,
        "mood_stability": mood_stability,
        "blink_rate": blink_rate_str,
        "sad_time": sad_time,
        "current_emotion": current_emotion,
        "gaze_estimation": gaze_estimation,
        "head_pose": head_pose
    }


result = None
def analyze_and_delete(video_path):
    frames = extract_frames_from_video(video_path)
    global result
    result = analyze_video_frames(frames)
    os.remove(video_path)

def minimize_all_windows():
    for win in getWindowsWithTitle(''):
        try:
            title = win.title.lower()
            if "visual studio code" in title or "vscode" in title:
                continue
            win.minimize()
        except Exception as e:
            continue

def play_calming_music():
    def _play():
        try:
            pygame.mixer.init()
            pygame.mixer.music.load("Calm.wav")
            pygame.mixer.music.play(-1)
        except Exception as e:
            print("[ERROR] Music:", e)
    threading.Thread(target=_play, daemon=True).start()

def stop_music():
    try:
        pygame.mixer.music.stop()
    except:
        pass

def reduce_brightness():
    try:
        sbc.set_brightness(30)
    except:
        pass

def lock_workstation():
    ctypes.windll.user32.LockWorkStation()

def break_suggest():
    message = "Take a 5-minute break "
    title = "Break Reminder"
    ctypes.windll.user32.MessageBoxW(0, message, title, 0x40 | 0x1)

def focus_suggest():
    message = "Please Focus on the work"
    title = "Focus Reminder"
    ctypes.windll.user32.MessageBoxW(0, message, title, 0x40 | 0x1)

distraction_time = 0
sad_time = 0
high_load = 0
persence_time = 0
no_face_time = 0

def action_trigger():
    global distraction_time, sad_time, high_load, persence_time, no_face_time, session_summary

    logging.info("Trigger evaluation started...")

    with open("graph.json", "r") as f:
        graph_data = json.load(f)

    if result:
        distraction_time += result['distraction_time']
        sad_time += result['sad_time']
        high_load += result['high_load_duration']
        persence_time += result['face_presence_time']
        no_face_time += 10 - result['face_presence_time']

        logging.debug(f"Distraction Time: {distraction_time}")
        logging.debug(f"Sad Time: {sad_time}")
        logging.debug(f"High Load Duration: {high_load}")
        logging.debug(f"Presence Time: {persence_time}")
        logging.debug(f"No Face Time: {no_face_time}")

        flag = False

        if distraction_time > 20:
            minimize_all_windows()
            focus_suggest()
            distraction_time = 0
            graph_data['system_action'] = "Minimize all windows and suggest focus popup"
            flag = True
            logging.info("Distraction trigger: Minimized windows and suggested focus popup.")

        if sad_time > 60:
            break_suggest()
            play_calming_music()
            session_summary["break_suggest"] += 1
            sad_time = 0
            graph_data['system_action'] = "Played a clam music and suggest take break popup"
            flag = True
            logging.info("Sadness trigger: Played calming music and suggested a break.")

        if high_load > 90:
            reduce_brightness()
            high_load = 0
            graph_data['system_action'] = "Reduced brightness"
            flag = True
            logging.info("High cognitive load trigger: Reduced screen brightness.")

        if persence_time > 40*60:
            break_suggest()
            session_summary["break_suggest"] += 1
            persence_time = 0
            graph_data['system_action'] = "Take break suggested"
            flag = True
            logging.info("Long presence trigger: Suggested a break.")

        if no_face_time > 60:
            lock_workstation()
            no_face_time = 0
            graph_data['system_action'] = "locked workstation"
            flag = True
            logging.info("No face detected trigger: Locked workstation.")

        if flag:
            session_summary["system_action"] += 1
            logging.info(f"System action triggered. Total actions so far: {session_summary['system_action']}")

    with open('data.json', 'w') as f:
        json.dump(graph_data, f, indent=2)
        logging.info("Updated graph data written to data.json")

def console_display():
    if result:
        with open("graph.json", "r") as f:
            graph_data = json.load(f)

        if result['face_presence'] > 88.0:
            graph_data['face_presence'] = "Present"
            logging.info("Face presence detected above 88% – Marked as Present.")
        else:
            graph_data['face_presence'] = "Not Present"
            logging.info("Face presence below 88% – Marked as Not Present.")

        graph_data["gaze_status"] = result['gaze_estimation']
        graph_data["dominant_emotion"] = result["dominant_emotion"]
        graph_data["dominant_confidence"] = result["emotion_confidence"]
        graph_data['mood_stability'] = result["mood_stability"]
        graph_data['cognitive_load'] = result["cognitive_load"]
        graph_data["blink_rate"] = result["blink_rate"] // 50
        graph_data["active_time"] = result["face_presence_time"]
        graph_data["head_pose"] = result['head_pose']

        logging.info("Updated console display data: "
                     f"Gaze={graph_data['gaze_status']}, "
                     f"Emotion={graph_data['dominant_emotion']} ({graph_data['dominant_confidence']}%), "
                     f"Load={graph_data['cognitive_load']}, "
                     f"Mood={graph_data['mood_stability']}, "
                     f"Blink rate={graph_data['blink_rate']}, "
                     f"Head pose={graph_data['head_pose']}")
        with open("graph.json", "r") as f:
            data_console = json.load(f)
        
        print(json.dumps(data_console, indent = 2))
        with open('graph.json', 'w') as f:
            json.dump(graph_data, f, indent=2)
            logging.info("graph.json updated with current console display data.")

data_graph = {
  "status": "false",
  "emotion_counts": {
    "happy": 0,
    "sad": 0,
    "neutral": 0,
    "angry": 0
  },
  "cognitive_load_levels": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "face_presence_confidence": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "attention_time_level": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "count_len": 0,
  "time": 0
}

reset_data_graph1 = {
  "status": "false",
  "emotion_counts": {
    "happy": 0,
    "sad": 0,
    "neutral": 0,
    "angry": 0
  },
  "cognitive_load_levels": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "face_presence_confidence": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "attention_time_level": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "count_len": 0,
  "time": 0
}

reset_data_graph = {
  "status": "false",
  "emotion_counts": {
    "happy": 0,
    "sad": 0,
    "neutral": 0,
    "angry": 0
  },
  "cognitive_load_levels": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "face_presence_confidence": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "attention_time_level": {
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "count_len": 0,
  "time": 0
}

count_len = 0

def update_graph():
    global data_graph, count_len, session_summary
    if result:
        if result['dominant_emotion'] in data_graph['emotion_counts']:
            data_graph['emotion_counts'][result['dominant_emotion']] += int(result['emotion_confidence'])
            logging.info(f"Updated emotion count: {result['dominant_emotion']} += {int(result['emotion_confidence'])}")

        if result['cognitive_load'] in data_graph['cognitive_load_levels']:
            data_graph["cognitive_load_levels"][result["cognitive_load"]] += 1
            logging.info(f"Incremented cognitive load level: {result['cognitive_load']}")

        if result["face_presence"] > 90:
            face_level = "high"
        elif result["face_presence"] > 60 and result["face_presence"] < 90:
            face_level = "medium"
        else:
            face_level = "low"
        if face_level in data_graph['face_presence_confidence']:
            data_graph["face_presence_confidence"][face_level] += int(result["face_presence"])
            logging.info(f"Updated face presence confidence level '{face_level}' += {int(result['face_presence'])}")

        if result["face_presence_time"] >= 8:
            face_attention_level = "high"
        elif result["face_presence_time"] >= 5 and result["face_presence_time"] < 8:
            face_attention_level = "medium"
        else:
            face_attention_level = "low"
        if face_attention_level in data_graph['attention_time_level']:
            data_graph["attention_time_level"][face_attention_level] += 1
            logging.info(f"Incremented attention time level: {face_attention_level}")

        prev_fp = session_summary['face_persence']
        session_summary['face_persence'] = int((session_summary['face_persence'] + int(result["face_presence"])) % 100)
        logging.info(f"Updated session_summary face_persence: {prev_fp} -> {session_summary['face_persence']}")

        if result["cognitive_load"] == "high":
            session_summary["cognitive_peak_load"] += 1
            logging.info("Incremented cognitive_peak_load in session_summary")

        count_len += 1
        logging.info(f"Frame count_len incremented to: {count_len}")

session_summary = {
    "heading": "CEREBRO AI Workststion",
    "session_time": None,
    "face_persence": 0,
    "attention_time": 0,
    "emotion_data": None,
    "cognitive_peak_load": 0,
    "break_suggest": 0,
    "system_action": 0
}
def summary_data():
    print(json.dumps(session_summary, indent = 2))

def generate_session_summary_pdf(data, output_path="session_summary.pdf"):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    y = height - inch

    def draw_line(text, value=""):
        nonlocal y
        c.setFont("Helvetica", 11)
        c.drawString(50, y, f"{text:<25} : {value}")
        y -= 20
        logging.info(f"Added line to PDF - {text}: {value}")

    try:
        # Title
        c.setFont("Helvetica-Bold", 16)
        title_text = "SESSION SUMMARY – " + data["heading"].upper()
        c.drawCentredString(width / 2, y, title_text)
        logging.info(f"Added title to PDF: {title_text}")
        y -= 40

        # Data entries
        draw_line("Total Time", data["session_time"].split(": ", 1)[1])
        draw_line("Face Presence", f"{data['face_persence']}% active")
        draw_line("Focused Screen Time", f"{data['attention_time']} seconds")

        # Emotions
        emotions = data["emotion_data"]["emotion_counts"]
        emotion_breakdown = ", ".join(
            [f"{k.capitalize()} ({v})" for k, v in emotions.items() if v > 0]
        )
        draw_line("Emotion Breakdown", emotion_breakdown if emotion_breakdown else "None")

        draw_line("Cognitive Load Peaks", data["cognitive_peak_load"])
        draw_line("Breaks Suggested", data["break_suggest"])
        draw_line("System Actions Taken", data["system_action"])

        # Save PDF
        c.showPage()
        c.save()
        logging.info(f"Session summary PDF saved successfully to: {output_path}")
        print(f"PDF saved as {output_path}")

    except Exception as e:
        logging.error(f"Error generating session summary PDF: {e}")
        raise


# === Main Continuous Live Camera Recorder ===
def live_camera_monitor():
    global data_graph, count_len, reset_data_graph, session_summary
    cap = cv2.VideoCapture(0)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    buffer = deque(maxlen=FRAME_COUNT_LIMIT)
    last_record_time = time.time()

    session_start = time.time()
    graph_time = time.time()
    logging.info("Camera monitoring started.")
    print("🎥 Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame capture failed.")
            break

        cv2.imshow("Live Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User quit the camera session.")
            break

        buffer.append(frame)

        if time.time() - last_record_time >= RECORD_SECONDS:
            last_record_time = time.time()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{VIDEO_DIR}/vid_{timestamp}.avi"
            out = cv2.VideoWriter(path, fourcc, TARGET_FPS, (width, height))
            for f in buffer:
                out.write(f)
            out.release()

            logging.info(f"Saved video buffer to: {path}")

            threading.Thread(target=analyze_and_delete, args=(path,)).start()
            logging.info(f"Analysis thread started for: {path}")

            print(json.dumps(result, indent=2))

            try:
                console_display()
                action_trigger()
                update_graph()
            except Exception as e:
                logging.error(f"Error during analysis and update cycle: {e}")

            if time.time() - graph_time >= 30:
                logging.info("30-second graph update triggered.")
                print("Saving graph data after 30 seconds...")

                try:
                    data_graph['status'] = "true"
                    data_graph['count_len'] = count_len
                    data_graph['time'] += 30

                    # Normalize values
                    for k in data_graph["emotion_counts"]:
                        data_graph["emotion_counts"][k] %= 100
                    for k in data_graph["cognitive_load_levels"]:
                        data_graph["cognitive_load_levels"][k] %= 100
                    for k in data_graph["face_presence_confidence"]:
                        data_graph["face_presence_confidence"][k] %= 100
                    for k in data_graph["attention_time_level"]:
                        data_graph["attention_time_level"][k] %= 100

                    # Save to JSON files
                    with open('dashboard-app\\public\\graph_update.json', 'w') as f:
                        json.dump(data_graph, f, indent=2)
                    logging.info("Saved graph_update.json")

                    with open('dashboard-app\\public\\graph_update.json', 'r') as src_file:
                        source_data = json.load(src_file)

                    with open('dashboard-app\\public\\graph_update1.json', 'w') as f:
                        json.dump(source_data, f, indent=2)
                    logging.info("Copied graph_update.json to graph_update1.json")

                    count_len = 0
                    data_graph = reset_data_graph1.copy()
                    graph_time = time.time()

                except Exception as e:
                    logging.error(f"Error updating graph files: {e}")

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Camera released and windows destroyed.")

    session_end = time.time()
    session_duration = int(session_end - session_start)
    hours, remainder = divmod(session_duration, 3600)
    minutes, seconds = divmod(remainder, 60)

    try:
        with open('dashboard-app\\public\\graph_update.json', 'w') as f:
            json.dump(reset_data_graph, f, indent=2)
        logging.info("Reset graph_update.json on session end.")

        emotion_summary = {
            "emotion_counts": {"happy": 0, "sad": 0, "neutral": 0, "angry": 0}
        }

        with open('dashboard-app\\public\\graph_update1.json', 'r') as src_file:
            source_data = json.load(src_file)

        for emotion in emotion_summary['emotion_counts']:
            emotion_summary['emotion_counts'][emotion] = source_data['emotion_counts'][emotion]

        session_summary["emotion_data"] = emotion_summary
        session_summary['session_time'] = f"Session Duration: {hours}h {minutes}m {seconds}s"
        logging.info(f"Session duration calculated: {session_summary['session_time']}")
        # print(f"Session Duration: {hours}h {minutes}m {seconds}s")

        generate_session_summary_pdf(session_summary)
        logging.info("PDF generation completed.")


    except Exception as e:
        logging.error(f"Error during final summary handling: {e}")

if __name__ == "__main__":
    live_camera_monitor()



