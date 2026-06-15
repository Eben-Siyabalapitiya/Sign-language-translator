import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
import urllib.request
from collections import deque, Counter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── DOWNLOAD MODEL FILES (one-time, cached after first run) ──
HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker.task"

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

if not os.path.exists(HAND_MODEL_PATH):
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

if not os.path.exists(POSE_MODEL_PATH):
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

# ── SETUP DETECTORS (new Tasks API) ──
hand_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.7,
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

pose_options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    min_pose_detection_confidence=0.5,
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# ── LOAD YOUR TRAINED MODEL ──
model = tf.keras.models.load_model("gesture_model.keras")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

prediction_buffer = deque(maxlen=10)

# ── HAND / POSE CONNECTION TOPOLOGY (hardcoded, standard MediaPipe layout) ──
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

POSE_CONNECTIONS = [
    (0,1),(0,4),(1,2),(2,3),(3,7),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(11,23),(12,14),(12,24),
    (13,15),(14,16),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
    (17,19),(18,20),(19,21),(20,22),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(27,31),(28,30),(28,32),(29,31),(30,32)
]

FINGERTIPS = [4, 8, 12, 16, 20]


def normalize_hand(landmarks, wrist):
    normalized = []
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
    distances = []
    for lm in landmarks:
        dist = np.sqrt((lm.x - wrist_x)**2 + (lm.y - wrist_y)**2)
        distances.append(dist)
    hand_size = max(distances) if max(distances) > 0 else 1
    for lm in landmarks:
        normalized.append((lm.x - wrist_x) / hand_size)
        normalized.append((lm.y - wrist_y) / hand_size)
        normalized.append((lm.z - wrist_z) / hand_size)
    return normalized


def get_body_relative_position(hand_wrist, pose_landmarks):
    if not pose_landmarks:
        return [0, 0, 0, 0, 0, 0]
    nose       = pose_landmarks[0]
    l_shoulder = pose_landmarks[11]
    r_shoulder = pose_landmarks[12]
    shoulder_width = abs(l_shoulder.x - r_shoulder.x)
    if shoulder_width < 0.01:
        shoulder_width = 0.01
    wx, wy = hand_wrist.x, hand_wrist.y
    return [
        (wx - nose.x)       / shoulder_width,
        (wy - nose.y)       / shoulder_width,
        (wx - l_shoulder.x) / shoulder_width,
        (wy - l_shoulder.y) / shoulder_width,
        (wx - r_shoulder.x) / shoulder_width,
        (wy - r_shoulder.y) / shoulder_width,
    ]


def draw_rounded_rect(img, x, y, w, h, r, color, alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, (x+r, y),   (x+w-r, y+h), color, -1)
    cv2.rectangle(overlay, (x,   y+r), (x+w,   y+h-r), color, -1)
    cv2.circle(overlay, (x+r,   y+r),   r, color, -1)
    cv2.circle(overlay, (x+w-r, y+r),   r, color, -1)
    cv2.circle(overlay, (x+r,   y+h-r), r, color, -1)
    cv2.circle(overlay, (x+w-r, y+h-r), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


def draw_progress_bar(frame, x, y, w, h, progress, color_fill, color_bg):
    cv2.rectangle(frame, (x, y), (x+w, y+h), color_bg, -1)
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x+fill_w, y+h), color_fill, -1)


def draw_hand(img, landmarks, w, h):
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(img, pts[s], pts[e], (60, 140, 255), 2)
    for i, pt in enumerate(pts):
        color = (200, 160, 255) if i in FINGERTIPS else (160, 100, 220)
        cv2.circle(img, pt, 4, color, -1)


def draw_pose(img, landmarks, w, h):
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
    for s, e in POSE_CONNECTIONS:
        if s < len(pts) and e < len(pts):
            cv2.line(img, pts[s], pts[e], (45, 45, 45), 1)
    for i, pt in enumerate(pts):
        if i in (0, 11, 12):  # nose, shoulders
            cv2.circle(img, pt, 3, (80, 220, 180), -1)


WHITE  = (255, 255, 255)
GREEN  = (80, 200, 120)
DARK   = (12, 12, 12)
LGRAY  = (140, 140, 140)
ACCENT = (255, 200, 0)

word_history = []
last_word    = ""


def process_frame(img):
    global last_word

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hand_result = hand_detector.detect(mp_image)
    pose_result = pose_detector.detect(mp_image)

    display_word   = ""
    confidence_val  = 0.0
    num_hands       = 0

    pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

    if hand_result.hand_landmarks:
        num_hands = len(hand_result.hand_landmarks)

        for hand_lms in hand_result.hand_landmarks:
            draw_hand(img, hand_lms, w, h)

        h1 = hand_result.hand_landmarks[0]
        wrist1 = h1[0]
        hand1_data = normalize_hand(h1, wrist1)
        body_rel = get_body_relative_position(wrist1, pose_landmarks)

        hand2_data = [0.0] * 63
        if num_hands > 1:
            h2 = hand_result.hand_landmarks[1]
            hand2_data = normalize_hand(h2, h2[0])

        input_data = np.array(hand1_data + hand2_data + body_rel).reshape(1, -1)
        preds = model.predict(input_data, verbose=0)
        confidence_val = float(np.max(preds))
        predicted_class = encoder.inverse_transform([np.argmax(preds)])[0]
        prediction_buffer.append(predicted_class)

        if confidence_val > 0.85:
            smoothed = Counter(prediction_buffer).most_common(1)[0][0]
            display_word = smoothed.upper()
            if display_word != last_word:
                last_word = display_word
                if not word_history or word_history[-1] != display_word:
                    word_history.append(display_word)
                    if len(word_history) > 5:
                        word_history.pop(0)

    if pose_landmarks:
        draw_pose(img, pose_landmarks, w, h)

    # Prediction card
    card_w, card_h = min(380, w-20), 80
    card_x = (w - card_w) // 2
    card_y = h - card_h - 14
    draw_rounded_rect(img, card_x, card_y, card_w, card_h, 12, DARK, alpha=0.85)

    if display_word and confidence_val > 0.85:
        (tw, _), _ = cv2.getTextSize(display_word, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
        tx = card_x + (card_w - tw) // 2
        cv2.putText(img, display_word, (tx, card_y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, GREEN, 3)
        draw_progress_bar(img, card_x+20, card_y+card_h-12,
                          card_w-40, 5, confidence_val, GREEN, (40,40,40))
    else:
        (tw, _), _ = cv2.getTextSize("Detecting...", cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        tx = card_x + (card_w - tw) // 2
        cv2.putText(img, "Detecting...", (tx, card_y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, LGRAY, 2)

    # History
    if word_history:
        hh = 28 + len(word_history) * 22
        draw_rounded_rect(img, 8, 8, 185, hh, 10, DARK, alpha=0.72)
        cv2.putText(img, "HISTORY", (18, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, LGRAY, 1)
        for i, word in enumerate(reversed(word_history)):
            cv2.putText(img, word, (18, 40 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 if i == 0 else 0.42,
                        WHITE if i == 0 else LGRAY,
                        2 if i == 0 else 1)

    # Confidence badge
    conf_str = f"{confidence_val*100:.0f}% CONF"
    cc = GREEN if confidence_val > 0.85 else ACCENT if confidence_val > 0.5 else LGRAY
    (cw2,_),_ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    draw_rounded_rect(img, w-cw2-24, 8, cw2+16, 24, 5, DARK, alpha=0.72)
    cv2.putText(img, conf_str, (w-cw2-16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, cc, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


with gr.Blocks(title="Sign Language Translator") as demo:
    gr.Markdown("# Sign Language Translator")
    gr.Markdown("Real-time sign language recognition using MediaPipe and TensorFlow. Allow camera access and hold a sign steady for about a second.")
    with gr.Row():
        webcam = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
        output = gr.Image(label="Sign Language Translator")
    webcam.stream(fn=process_frame, inputs=webcam, outputs=output)

if __name__ == "__main__":
    demo.launch()