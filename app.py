import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque, Counter

mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
pose  = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

model = tf.keras.models.load_model("gesture_model.keras")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

prediction_buffer = deque(maxlen=10)


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
    if pose_landmarks is None:
        return [0, 0, 0, 0, 0, 0]
    lms        = pose_landmarks.landmark
    nose       = lms[mp_pose.PoseLandmark.NOSE]
    l_shoulder = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
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

    pose_result  = pose.process(rgb)
    hands_result = hands.process(rgb)

    display_word  = ""
    confidence_val = 0.0
    num_hands = 0

    if hands_result.multi_hand_landmarks:
        num_hands = len(hands_result.multi_hand_landmarks)
        for h_lm in hands_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img, h_lm, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

        h1 = hands_result.multi_hand_landmarks[0]
        wrist1 = h1.landmark[0]
        hand1_data = normalize_hand(h1.landmark, wrist1)
        body_rel = get_body_relative_position(wrist1, pose_result.pose_landmarks)

        hand2_data = [0.0] * 63
        if num_hands > 1:
            h2 = hands_result.multi_hand_landmarks[1]
            hand2_data = normalize_hand(h2.landmark, h2.landmark[0])

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

    if pose_result.pose_landmarks:
        mp_draw.draw_landmarks(
            img, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(45,45,45), thickness=1, circle_radius=1),
            mp_draw.DrawingSpec(color=(35,35,35), thickness=1)
        )

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
    demo.launch(share=True)
