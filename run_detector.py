import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque, Counter

model = tf.keras.models.load_model("gesture_model.keras")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
pose  = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
    lms = pose_landmarks.landmark
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
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    cv2.circle(overlay, (x + r,     y + r),     r, color, -1)
    cv2.circle(overlay, (x + w - r, y + r),     r, color, -1)
    cv2.circle(overlay, (x + r,     y + h - r), r, color, -1)
    cv2.circle(overlay, (x + w - r, y + h - r), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_progress_bar(frame, x, y, w, h, progress, color_fill, color_bg):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bg, -1)
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color_fill, -1)
    if fill_w > h // 2:
        cv2.circle(frame, (x + fill_w, y + h // 2), h // 2, color_fill, -1)

# ── COLORS ───────────────────────────────────────────
WHITE   = (255, 255, 255)
GREEN   = (80, 200, 120)
DARK    = (12, 12, 12)
GRAY    = (55, 55, 55)
LGRAY   = (140, 140, 140)
ACCENT  = (255, 200, 0)
BLUE    = (220, 120, 60)
RED     = (80, 80, 220)

cap = cv2.VideoCapture(1)

last_prediction  = ""
display_word     = ""
confidence_val   = 0.0
word_history     = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_result  = pose.process(rgb)
    hands_result = hands.process(rgb)

    hand1_data    = [0.0] * 63
    hand2_data    = [0.0] * 63
    body_rel      = [0.0] * 6
    hand_detected = False
    num_hands     = 0

    if hands_result.multi_hand_landmarks:
        hand_detected = True
        all_hands = hands_result.multi_hand_landmarks
        num_hands = len(all_hands)

        h1 = all_hands[0]
        mp_draw.draw_landmarks(frame, h1, mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())
        wrist1     = h1.landmark[0]
        hand1_data = normalize_hand(h1.landmark, wrist1)
        body_rel   = get_body_relative_position(wrist1, pose_result.pose_landmarks)

        if num_hands > 1:
            h2 = all_hands[1]
            mp_draw.draw_landmarks(frame, h2, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())
            wrist2     = h2.landmark[0]
            hand2_data = normalize_hand(h2.landmark, wrist2)

        input_data   = np.array(hand1_data + hand2_data + body_rel).reshape(1, -1)
        predictions  = model.predict(input_data, verbose=0)
        confidence_val = float(np.max(predictions))
        predicted_class = encoder.inverse_transform([np.argmax(predictions)])[0]
        prediction_buffer.append(predicted_class)

        if confidence_val > 0.85:
            smoothed = Counter(prediction_buffer).most_common(1)[0][0]
            if smoothed != last_prediction:
                last_prediction = smoothed
                display_word    = smoothed.upper()
                if len(word_history) == 0 or word_history[-1] != display_word:
                    word_history.append(display_word)
                    if len(word_history) > 5:
                        word_history.pop(0)

    # Draw pose skeleton very faintly
    if pose_result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(45, 45, 45), thickness=1, circle_radius=1),
            mp_draw.DrawingSpec(color=(35, 35, 35), thickness=1)
        )

    # ── MAIN PREDICTION CARD — bottom center ─────────
    card_w, card_h = 420, 110
    card_x = (fw - card_w) // 2
    card_y = fh - card_h - 20

    draw_rounded_rect(frame, card_x, card_y, card_w, card_h, 16, DARK, alpha=0.82)

    if display_word and confidence_val > 0.85:
        # Big word
        font_scale = 2.2
        thickness  = 3
        (tw, th), _ = cv2.getTextSize(display_word, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        tx = card_x + (card_w - tw) // 2
        ty = card_y + 68
        cv2.putText(frame, display_word, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, GREEN, thickness)

        # Confidence bar under word
        bar_x = card_x + 30
        bar_y2 = card_y + card_h - 18
        bar_w2 = card_w - 60
        draw_progress_bar(frame, bar_x, bar_y2, bar_w2, 6,
                          confidence_val, GREEN, (40, 40, 40))
    else:
        # Detecting state
        dot_text = "Detecting..."
        (tw, _), _ = cv2.getTextSize(dot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        tx = card_x + (card_w - tw) // 2
        cv2.putText(frame, dot_text, (tx, card_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, LGRAY, 2)

    # ── WORD HISTORY — top left ───────────────────────
    if word_history:
        hist_w, hist_h = 220, 30 + len(word_history) * 28
        draw_rounded_rect(frame, 10, 10, hist_w, hist_h, 12, DARK, alpha=0.7)
        cv2.putText(frame, "HISTORY", (24, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, LGRAY, 1)
        for i, word in enumerate(reversed(word_history)):
            alpha_color = WHITE if i == 0 else LGRAY
            scale       = 0.65 if i == 0 else 0.52
            cv2.putText(frame, word, (24, 52 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, alpha_color, 1 if i > 0 else 2)

    # ── STATUS BADGES — top right ─────────────────────
    # Confidence %
    conf_str = f"{confidence_val*100:.0f}% CONF"
    conf_color = GREEN if confidence_val > 0.85 else ACCENT if confidence_val > 0.5 else LGRAY
    (cw, _), _ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    draw_rounded_rect(frame, fw - cw - 30, 10, cw + 20, 28, 6, DARK, alpha=0.7)
    cv2.putText(frame, conf_str, (fw - cw - 20, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1)

    # Hand count badge
    hand_str   = f"{num_hands} HAND{'S' if num_hands != 1 else ''}"
    hand_color = GREEN if num_hands > 0 else LGRAY
    (hw2, _), _ = cv2.getTextSize(hand_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    draw_rounded_rect(frame, fw - hw2 - 30, 46, hw2 + 20, 28, 6, DARK, alpha=0.7)
    cv2.putText(frame, hand_str, (fw - hw2 - 20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, hand_color, 1)

    # ── QUIT HINT — bottom left ───────────────────────
    cv2.putText(frame, "Q  quit", (18, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    cv2.imshow("Sign Language Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()