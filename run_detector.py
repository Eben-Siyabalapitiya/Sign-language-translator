import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque, Counter
import pyttsx3
import threading

model = tf.keras.models.load_model("gesture_model.keras")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
pose  = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

prediction_buffer = deque(maxlen=10)

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

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
    if fill_w > h//2:
        cv2.circle(frame, (x+fill_w, y+h//2), h//2, color_fill, -1)

# Colors
WHITE  = (255, 255, 255)
GREEN  = (80, 200, 120)
DARK   = (12, 12, 12)
GRAY   = (55, 55, 55)
LGRAY  = (140, 140, 140)
ACCENT = (255, 200, 0)
CYAN   = (80, 220, 180)
ORANGE = (60, 140, 255)
PURPLE = (255, 140, 180)

# Layout
CAM_W, CAM_H = 640, 360
CALC_H        = 280
TOTAL_W       = CAM_W * 2
TOTAL_H       = CAM_H + CALC_H

cap = cv2.VideoCapture(1)

last_spoken     = ""
display_word    = ""
confidence_val  = 0.0
word_history    = []
speak_cooldown  = 0
predictions_raw = None
predicted_class = ""
hand1_norm      = [0.0] * 63
body_rel        = [0.0] * 6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAM_W, CAM_H))
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_result  = pose.process(rgb)
    hands_result = hands.process(rgb)

    hand1_data = [0.0] * 63
    hand2_data = [0.0] * 63
    body_rel   = [0.0] * 6
    num_hands  = 0

    if speak_cooldown > 0:
        speak_cooldown -= 1

    if hands_result.multi_hand_landmarks:
        num_hands  = len(hands_result.multi_hand_landmarks)
        h1         = hands_result.multi_hand_landmarks[0]
        wrist1     = h1.landmark[0]
        hand1_data = normalize_hand(h1.landmark, wrist1)
        hand1_norm = hand1_data
        body_rel   = get_body_relative_position(wrist1, pose_result.pose_landmarks)

        if num_hands > 1:
            h2         = hands_result.multi_hand_landmarks[1]
            hand2_data = normalize_hand(h2.landmark, h2.landmark[0])

        input_data      = np.array(hand1_data + hand2_data + body_rel).reshape(1, -1)
        predictions_raw = model.predict(input_data, verbose=0)
        confidence_val  = float(np.max(predictions_raw))
        predicted_class = encoder.inverse_transform([np.argmax(predictions_raw)])[0]
        prediction_buffer.append(predicted_class)

        if confidence_val > 0.85:
            smoothed     = Counter(prediction_buffer).most_common(1)[0][0]
            display_word = smoothed.upper()
            if smoothed != last_spoken and speak_cooldown == 0:
                last_spoken    = smoothed
                speak_cooldown = 60
                threading.Thread(target=speak, args=(smoothed,), daemon=True).start()
            if len(word_history) == 0 or word_history[-1] != display_word:
                word_history.append(display_word)
                if len(word_history) > 5:
                    word_history.pop(0)
        else:
            display_word = ""

    # ── PANEL 1 — LIVE FEED ──────────────────────────
    live = frame.copy()

    if pose_result.pose_landmarks:
        mp_draw.draw_landmarks(live, pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(45,45,45), thickness=1, circle_radius=1),
            mp_draw.DrawingSpec(color=(35,35,35), thickness=1))

    if hands_result.multi_hand_landmarks:
        for h_lm in hands_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(live, h_lm, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

    # Prediction card — bottom center
    card_w, card_h = 380, 80
    card_x = (CAM_W - card_w) // 2
    card_y = CAM_H - card_h - 14
    draw_rounded_rect(live, card_x, card_y, card_w, card_h, 12, DARK, alpha=0.85)

    if display_word and confidence_val > 0.85:
        (tw, _), _ = cv2.getTextSize(display_word, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
        tx = card_x + (card_w - tw) // 2
        cv2.putText(live, display_word, (tx, card_y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, GREEN, 3)
        draw_progress_bar(live, card_x+20, card_y+card_h-12,
                          card_w-40, 5, confidence_val, GREEN, (40,40,40))
    else:
        (tw, _), _ = cv2.getTextSize("Detecting...", cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        tx = card_x + (card_w - tw) // 2
        cv2.putText(live, "Detecting...", (tx, card_y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, LGRAY, 2)

    # History panel top left
    if word_history:
        hh = 28 + len(word_history) * 22
        draw_rounded_rect(live, 8, 8, 185, hh, 10, DARK, alpha=0.72)
        cv2.putText(live, "HISTORY", (18, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, LGRAY, 1)
        for i, word in enumerate(reversed(word_history)):
            cv2.putText(live, word, (18, 40 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 if i == 0 else 0.42,
                        WHITE if i == 0 else LGRAY,
                        2 if i == 0 else 1)

    # Badges top right
    conf_str = f"{confidence_val*100:.0f}% CONF"
    cc = GREEN if confidence_val > 0.85 else ACCENT if confidence_val > 0.5 else LGRAY
    (cw2,_),_ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    draw_rounded_rect(live, CAM_W-cw2-24, 8, cw2+16, 24, 5, DARK, alpha=0.72)
    cv2.putText(live, conf_str, (CAM_W-cw2-16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, cc, 1)

    hs = f"{num_hands} HAND{'S' if num_hands!=1 else ''}"
    (hw,_),_ = cv2.getTextSize(hs, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    draw_rounded_rect(live, CAM_W-hw-24, 38, hw+16, 24, 5, DARK, alpha=0.72)
    cv2.putText(live, hs, (CAM_W-hw-16, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN if num_hands>0 else LGRAY, 1)

    # Live feed label bottom left
    draw_rounded_rect(live, 8, CAM_H-26, 120, 20, 4, DARK, alpha=0.72)
    cv2.putText(live, "LIVE FEED", (14, CAM_H-11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, LGRAY, 1)

    # ── PANEL 2 — PIPELINE VISION ────────────────────
    pipe = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    for y in range(0, CAM_H, 4):
        cv2.line(pipe, (0,y), (CAM_W,y), (8,8,8), 1)

    if pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks.landmark
        for conn in mp_pose.POSE_CONNECTIONS:
            s, e = conn
            sl, el = lms[s], lms[e]
            if sl.visibility > 0.5 and el.visibility > 0.5:
                sx,sy = int(sl.x*CAM_W), int(sl.y*CAM_H)
                ex,ey = int(el.x*CAM_W), int(el.y*CAM_H)
                cv2.line(pipe, (sx,sy), (ex,ey), (40,120,100), 4)
                cv2.line(pipe, (sx,sy), (ex,ey), (80,220,180), 1)
        for lm in lms:
            if lm.visibility > 0.5:
                cx2,cy2 = int(lm.x*CAM_W), int(lm.y*CAM_H)
                cv2.circle(pipe, (cx2,cy2), 5, (40,180,140), -1)
                cv2.circle(pipe, (cx2,cy2), 2, (180,255,230), -1)

    if hands_result.multi_hand_landmarks:
        for h_lm in hands_result.multi_hand_landmarks:
            lms = h_lm.landmark
            for conn in mp_hands.HAND_CONNECTIONS:
                s,e = conn
                sx,sy = int(lms[s].x*CAM_W), int(lms[s].y*CAM_H)
                ex,ey = int(lms[e].x*CAM_W), int(lms[e].y*CAM_H)
                cv2.line(pipe, (sx,sy), (ex,ey), (120,60,20), 5)
                cv2.line(pipe, (sx,sy), (ex,ey), (255,140,60), 2)
            fingertips = [4,8,12,16,20]
            for i,lm in enumerate(lms):
                cx2,cy2 = int(lm.x*CAM_W), int(lm.y*CAM_H)
                if i in fingertips:
                    cv2.circle(pipe, (cx2,cy2), 8, (60,40,200),  -1)
                    cv2.circle(pipe, (cx2,cy2), 4, (180,140,255),-1)
                    cv2.circle(pipe, (cx2,cy2), 2, (255,255,255),-1)
                else:
                    cv2.circle(pipe, (cx2,cy2), 4, (80,50,160),  -1)
                    cv2.circle(pipe, (cx2,cy2), 2, (200,160,255),-1)

    if hands_result.multi_hand_landmarks and pose_result.pose_landmarks:
        nose = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nx,ny = int(nose.x*CAM_W), int(nose.y*CAM_H)
        for h_lm in hands_result.multi_hand_landmarks:
            wr = h_lm.landmark[0]
            wx,wy = int(wr.x*CAM_W), int(wr.y*CAM_H)
            for i in range(14):
                t  = i/14
                dx = int(wx+(nx-wx)*t)
                dy = int(wy+(ny-wy)*t)
                cv2.circle(pipe, (dx,dy), 2, (50,50,80), -1)

    # Corner brackets
    bc,bl,bt = (60,60,80), 24, 1
    cv2.line(pipe,(8,8),(8+bl,8),bc,bt)
    cv2.line(pipe,(8,8),(8,8+bl),bc,bt)
    cv2.line(pipe,(CAM_W-8,8),(CAM_W-8-bl,8),bc,bt)
    cv2.line(pipe,(CAM_W-8,8),(CAM_W-8,8+bl),bc,bt)
    cv2.line(pipe,(8,CAM_H-8),(8+bl,CAM_H-8),bc,bt)
    cv2.line(pipe,(8,CAM_H-8),(8,CAM_H-8-bl),bc,bt)
    cv2.line(pipe,(CAM_W-8,CAM_H-8),(CAM_W-8-bl,CAM_H-8),bc,bt)
    cv2.line(pipe,(CAM_W-8,CAM_H-8),(CAM_W-8,CAM_H-8-bl),bc,bt)

    # Stats panel
    body_ok = pose_result.pose_landmarks is not None
    draw_rounded_rect(pipe, 8, 8, 200, 72, 8, (10,10,10), alpha=0.85)
    cv2.putText(pipe, "PIPELINE VIEW", (18,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, CYAN, 1)
    cv2.putText(pipe, f"HANDS  {num_hands}", (18,48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ORANGE if num_hands>0 else GRAY, 1)
    cv2.putText(pipe, f"BODY   {'OK' if body_ok else '--'}", (18,66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN if body_ok else GRAY, 1)

    # Legend bottom right
    draw_rounded_rect(pipe, CAM_W-148, CAM_H-70, 138, 62, 8, (10,10,10), alpha=0.85)
    cv2.circle(pipe,(CAM_W-128,CAM_H-52),4,CYAN,-1)
    cv2.putText(pipe,"BODY",(CAM_W-118,CAM_H-48),cv2.FONT_HERSHEY_SIMPLEX,0.32,CYAN,1)
    cv2.circle(pipe,(CAM_W-128,CAM_H-36),4,ORANGE,-1)
    cv2.putText(pipe,"HANDS",(CAM_W-118,CAM_H-32),cv2.FONT_HERSHEY_SIMPLEX,0.32,ORANGE,1)
    cv2.circle(pipe,(CAM_W-128,CAM_H-20),4,PURPLE,-1)
    cv2.putText(pipe,"FINGERTIPS",(CAM_W-118,CAM_H-16),cv2.FONT_HERSHEY_SIMPLEX,0.32,PURPLE,1)

    # ── PANEL 3 — CALCULATIONS BOTTOM FULL WIDTH ─────
    calc      = np.zeros((CALC_H, TOTAL_W, 3), dtype=np.uint8)
    section_w = TOTAL_W // 4

    for y in range(0, CALC_H, 4):
        cv2.line(calc, (0,y), (TOTAL_W,y), (6,6,6), 1)

    # Section dividers
    for i in range(1, 4):
        cv2.line(calc, (section_w*i, 6), (section_w*i, CALC_H-6), (30,30,30), 1)

    # Section header line
    cv2.line(calc, (0,32), (TOTAL_W,32), (25,25,25), 1)

    # ── S1 HAND SKELETON ─────────────────────────────
    s1x = 10
    cv2.putText(calc, "HAND SKELETON", (s1x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ORANGE, 1)

    if hands_result.multi_hand_landmarks:
        h_lm = hands_result.multi_hand_landmarks[0]
        lms  = h_lm.landmark
        xs   = [lm.x for lm in lms]
        ys   = [lm.y for lm in lms]
        mn_x, mx_x = min(xs), max(xs)
        mn_y, mx_y = min(ys), max(ys)
        pad  = 0.15
        rx2  = max(mx_x - mn_x, 0.01)
        ry2  = max(mx_y - mn_y, 0.01)
        zw   = section_w - 28
        zh   = CALC_H - 48
        hand_pts = []
        for lm in lms:
            px = s1x + int(((lm.x-mn_x+pad*rx2)/(rx2*(1+2*pad)))*zw)
            py = 38  + int(((lm.y-mn_y+pad*ry2)/(ry2*(1+2*pad)))*zh)
            hand_pts.append((px, py))
        for conn in mp_hands.HAND_CONNECTIONS:
            s,e = conn
            cv2.line(calc, hand_pts[s], hand_pts[e], (120,60,20), 3)
            cv2.line(calc, hand_pts[s], hand_pts[e], (255,140,60), 1)
        fingertips = [4,8,12,16,20]
        for i,pt in enumerate(hand_pts):
            if i in fingertips:
                cv2.circle(calc, pt, 5,  (60,40,200),  -1)
                cv2.circle(calc, pt, 3,  (180,140,255),-1)
                cv2.circle(calc, pt, 1,  (255,255,255),-1)
            else:
                cv2.circle(calc, pt, 3,  (80,50,160),  -1)
                cv2.circle(calc, pt, 1,  (200,160,255),-1)
            cv2.putText(calc, str(i), (pt[0]+4, pt[1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (80,80,100), 1)

    # ── S2 NORM LANDMARKS ────────────────────────────
    s2x = section_w + 10
    cv2.putText(calc, "NORM LANDMARKS", (s2x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)
    row_h = (CALC_H - 40) // 13
    if hand1_norm:
        for i in range(0, min(14, len(hand1_norm)), 2):
            vx = hand1_norm[i]
            vy = hand1_norm[i+1] if i+1 < len(hand1_norm) else 0
            bv = min(abs(vx), 1.0)
            bc2 = CYAN if vx >= 0 else (80,80,220)
            yy  = 38 + (i//2) * row_h
            cv2.rectangle(calc, (s2x, yy), (s2x+int(55*bv), yy+5), bc2, -1)
            cv2.putText(calc, f"L{i//2:02d} {vx:+.2f} {vy:+.2f}",
                        (s2x+62, yy+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, (160,160,160), 1)

    # Body position
    yoff = 38 + 7 * row_h + 6
    cv2.line(calc,(s2x,yoff),(s2x+section_w-20,yoff),(30,30,30),1)
    yoff += 10
    cv2.putText(calc,"BODY POSITION",(s2x,yoff),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,LGRAY,1)
    yoff += 14
    blabels = ["nose_x","nose_y","lsh_x","lsh_y","rsh_x","rsh_y"]
    for lb,val in zip(blabels, body_rel):
        if yoff > CALC_H - 10:
            break
        bv  = min(abs(val)/3.0, 1.0)
        bc2 = CYAN if val>=0 else (80,80,220)
        cv2.rectangle(calc,(s2x,yoff),(s2x+int(45*bv),yoff+5),bc2,-1)
        cv2.putText(calc,f"{lb} {val:+.2f}",(s2x+52,yoff+6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.25,(160,160,160),1)
        yoff += 12

    # ── S3 CLASS PROBABILITIES ───────────────────────
    s3x = section_w*2 + 10
    cv2.putText(calc, "CLASS PROBABILITIES", (s3x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, LGRAY, 1)

    if predictions_raw is not None:
        probs      = predictions_raw[0]
        sorted_idx = np.argsort(probs)[::-1]
        bar_max    = section_w - 90
        for rank, idx in enumerate(sorted_idx):
            label  = encoder.classes_[idx]
            prob   = probs[idx]
            is_top = (label == predicted_class)
            yy     = 38 + rank * 15
            if yy > CALC_H - 14:
                break
            blen   = int(bar_max * prob)
            cv2.rectangle(calc,(s3x,yy),(s3x+blen,yy+9),
                          GREEN if is_top else (35,35,55),-1)
            cv2.putText(calc,f"{label[:11]:<11} {prob*100:4.1f}%",
                        (s3x+bar_max+6,yy+9),
                        cv2.FONT_HERSHEY_SIMPLEX,0.27,
                        GREEN if is_top else (90,90,90),1)

    # ── S4 PREDICTION ────────────────────────────────
    s4x       = section_w*3 + 10
    s4w       = section_w - 20
    pred_color = GREEN if confidence_val > 0.85 else ACCENT

    cv2.putText(calc, "PREDICTION", (s4x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, LGRAY, 1)

    word_show = predicted_class.upper() if predicted_class else "---"
    (tw,_),_  = cv2.getTextSize(word_show, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    tx        = s4x + (s4w - tw) // 2
    cv2.putText(calc, word_show, (tx, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 2)

    # Confidence bar
    draw_progress_bar(calc, s4x+8, 84, s4w-16, 7,
                      confidence_val, pred_color, (30,30,30))
    cv2.putText(calc, f"{confidence_val*100:.1f}% confidence",
                (s4x+8, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, pred_color, 1)

    # Buffer votes
    cv2.line(calc,(s4x,114),(s4x+s4w,114),(30,30,30),1)
    cv2.putText(calc, "BUFFER VOTES", (s4x+8, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, LGRAY, 1)

    if len(prediction_buffer) > 0:
        vote_counts = Counter(prediction_buffer)
        vy = 142
        for word_v, count in vote_counts.most_common(4):
            if vy > CALC_H - 10:
                break
            is_w    = (word_v == predicted_class)
            bar_len = int((s4w - 20) * count / 10)
            cv2.rectangle(calc,(s4x+8,vy-8),(s4x+8+bar_len,vy),
                          GREEN if is_w else (45,45,45),-1)
            cv2.putText(calc,f"{word_v[:10]}  {count}/10",
                        (s4x+8,vy+12),
                        cv2.FONT_HERSHEY_SIMPLEX,0.26,
                        GREEN if is_w else (75,75,75),1)
            vy += 26

    # Corner brackets on calc panel
    bc,bl,bt = (40,40,60), 20, 1
    cv2.line(calc,(4,4),(4+bl,4),bc,bt)
    cv2.line(calc,(4,4),(4,4+bl),bc,bt)
    cv2.line(calc,(TOTAL_W-4,4),(TOTAL_W-4-bl,4),bc,bt)
    cv2.line(calc,(TOTAL_W-4,4),(TOTAL_W-4,4+bl),bc,bt)
    cv2.line(calc,(4,CALC_H-4),(4+bl,CALC_H-4),bc,bt)
    cv2.line(calc,(4,CALC_H-4),(4,CALC_H-4-bl),bc,bt)
    cv2.line(calc,(TOTAL_W-4,CALC_H-4),(TOTAL_W-4-bl,CALC_H-4),bc,bt)
    cv2.line(calc,(TOTAL_W-4,CALC_H-4),(TOTAL_W-4,CALC_H-4-bl),bc,bt)

    # ── COMBINE ──────────────────────────────────────
    top_row = np.hstack([live, pipe])
    canvas  = np.vstack([top_row, calc])

    cv2.imshow("Sign Language Translator", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()