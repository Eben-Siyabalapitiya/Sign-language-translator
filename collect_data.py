import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

labels = [
    "you", "me", "hello", "goodbye", "thankyou", "please", "sorry",
    "yes", "no", "help", "more", "finish",
    "i love you", "friend", "together", "good", "bad"
]

csv_file = "gesture_data.csv"
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"h1_{i}" for i in range(63)] + \
                 [f"h2_{i}" for i in range(63)] + \
                 ["rel_nose_x", "rel_nose_y",
                  "rel_lshoulder_x", "rel_lshoulder_y",
                  "rel_rshoulder_x", "rel_rshoulder_y"] + ["label"]
        writer.writerow(header)

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

def draw_rounded_rect(img, x, y, w, h, r, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    cv2.circle(overlay, (x + r, y + r), r, color, -1)
    cv2.circle(overlay, (x + w - r, y + r), r, color, -1)
    cv2.circle(overlay, (x + r, y + h - r), r, color, -1)
    cv2.circle(overlay, (x + w - r, y + h - r), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_progress_bar(frame, x, y, w, h, progress, color_fill, color_bg):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bg, -1)
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color_fill, -1)
    if fill_w > h // 2:
        cv2.circle(frame, (x + fill_w, y + h // 2), h // 2, color_fill, -1)

def collect_data():
    cap = cv2.VideoCapture(1)
    current_label_idx = 0
    samples_per_label = 300
    current_samples = 0
    collecting = False
    
    WHITE  = (255, 255, 255)
    GREEN  = (80, 200, 120)
    DARK   = (15, 15, 15)
    GRAY   = (60, 60, 60)
    ACCENT = (255, 200, 0)

    print(f"\nReady to collect: {labels[current_label_idx]}")
    print("Press SPACE to start/stop collecting")
    print("Press N to move to next gesture")
    print("Press Q to stop\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_result  = pose.process(rgb)
        hands_result = hands.process(rgb)

        hand_detected = False
        pose_detected = pose_result.pose_landmarks is not None

        # Default both hands to zeros i think
        hand1_data = [0.0] * 63
        hand2_data = [0.0] * 63
        body_rel   = [0.0] * 6
        primary_wrist = None
            
        if hands_result.multi_hand_landmarks:
            hand_detected = True
            all_hands = hands_result.multi_hand_landmarks
            
            # First hand
            h1 = all_hands[0]
            mp_draw.draw_landmarks(frame, h1, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())
            wrist1 = h1.landmark[0]
            hand1_data = normalize_hand(h1.landmark, wrist1)
            primary_wrist = wrist1
#testing 
    

#testing

            # Second hand if present
            if len(all_hands) > 1:
                h2 = all_hands[1]
                mp_draw.draw_landmarks(frame, h2, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
                wrist2 = h2.landmark[0]
                hand2_data = normalize_hand(h2.landmark, wrist2)

            if primary_wrist:
                body_rel = get_body_relative_position(
                    primary_wrist, pose_result.pose_landmarks
                )

            if collecting and current_samples < samples_per_label:
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(hand1_data + hand2_data + body_rel + [labels[current_label_idx]])
                current_samples += 1
                
        # Draw pose skeleton lightly
        if pose_result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(60, 60, 60), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(50, 50, 50), thickness=1)
            )
                
        # ── TOP LEFT PANEL ──────────────────────────
        panel_w, panel_h = 310, 175
        draw_rounded_rect(frame, 10, 10, panel_w, panel_h, 12, DARK, alpha=0.65)

        cv2.putText(frame, "GESTURE", (26, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(frame, labels[current_label_idx].upper(), (26, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)

        cv2.putText(frame, f"{current_label_idx+1}/{len(labels)}", (panel_w - 55, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        cv2.line(frame, (26, 82), (panel_w - 10, 82), GRAY, 1)

        cv2.putText(frame, "SAMPLES", (26, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        progress  = current_samples / samples_per_label
        bar_color = GREEN if collecting else GRAY
        draw_progress_bar(frame, 26, 110, panel_w - 46, 10,
                          progress, bar_color, (40, 40, 40))
        cv2.putText(frame, f"{current_samples} / {samples_per_label}", (26, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        # Indicators
        body_color = GREEN if pose_detected else (80, 80, 220)
        hand2_color = GREEN if (hands_result.multi_hand_landmarks and
                                len(hands_result.multi_hand_landmarks) > 1) else GRAY
        cv2.putText(frame, "BODY OK" if pose_detected else "NO BODY",
                    (26, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.38, body_color, 1)
        cv2.putText(frame, "2 HANDS" if hand2_color == GREEN else "1 HAND",
                    (130, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.38, hand2_color, 1)

        # ── STATUS BADGE top right ───────────────────
        if collecting:
            status_text, status_color = "REC", (60, 60, 220)
        elif current_samples >= samples_per_label:
            status_text, status_color = "DONE", GREEN
        elif hand_detected:
            status_text, status_color = "HAND OK", ACCENT
        else:
            status_text, status_color = "NO HAND", (100, 100, 100)

        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        badge_x = fw - tw - 30
        draw_rounded_rect(frame, badge_x - 8, 10, tw + 20, 34, 8, DARK, alpha=0.7)
        cv2.putText(frame, status_text, (badge_x, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

        # ── DONE MESSAGE center ──────────────────────
        if current_samples >= samples_per_label:
            msg = "Done! Press N for next gesture"
            (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            mx = (fw - mw) // 2
            draw_rounded_rect(frame, mx - 12, fh//2 - 24, mw + 24, 40, 10, DARK, alpha=0.75)
            cv2.putText(frame, msg, (mx, fh//2 + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 2)

        # ── BOTTOM CONTROLS BAR ─────────────────────
        bar_y = fh - 44
        draw_rounded_rect(frame, 10, bar_y, fw - 20, 34, 8, DARK, alpha=0.65)
        hints = [("SPACE", "Collect"), ("N", "Next"), ("Q", "Quit")]
        spacing = (fw - 20) // len(hints)
        for i, (key, action) in enumerate(hints):
            bx = 26 + i * spacing
            cv2.putText(frame, key, (bx, bar_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT, 1)
            cv2.putText(frame, action, (bx + 52, bar_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            if i < len(hints) - 1:
                cv2.line(frame, (bx + spacing - 10, bar_y + 4),
                         (bx + spacing - 10, bar_y + 28), GRAY, 1)

        cv2.imshow("Sign Language — Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if current_samples < samples_per_label:
                collecting = not collecting
        elif key == ord('n'):
            if current_label_idx < len(labels) - 1:
                current_label_idx += 1
                current_samples = 0
                collecting = False
                print(f"Moved to: {labels[current_label_idx]}")
            else:
                print("All gestures collected!")

    cap.release()
    cv2.destroyAllWindows()

collect_data()