import cv2
import mediapipe as mp
import numpy as np

# TEMP CHANGE 3: Marker comment for repeated sequence.
# Initialize MediaPipe modules used by the pipeline.
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils

# Create the hand and pose detector objects.
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
pose  = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def draw_rounded_rect(img, x, y, w, h, r, color, alpha=0.7):
    overlay = img.copy()
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    cv2.circle(overlay, (x + r,     y + r),     r, color, -1)
    cv2.circle(overlay, (x + w - r, y + r),     r, color, -1)
    cv2.circle(overlay, (x + r,     y + h - r), r, color, -1)
    cv2.circle(overlay, (x + w - r, y + h - r), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_normal(frame, pose_result, hands_result):
    """Render the normal live camera view with pose and hand overlays."""
    h, w = frame.shape[:2]

    # Draw faint pose
    if pose_result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(60, 60, 60), thickness=1, circle_radius=2),
            mp_draw.DrawingSpec(color=(45, 45, 45), thickness=1)
        )

    # Draw hands if any are detected.
    if hands_result.multi_hand_landmarks:
        for hand_lm in hands_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

    # Top label for the live feed.
    draw_rounded_rect(frame, 10, 10, 200, 36, 8, (12, 12, 12), alpha=0.75)
    cv2.putText(frame, "LIVE FEED (change 4)", (22, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Bottom hint
    cv2.putText(frame, "Q  quit", (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)

    return frame


def draw_mask(frame_shape, pose_result, hands_result):
    """Render the pipeline mask view with stylized pose and hand visuals."""
    h, w = frame_shape[:2]

    # Pure black canvas for the overlay mask.
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Scanline effect — very subtle horizontal lines
    for y in range(0, h, 4):
        cv2.line(mask, (0, y), (w, y), (8, 8, 8), 1)

    # ── POSE — glowing cyan skeleton ─────────────────
    if pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks.landmark

        # Draw connections manually with glow effect
        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start = lms[start_idx]
            end   = lms[end_idx]
            if start.visibility > 0.5 and end.visibility > 0.5:
                sx, sy = int(start.x * w), int(start.y * h)
                ex, ey = int(end.x * w),   int(end.y * h)
                # Outer glow
                cv2.line(mask, (sx, sy), (ex, ey), (40, 120, 100), 4)
                # Inner bright line
                cv2.line(mask, (sx, sy), (ex, ey), (80, 220, 180), 1)

        # Draw joints
        for lm in lms:
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(mask, (cx, cy), 5, (40, 180, 140), -1)
                cv2.circle(mask, (cx, cy), 2, (180, 255, 230), -1)

    # ── HANDS — glowing electric blue ────────────────
    if hands_result.multi_hand_landmarks:
        for hand_lm in hands_result.multi_hand_landmarks:
            lms = hand_lm.landmark

            # Connections
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = lms[start_idx]
                end   = lms[end_idx]
                sx, sy = int(start.x * w), int(start.y * h)
                ex, ey = int(end.x * w),   int(end.y * h)
                # Outer glow
                cv2.line(mask, (sx, sy), (ex, ey), (120, 60, 20), 5)
                # Inner bright
                cv2.line(mask, (sx, sy), (ex, ey), (255, 140, 60), 2)

            # Fingertip highlights
            fingertips = [4, 8, 12, 16, 20]
            for i, lm in enumerate(lms):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if i in fingertips:
                    cv2.circle(mask, (cx, cy), 8,  (60, 40, 200), -1)
                    cv2.circle(mask, (cx, cy), 4,  (180, 140, 255), -1)
                    cv2.circle(mask, (cx, cy), 2,  (255, 255, 255), -1)
                else:
                    cv2.circle(mask, (cx, cy), 4,  (80, 50, 160), -1)
                    cv2.circle(mask, (cx, cy), 2,  (200, 160, 255), -1)

    # ── WRIST POSITION INDICATOR — reference line from hand to nose ───
    if hands_result.multi_hand_landmarks and pose_result.pose_landmarks:
        pose_lms = pose_result.pose_landmarks.landmark
        nose = pose_lms[mp_pose.PoseLandmark.NOSE]
        nx, ny = int(nose.x * w), int(nose.y * h)

        for hand_lm in hands_result.multi_hand_landmarks:
            wrist = hand_lm.landmark[0]
            wx, wy = int(wrist.x * w), int(wrist.y * h)

            # Dotted line from wrist to nose (body reference line)
            num_dots = 12
            for i in range(num_dots):
                t = i / num_dots
                dx = int(wx + (nx - wx) * t)
                dy = int(wy + (ny - wy) * t)
                cv2.circle(mask, (dx, dy), 2, (50, 50, 80), -1)

    # ── CORNER BRACKETS — stylized border decorations ───────────────
    bracket_color = (60, 60, 80)
    blen = 30
    bt   = 2
    # Top left
    cv2.line(mask, (10, 10), (10 + blen, 10), bracket_color, bt)
    cv2.line(mask, (10, 10), (10, 10 + blen), bracket_color, bt)
    # Top right
    cv2.line(mask, (w - 10, 10), (w - 10 - blen, 10), bracket_color, bt)
    cv2.line(mask, (w - 10, 10), (w - 10, 10 + blen), bracket_color, bt)
    # Bottom left
    cv2.line(mask, (10, h - 10), (10 + blen, h - 10), bracket_color, bt)
    cv2.line(mask, (10, h - 10), (10, h - 10 - blen), bracket_color, bt)
    # Bottom right
    cv2.line(mask, (w - 10, h - 10), (w - 10 - blen, h - 10), bracket_color, bt)
    cv2.line(mask, (w - 10, h - 10), (w - 10, h - 10 - blen), bracket_color, bt)

    # ── STATS — display number of detected hands and pose status ─────
    num_hands = len(hands_result.multi_hand_landmarks) if hands_result.multi_hand_landmarks else 0
    body_ok   = pose_result.pose_landmarks is not None

    draw_rounded_rect(mask, 10, 10, 220, 80, 8, (10, 10, 10), alpha=0.85)
    cv2.putText(mask, "PIPELINE VIEW", (22, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 160), 1)
    cv2.putText(mask, f"HANDS  {num_hands}", (22, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 140, 60) if num_hands > 0 else (60, 60, 60), 1)
    cv2.putText(mask, f"BODY   {'OK' if body_ok else '--'}", (22, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (80, 220, 180) if body_ok else (60, 60, 60), 1)

    # Legend bottom right
    draw_rounded_rect(mask, w - 160, h - 80, 148, 68, 8, (10, 10, 10), alpha=0.85)
    cv2.circle(mask, (w - 140, h - 60), 5, (80, 220, 180), -1)
    cv2.putText(mask, "BODY", (w - 128, h - 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 220, 180), 1)
    cv2.circle(mask, (w - 140, h - 38), 5, (255, 140, 60), -1)
    cv2.putText(mask, "HANDS", (w - 128, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 140, 60), 1)
    cv2.circle(mask, (w - 140, h - 18), 4, (180, 140, 255), -1)
    cv2.putText(mask, "FINGERTIPS", (w - 128, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 140, 255), 1)

    return mask

# ── MAIN LOOP — capture webcam frames and show both views ───────────
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame and convert to RGB for MediaPipe processing.
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect body pose and hands in the current frame.
    pose_result  = pose.process(rgb)
    hands_result = hands.process(rgb)

    # Create both display views from the detection results.
    normal_view = draw_normal(frame.copy(), pose_result, hands_result)
    mask_view   = draw_mask(frame.shape, pose_result, hands_result)

    # Show the live camera feed and the stylized pipeline mask.
    cv2.imshow("Live Feed", normal_view)
    cv2.imshow("Pipeline Vision", mask_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()