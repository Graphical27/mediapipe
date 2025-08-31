import cv2
import mediapipe as mp
import autopy
import numpy as np
import time
import math
from HandTrackingModule import handDetector

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
screen_w, screen_h = autopy.screen.size()
cap.set(3, wCam)
cap.set(4, hCam)

# Hand tracking setup
detector = handDetector(detectionCon=0.7, maxHands=1)
frameR = 100
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Eye control smoothing & state
prev_x, prev_y = 0.0, 0.0
smooth_alpha = 0.20

# Blink state
blink_time = 0.0
blink_cooldown = 0.9
EAR_THRESHOLD = 0.20

# Tilt (left-click) state
tilt_click_time = 0.0
tilt_cooldown = 0.8
TILT_ANGLE_THRESHOLD = -12.0

def get_eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def get_face_tilt(landmarks):
    left = np.array([landmarks[234].x, landmarks[234].y])
    right = np.array([landmarks[454].x, landmarks[454].y])
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

# ------------------- Calibration / small ROI settings -------------------
# Estimate pixels-per-cm using a common screen DPI assumption (96 DPI -> 37.8 px/cm).
# This is only an initial estimate; you can re-center (press 'c') to align ROI to your finger.
pixels_per_cm = 96.0 / 2.54  # ≈ 37.8 px per cm
calib_size_cm = 3.0
calib_size_px = int(calib_size_cm * pixels_per_cm)  # default ROI side in px (~114 px)
roi_center_x, roi_center_y = wCam // 2, hCam // 2  # default center of ROI

# allow min/max ROI sizes
MIN_ROI_PX = 30
MAX_ROI_PX = min(wCam, hCam) - 10

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    # Hand tracking
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)

    hand_control = False
    finger_x, finger_y = None, None
    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]  # Index finger tip
        x2, y2 = lmList[12][1], lmList[12][2]  # Middle finger tip
        fingers = detector.fingersUp()
        # Draw the ROI (small square) that maps to full screen
        half = calib_size_px // 2
        roi_left = max(0, roi_center_x - half)
        roi_top = max(0, roi_center_y - half)
        roi_right = min(wCam - 1, roi_center_x + half)
        roi_bottom = min(hCam - 1, roi_center_y + half)
        cv2.rectangle(img, (roi_left, roi_top), (roi_right, roi_bottom), (255, 0, 255), 2)

        # If index finger up and middle finger down -> move mode (drag)
        if fingers[1] == 1 and fingers[2] == 0:
            finger_x, finger_y = x1, y1
            # Map finger within ROI to full screen range
            # If finger is outside ROI, we clamp to ROI edges (so small physical movement still maps to edge)
            mapped_x = np.interp(finger_x, (roi_left, roi_right), (0, screen_w))
            mapped_y = np.interp(finger_y, (roi_top, roi_bottom), (0, screen_h))

            # smoothing
            clocX = plocX + (mapped_x - plocX) / smoothening
            clocY = plocY + (mapped_y - plocY) / smoothening
            # Move mouse, invert X if needed for natural mapping (camera flip)
            try:
                autopy.mouse.move(int(screen_w - clocX), int(clocY))
            except Exception:
                pass
            plocX, plocY = clocX, clocY
            hand_control = True
            cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)

        # Hand click (optional)
        if fingers[1] == 1 and fingers[2] == 0:
            length, img, linedistance = detector.findDistance(8, 12, img, draw=True)
            if length < 40:
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                # autopy.mouse.click()  # keep commented if you want clicks only by eye/face

    # Eye/face control (for click only, or for mouse if no hand detected)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [263, 387, 385, 362, 380, 373]

        left_ear = get_eye_aspect_ratio(landmarks, left_eye_indices)
        right_ear = get_eye_aspect_ratio(landmarks, right_eye_indices)
        ear = (left_ear + right_ear) / 2.0

        now = time.time()
        if ear < EAR_THRESHOLD and (now - blink_time) > blink_cooldown:
            try:
                autopy.mouse.click(autopy.mouse.Button.RIGHT)
            except Exception:
                pass
            blink_time = now

        angle = get_face_tilt(landmarks)
        if angle < TILT_ANGLE_THRESHOLD and (now - tilt_click_time) > tilt_cooldown:
            try:
                autopy.mouse.click(autopy.mouse.Button.LEFT)
            except Exception:
                pass
            tilt_click_time = now

        # Only move mouse with eye if hand is NOT detected
        if not hand_control:
            iris_available = len(landmarks) > 468
            if iris_available and len(landmarks) >= 478:
                iris_points = [landmarks[i] for i in range(468, 478)]
                iris_x = np.mean([p.x for p in iris_points]) * img_w
                iris_y = np.mean([p.y for p in iris_points]) * img_h
                raw_x, raw_y = iris_x, iris_y
            else:
                eye_x = np.mean([landmarks[i].x for i in left_eye_indices]) * img_w
                eye_y = np.mean([landmarks[i].y for i in left_eye_indices]) * img_h
                raw_x, raw_y = eye_x, eye_y

            screen_x = np.interp(raw_x, [0, img_w], [0, screen_w])
            screen_y = np.interp(raw_y, [0, img_h], [0, screen_h])

            prev_x = prev_x if isinstance(prev_x, float) else 0.0
            prev_y = prev_y if isinstance(prev_y, float) else 0.0
            smooth_x = prev_x + smooth_alpha * (screen_x - prev_x)
            smooth_y = prev_y + smooth_alpha * (screen_y - prev_y)
            prev_x, prev_y = smooth_x, smooth_y

            try:
                autopy.mouse.move(int(smooth_x), int(smooth_y))
            except Exception:
                pass

        mp_drawing.draw_landmarks(img, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)
        cv2.putText(img, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(img, f"Angle: {angle:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # HUD: show ROI size and instructions
    cv2.putText(img, f"ROI: {calib_size_px}px (~{calib_size_cm:.1f}cm)", (10, img_h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    cv2.putText(img, "Press 'c' to center ROI on finger, '+'/'-' to resize, 'r' to reset", (10, img_h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cTime = time.time()
    fps = 1/(cTime - pTime) if pTime != 0 else 0
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Eye Control", img)

    key = cv2.waitKey(1) & 0xFF
    # Center ROI to current finger position (if available)
    if key == ord('c') and finger_x is not None and finger_y is not None:
        roi_center_x, roi_center_y = int(finger_x), int(finger_y)
    # increase ROI size
    elif key == ord('+') or key == ord('='):
        calib_size_px = min(MAX_ROI_PX, calib_size_px + 10)
        calib_size_cm = calib_size_px / pixels_per_cm
    # decrease ROI size
    elif key == ord('-'):
        calib_size_px = max(MIN_ROI_PX, calib_size_px - 10)
        calib_size_cm = calib_size_px / pixels_per_cm
    # reset ROI to center
    elif key == ord('r'):
        roi_center_x, roi_center_y = wCam // 2, hCam // 2
        calib_size_px = int(calib_size_cm * pixels_per_cm)

    if key == ord('q'):
        breakq

cap.release()
cv2.destroyAllWindows()
