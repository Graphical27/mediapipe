import cv2
import numpy as np
import time
from HandTrackingModule import handDetector

# AI Virtual Painter using MediaPipe Hands
# Gestures:
# - Both index and middle fingers up: Selection mode (choose color from top bar)
# - Only index finger up: Drawing mode
# - Press 'c' to clear canvas, 'q' to quit


def fingers_up(lm_list):
    if not lm_list or len(lm_list) < 21:
        return [0, 0, 0, 0, 0]

    fingers = [0, 0, 0, 0, 0]

    thumb_tip_x = lm_list[4][1]
    thumb_mcp_x = lm_list[2][1]
    fingers[0] = 1 if thumb_tip_x > thumb_mcp_x else 0

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for i, (tip, pip) in enumerate(zip(tips, pips), start=1):
        fingers[i] = 1 if lm_list[tip][2] < lm_list[pip][2] else 0

    return fingers


def main():
    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # width
    cap.set(4, 720)   # height

    detector = handDetector(detectionCon=0.7, trackCon=0.7)

    # Drawing helpers
    brush_thickness = 15
    eraser_thickness = 60

    # Color palette (BGR)
    colors = [
        (0, 0, 255),   # Red
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
        (0, 255, 255), # Yellow
        (0, 0, 0)      # Eraser (draw black on canvas)
    ]
    labels = ["Red", "Green", "Blue", "Yellow", "Eraser"]
    draw_color = colors[0]

    success, frame = cap.read()
    if not success:
        print("Unable to access camera. Exiting.")
        return
    h, w = frame.shape[:2]
    img_canvas = np.zeros((h, w, 3), np.uint8)

    bar_height = 100
    cell_width = w // len(colors)

    xp, yp = 0, 0 
    p_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)

        img = detector.findHands(img, draw=False)
        lm_list = detector.findPosition(img, draw=False)

        for i, (color, label) in enumerate(zip(colors, labels)):
            x1, x2 = i * cell_width, (i + 1) * cell_width
            if label == "Eraser":
                cv2.rectangle(img, (x1, 0), (x2, bar_height), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, label, (x1 + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            else:
                cv2.rectangle(img, (x1, 0), (x2, bar_height), color, cv2.FILLED)
                cv2.rectangle(img, (x1, 0), (x2, bar_height), (255, 255, 255), 2)
                cv2.putText(img, label, (x1 + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        if lm_list:
            x1, y1 = lm_list[8][1], lm_list[8][2]
            x2, y2 = lm_list[12][1], lm_list[12][2]

            f_up = fingers_up(lm_list)
            index_up, middle_up = f_up[1] == 1, f_up[2] == 1

            if index_up and middle_up:
                xp, yp = 0, 0 
                cv2.rectangle(img, (x1 - 25, y1 - 25), (x2 + 25, y2 + 25), (255, 0, 255), 2)
                if y1 < bar_height and y2 < bar_height:
                    cell_index = min(x1 // cell_width, len(colors) - 1)
                    draw_color = colors[cell_index]
            elif index_up and not middle_up:
                cv2.circle(img, (x1, y1), 10, draw_color if draw_color != (0, 0, 0) else (255, 255, 255), cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                thickness = eraser_thickness if draw_color == (0, 0, 0) else brush_thickness
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, thickness)
                xp, yp = x1, y1

            else:
                xp, yp = 0, 0

        # Merge canvas with the camera image
        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, img_canvas)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if c_time != p_time else 0
        p_time = c_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("AI Virtual Painter", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            img_canvas = np.zeros((h, w, 3), np.uint8)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()