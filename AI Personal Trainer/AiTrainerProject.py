import cv2
import mediapipe as mp
import time
from PoseModule import PoseDetector as pm
import numpy as np
import math

cap = cv2.VideoCapture("C:\Games\!Projects\mediapipe\AI Personal Trainer\Trainer_1.mp4")
# IMG_PATH = r"C:\Games\!Projects\mediapipe\AI Personal Trainer\Trainer_2.png"
detector = pm()
pTime = 0

while True:
    # img = cv2.imread(IMG_PATH)  # Reload image every loop
    # if img is None:
    #     raise FileNotFoundError(f"Image not found: {IMG_PATH}")
    success, img = cap.read()
    img = detector.findPose(img, draw=False)
    lmlisst = detector.getPosition(img, draw=False)
    if len(lmlisst) != 0:
        angle = detector.findAngle(img, 12, 14, 16)
        # detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        print(angle, per)
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
