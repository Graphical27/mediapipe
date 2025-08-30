import cv2
import time 
import PoseModule as pm

pTime = 0
cTime = 0
cap =  cv2.VideoCapture("C:\Games\!Projects\mediapipe\Pose Estimation\Run_1.mp4")
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getPosition(img)
    if len(lmList) != 0:
        print(lmList[14])  # Example: print the position of landmark 14
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break