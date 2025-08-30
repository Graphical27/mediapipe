import cv2
import mediapipe as mp
import time
import autopy
from HandTrackingModule import handDetector
import numpy as np
cap =  cv2.VideoCapture(0);

wCam, hCam = 640, 480
wScreen, hScreen = autopy.screen.size()
cap.set(3, wCam)
cap.set(4, hCam)

detector = handDetector(detectionCon=0.7,maxHands=1)
frameR = 100
smoothening = 7
plocX, plocY = 0,0
clocX, clocY = 0,0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img,draw=True)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]  
        x2, y2 = lmList[12][1], lmList[12][2]
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR ), (0, wScreen))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            plocX, plocY = clocX, clocY
            autopy.mouse.move(wScreen - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 0:
            length,img, linedistance = detector.findDistance(8,12,img,draw=True)
            if length < 40:
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1/(cTime - pTime) if 'pTime' in locals() else 0
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break