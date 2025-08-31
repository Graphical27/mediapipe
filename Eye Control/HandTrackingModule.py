import cv2
import mediapipe as mp
import time 

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # Store handedness labels if available (e.g., 'Left' or 'Right')
        self.handednessLabels = []
        if self.results and self.results.multi_handedness:
            self.handednessLabels = [h.classification[0].label for h in self.results.multi_handedness]
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Return list of landmarks and bounding box for the selected hand.
        bbox is a tuple: (xMin, yMin, xMax, yMax) in pixel coordinates.
        """
        lmList = []
        bbox = None
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            xList, yList = [], []
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            if xList and yList:
                xMin, xMax = min(xList), max(xList)
                yMin, yMax = min(yList), max(yList)
                bbox = (xMin, yMin, xMax, yMax)
                if draw:
                    # Draw an expanded rectangle around the hand for visibility
                    cv2.rectangle(img, (xMin - 20, yMin - 20), (xMax + 20, yMax + 20), (0, 255, 0), 2)
        # Store for later use by utility methods (e.g., fingersUp)
        self.lmList = lmList
        self.bbox = bbox
        self.currentHandLabel = self.handednessLabels[handNo] if hasattr(self, 'handednessLabels') and len(self.handednessLabels) > handNo else None
        return lmList, bbox

    def fingersUp(self):
        """Return list [Thumb, Index, Middle, Ring, Pinky] where 1=up and 0=down.
        Call findHands+findPosition each frame before using this.
        """
        if not hasattr(self, 'lmList') or not self.lmList:
            return []

        tipIds = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb: horizontal check depends on handedness
        if hasattr(self, 'currentHandLabel') and self.currentHandLabel == 'Left':
            fingers.append(1 if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0]-1][1] else 0)
        else:  # Assume Right if unknown
            fingers.append(1 if self.lmList[tipIds[0]][1] > self.lmList[tipIds[0]-1][1] else 0)

        # Other fingers: tip above PIP (y smaller) -> finger up
        for tipId in tipIds[1:]:
            fingers.append(1 if self.lmList[tipId][2] < self.lmList[tipId-2][2] else 0)

        return fingers

    def findDistance(self, p1, p2, img=None, draw=True, r=10, t=3):
        """Compute Euclidean distance between two landmarks.
        Returns (length, img, lineInfo) where lineInfo=[x1, y1, x2, y2, cx, cy].
        If img is provided and draw=True, draws line and circles.
        Usage: length, img, lineInfo = detector.findDistance(8, 12, img, draw=True)
        """
        if not hasattr(self, 'lmList') or not self.lmList:
            return 0, img, None

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 255, 0), cv2.FILLED)

        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap =  cv2.VideoCapture(0);
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        print(lmList[4]) if len(lmList) != 0 else None
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.imshow("image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()