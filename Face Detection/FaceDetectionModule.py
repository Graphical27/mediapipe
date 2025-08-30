import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()
        self.pTime = 0

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                break
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.faceDetection.process(imgRGB)

            if results.detections:
                for id, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = img.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cTime = time.time()
            fps = 1 / (cTime - self.pTime) if self.pTime else 0
            self.pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv2.imshow("image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


detector = FaceDetector(r"C:\Games\!Projects\mediapipe\Face Detection\Face_1.mp4")
detector.run()
