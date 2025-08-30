import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self, mode=False, maxFaces=2, refineLandmarks=False,
                 minDetectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.mode,
            max_num_faces=self.maxFaces,
            refine_landmarks=self.refineLandmarks,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                        self.drawSpec, self.drawSpec
                    )
                face = []
                h, w, c = img.shape
                for id, lm in enumerate(faceLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face.append((cx, cy))
                faces.append(face)
        return img, faces

    def getFacePositions(self, img, faceNo=0):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        face = []
        if results.multi_face_landmarks:
            if faceNo < len(results.multi_face_landmarks):
                faceLms = results.multi_face_landmarks[faceNo]
                h, w, c = img.shape
                for id, lm in enumerate(faceLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face.append((id, cx, cy))
        return face

        
def main():
    cap = cv2.VideoCapture(r"C:\Games\!Projects\mediapipe\Face Mesh\Face_1.mp4")
    detector = FaceMeshDetector(maxFaces=2)
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        img, faces = detector.findFaceMesh(img, draw=True)
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()