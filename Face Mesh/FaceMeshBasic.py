import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("C:\Games\!Projects\mediapipe\Face Mesh\Face_1.mp4");
# cap = cv2.VideoCapture(0);
mpFaceMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
# faceMesh = mpFaceMesh.FaceMesh() #? Both are correct
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_CONTOURS,
                                 drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                print(id)
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(cx, cy)
                cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1/(cTime - pTime) if 'pTime' in locals() else 0
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
