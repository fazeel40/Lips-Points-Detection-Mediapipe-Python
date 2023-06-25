import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh()

while 1:
    _,frame = webcam.read()
    frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output  = face_mesh.process(frame_grayscale)
    frame_landmarks = output.multi_face_landmarks
    frame_h,frame_w,_ = frame.shape
    if frame_landmarks:
        landmarks = frame_landmarks[0].landmark
        for id ,landmark in enumerate(landmarks):
            x = int(landmark.x*frame_w)
            y = int(landmark.y*frame_h)
            if id == 0:
                cv2.circle(frame,(x,y),3,(0,255,0))
            if id == 14:
                cv2.circle(frame,(x,y),3,(0,255,255))
    cv2.imshow("Detection",frame)
    cv2.waitKey(1)
webcam.release()