from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor('face_landmark/shape_predictor_5_face_landmarks.dat')

cam = cv2.VideoCapture(0)

while True:
    ret,image = cam.read()
    if ret:
        print('IMage Read')
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        rects = face_detector(gray, 1)

        print(len(rects))
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            landmarks = landmark_detector(gray, rect,)
            landmarks = face_utils.shape_to_np(landmarks)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(image, f"Face {i + 1}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # for (x, y) in landmarks:
            cv2.circle(image, landmarks[-1], 1, (0, 0, 255), -1)
        cv2.imshow('Live feed',image)
    
    if cv2.waitKey(1)==27:
        break

cam.release()
cv2.destroyAllWindows()