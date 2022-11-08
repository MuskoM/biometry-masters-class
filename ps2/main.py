import sys
import typing as t

import cv2
import numpy as np
import mediapipe as mp

from PySide6 import (
    QtCore as QCore,
    QtGui as QGui,
    QtWidgets as QWidgets,
)

faces_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('public/cascade_classifiers/haarcascade_mcs_mouth.xml')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)


def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms)

    faces = faces_cascade.detectMultiScale(frame_gray)
    for x,y,w,h in faces:
        faceROI = frame_gray[y:y+h, x: x+w]

        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (xe,ye,we,he) in eyes:
            eye_center = (x + xe + we//2, y + ye + he//2 )
            radius = int(round(we+he)*0.25)
            frame = cv2.circle(frame,eye_center, radius, (255,0,0), 4)
        
        mouth_rects = mouth_cascade.detectMultiScale(faceROI)
        for (xm,ym,wm,hm) in mouth_rects:
            frame = cv2.rectangle(frame, (x+xm,y+ym), (x + xm + wm,y + ym + hm), (0,255,0),3)
            break

    return frame


def run(zadanie: int):
    # Get camera
    cap = cv2.VideoCapture(0)

    #Check if camera is available
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    lower = np.array([0,10,60])
    upper = np.array([20,150,255])
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # EventLoop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is Trueq
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Face detection using image processing
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img_photo_processing = cv2.inRange(hsv_image ,lower, upper)
        erode = cv2.erode(img_photo_processing, kernel)
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        subset = erode - opening
        img_photo_processing = cv2.bitwise_or(subset, img_photo_processing)
        img_photo_processing = cv2.Sobel(img_photo_processing,cv2.CV_8U,2,2)

        # Our operations on the frame come here
        # Object detection with already available tools
        img_from_haar = detectAndDisplay(frame)
        
        # Display the resulting frame
        cv2.imshow('Using available tools', img_from_haar)
        cv2.imshow('Using image processiong', img_photo_processing)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()