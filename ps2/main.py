import sys
import typing as t
import inspect

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

from PySide6 import (
    QtCore as QCore,
    QtGui as QGui,
    QtWidgets as QWidgets,
)

@dataclass
class Landmark:
    x: float
    y: float
    z: float

faces_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('public/cascade_classifiers/haarcascade_mcs_mouth.xml')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5)

force = 0 

def calculate_avg_force(landmarks, saved_landmarks):
    try:
        x_forces = [0]
        y_forces = [0]
        z_forces = [0]

        for l, sl in zip(landmarks, saved_landmarks):
            x_forces.append(l.x-sl.x)
            y_forces.append(l.y-sl.y)
            z_forces.append(l.z-sl.z)

        x_avg = round(abs(np.average(x_forces)),2)
        y_avg = round(abs(np.average(y_forces)),2)
        z_avg = round(abs(np.average(z_forces)),2)
    except Exception:
        x_avg = 0
        y_avg = 0
        z_avg = 0
    return x_avg, y_avg, z_avg
    
    


def detectAndDisplay(frame, saved_landmarks):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            landmarks = [x for x in faceLms.ListFields()]
            landmarks = landmarks[0][1][:10]
            if saved_landmarks is None:
                saved_landmarks = landmarks
            forces = calculate_avg_force(landmarks, saved_landmarks)
            saved_landmarks = landmarks
            cv2.putText(frame, str(forces), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            mpDraw.draw_landmarks(frame, faceLms, landmark_drawing_spec=drawing_spec)

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

    return frame, saved_landmarks, 


def run(zadanie: int):
    landmarks = []
    # Get camera
    cap = cv2.VideoCapture(0)

    #Check if camera is available
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    lower = np.array(90)
    upper = np.array(250)
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
        bw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        img_photo_processing = cv2.inRange(bw_img ,lower, upper)
        erode = cv2.erode(img_photo_processing, kernel)
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        subset = erode - opening
        img_photo_processing = cv2.bitwise_or(subset, img_photo_processing)
        img_photo_processing = cv2.Sobel(img_photo_processing, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

        # Our operations on the frame come here
        # Object detection with already available tools
        img_from_haar, saved_landmarks = detectAndDisplay(frame, landmarks)
        landmarks = saved_landmarks
        # Display the resulting frame
        cv2.imshow('Using available tools', img_from_haar)
        cv2.imshow('Using image processiong', img_photo_processing)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()