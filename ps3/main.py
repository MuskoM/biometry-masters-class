import sys
import typing as t
import inspect

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from imutils import paths
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
from imutils import build_montages
import numpy as np
import time

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
            saved_landmarks = landmarks
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
            frame = cv2.rectangle(frame, (x+xm,y+ym), (x + xm + wm, y + ym + hm), (0,255,0),3)
            break

    return frame, saved_landmarks, 

def load_faces(inputPath):
    imagePaths = list(paths.list_images(inputPath))
    print(f'imagePaths: {imagePaths}\n')
    names = [p.split(os.path.sep)[-2] for p in imagePaths]
    print(f'names: {names}')
    (names, counts) = np.unique(names, return_counts=True)
    names = names.tolist()

    faces = []
    labels = []

    for imagePath in imagePaths:
        landmarks = []
        image = cv2.imread(imagePath)
        img_from_haar, saved_landmarks = detectAndDisplay(image, landmarks)
        cv2.imshow('Using available tools', img_from_haar)
        while True:
            if cv2.waitKey(1) == ord('q'):
                break
    return faces, labels

def run(zadanie: int):
    landmarks = []
    # # Get camera
    # cap = cv2.VideoCapture(0)

    # #Check if camera is available
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()

    lower = np.array(80)
    upper = np.array(230)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    faces, labels = load_faces("ps3/zdjecia/treningowe")


    # landmarks = saved_landmarks
    #     # Display the resulting frame
    # while True:
    #     cv2.imshow('Using available tools', img_from_haar)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # # EventLoop
    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
        
    #     # if frame is read correctly ret is Trueq
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break

    #     img_from_haar, saved_landmarks = detectAndDisplay(frame, landmarks)
    #     landmarks = saved_landmarks
    #     # Display the resulting frame
    #     cv2.imshow('Using available tools', img_from_haar)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # # When everything done, release the capture
    # cap.release()
