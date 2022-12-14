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
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
def detectAndDisplay(img, scaleFactor=1.2, minNeighbors=5):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faces_cascade.detectMultiScale(img_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for x,y,w,h in faces:
        faceROI = img_gray[y:y+w, x:x+h]
        eyes = eyes_cascade.detectMultiScale(faceROI, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        for (xe,ye,we,he) in eyes:
            eye_center = (x + xe + we//2, y + ye + he//2 )
            radius = int(round(we+he)*0.25)
            img = cv2.circle(img,eye_center, radius, (255,0,0), 4)
        
        mouth_rects = mouth_cascade.detectMultiScale(faceROI, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        for (xm,ym,wm,hm) in mouth_rects:
            img = cv2.rectangle(img, (x+xm,y+ym), (x + xm + wm, y + ym + hm), (0,255,0),3)
            break
        img = cv2.rectangle(img, (x,y), (x + w, y + h), (0,0,255),3)
    return img 

def detectFace(img, scaleFactor=1.2, minNeighbors=5):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faces_cascade.detectMultiScale(img_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return img_gray[y:y+w, x:x+h], faces[0]

def load_faces(inputPath):
    dirs = os.listdir(inputPath)

    faces = []
    labels = []
    
    for dir_name in dirs:
        label = int(dir_name.split("_")[-1])
        subject_dir_path = inputPath + "/" + dir_name
        imageNames = os.listdir(subject_dir_path)
        for imageName in imageNames:
            imagePath = subject_dir_path + "/" + imageName
            print(f'Reading file {imagePath}...\n')
            image = cv2.imread(imagePath)
            cv2.imshow("Training on image", image)
            cv2.waitKey(100)
            # img_from_haar = detectAndDisplay(image)
            face, faceROI = detectFace(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

def run(zadanie: int):
    # # Get camera
    # cap = cv2.VideoCapture(0)

    # #Check if camera is available
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()

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
