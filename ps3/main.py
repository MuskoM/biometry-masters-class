import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
import os

@dataclass
class Landmark:
    x: float
    y: float
    z: float

subjects = ["", "George_W_Bush", "Kinga", "Mateusz", "Winona_Ryder"]
faces_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('public/cascade_classifiers/haarcascade_mcs_mouth.xml')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
def detectAndDisplay(source_img, scaleFactor=1.2, minNeighbors=5):
    img = source_img.copy()
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

def detectFace(source_img, scaleFactor=1.2, minNeighbors=5):
    img = source_img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faces_cascade.detectMultiScale(img_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return img_gray[y:y+w, x:x+h], faces[0]

def detectAllFaces(source_img, scaleFactor=1.2, minNeighbors=5):
    img = source_img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faces_cascade.detectMultiScale(img_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    rects = []
    faces_imgs = []
    for x,y,w,h in faces:
        rects.append((x,y,w,h))
        faces_imgs.append(img_gray[y:y+w, x:x+h])
    return faces_imgs, rects

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
            # cv2.imshow("Training on image", image)
            cv2.waitKey(100)
            # img_from_haar = detectAndDisplay(image)
            print(f'Detecting face \n')
            face, faceROI = detectFace(image)
            if face is not None:
                print(f'Face detected for {imagePath}\n')
                face = cv2.resize(face, (47,62))
                faces.append(face)
                labels.append(label)
            else:
                print(f'Face not detected in {imagePath}')
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

def predict(source_img, face_recognizer):
    img = source_img.copy()
    print(f'Detecting face to predict')

    faces, rects = detectAllFaces(img)
    for idx, face in enumerate(faces):
        if face is None:
            print(f'Face not detected from camera')
            continue
        face = cv2.resize(face, (47,62))
        label, confidence = face_recognizer.predict(face)

        confidence = round(100 * (1000 / confidence), 2)
        label_text = f'{subjects[label]} {confidence}%'
        
        #draw a rectangle around face detected
        (x, y, w, h) = rects[idx]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #draw name of predicted person
        cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
    return img

def trainOnData(inputPath):
    print("Preparing training data\n")
    faces, labels = load_faces(inputPath)
    print("Data successfully read")
    print(f'Faces detected: {len(faces)}')
    print(f'Total labels: ', len(labels))

    recognise = cv2.face.EigenFaceRecognizer_create()  
    recognise.train(faces, np.array(labels))
    return recognise

def run(zadanie: int):
    model = trainOnData("ps3/zdjecia/treningowe")

    # Get camera
    cap = cv2.VideoCapture(0)

    #Check if camera is available
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # EventLoop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is Trueq
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        predicted_img = predict(frame, model)

        cv2.imshow("Predicted image", predicted_img)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()