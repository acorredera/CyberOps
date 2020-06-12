# FISHERFACE

import cv2
import glob
import random
import numpy as np
import os
import csv

PATH = os.getcwd() #/home/marta/PycharmProjects/RECONOCIMIENTO_FACIAL

faceDet = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_alt_tree.xml")

emotions = [ "anger", "happy", "neutral", "sadness", "surprise"]  # Emotion list

# Go over detected faces, stop at first detected face, return empty if no face.
# emotions_trained_model=["neutral", "anger", "happy", "sadness", "surprise"]
for emotion in emotions:

    if emotion == "anger":
        emotion_real = 1
    elif emotion == "happy":
        emotion_real = 2
    elif emotion == "neutral":
        emotion_real = 0
    elif emotion == "sadness":
        emotion_real = 3
    elif emotion == "surprise":
        emotion_real = 4

    # Modify image for classification
    files = glob.glob(PATH+"/BANCO_IMAGENES/IMAGENES_PROPIAS/%s/TEST/*.png" %emotion)
    # for picture in files:
    # print(picture)
    # if picture == path_new_file[0]:
    for file in files:
        # picture = PATH+'/Webcam_photos/' + filename
        frame = cv2.imread(file)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""

        # Cut and save face
        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            print("face found in file: %s" % file)
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                # cv2.imwrite(PATH+'/BANCO_IMAGENES/IMAGENES_PROPIAS/%s/GRAY_FILES/' + file, out) #(picture[19:]), out)  # Write image (picture[19:])
            except:
                 pass  # If error, pass file

        fishface = cv2.face.FisherFaceRecognizer_create() #fishface = cv2.createFisherFaceRecognizer()-> python 2  # Initialize fisher face classifier
        fishface.read(PATH+"/MODELOS/fishface_model_test.xml") #own_fisher_face_model_py35.xml # LOADING THE FISHER_RECOG ALREADY TRAINED AND SAVED (see face_emotion_recognition_classifier.py)

        pred, conf = fishface.predict(out)
        # emotions_trained_model=["neutral", "anger", "happy", "sadness", "surprise"]
        if pred == 0:
            emotion = "neutral"
            print(emotion)
        elif pred == 1:
            emotion = "anger"
            print(emotion)
        elif pred == 2:
            emotion = "happy"
            print(emotion)
        elif pred == 3:
            emotion = "sadness"
            print(emotion)
        elif pred == 4:
            emotion = "surprise"
            print(emotion)

        f = open((PATH+'/BANCO_IMAGENES/IMAGENES_PROPIAS/emotions.csv'), 'a')
        f.write(str(file) + ";" + str(emotion_real) + ";" + str(pred) + '\n')  # Give your csv text here.
        ## Python will convert \n to os.linesep
        f.close()


        # with open((PATH+'/BANCO_IMAGENES/IMAGENES_PROPIAS/%s/TEST/emotions_%s.csv' %(emotion,emotion)), 'a') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=';', lineterminator='\n')
        #     writer.writerow([file, emotion_real, pred])