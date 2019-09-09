# FISHERFACE

import cv2
import glob
import random
import numpy as np
import os
import csv
import math
import dlib
from sklearn.externals import joblib
import settings as settings

def get_landmarks(image,predictor,detector):
    data = {}  # Make dictionary for all values

    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"
    return  data

# ------------------------------------------------------------

PATH = os.getcwd() #/home/marta/PycharmProjects/RECONOCIMIENTO_FACIAL

faceDet = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_alt_tree.xml")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Contrast Limited Adaptive Histogram Equalization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PATH + "/MODELOS/shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file

prediction_data = []
voted_class_list = []

emotions = [ "anger", "happy", "neutral", "sadness", "surprise"]  # Emotion list

for emotion in emotions:

    if emotion == "anger":
        emotion_real = 0
    elif emotion == "happy":
        emotion_real = 1
    elif emotion == "neutral":
        emotion_real = 2
    elif emotion == "sadness":
        emotion_real = 3
    elif emotion == "surprise":
        emotion_real = 4

        # Modify image for classification
    files = glob.glob(PATH + "/BANCO_IMAGENES/IMAGENES_PROPIAS/%s/TEST/*.png" % emotion)
    # for picture in files:
    # print(picture)
    # if picture == path_new_file[0]:
    for file in files:
        # picture = PATH+'/Webcam_photos/' + filename
        frame = cv2.imread(file)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

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

       # Pass classifier
       #  gray_files = glob.glob(PATH+"/Webcam_photos/Classifier/*.png")
       #  # for gray_picture in gray_files:
       #  #     if gray_picture == PATH+"/Webcam_photos/Classifier/%s" %filename:
       #  gray_picture = PATH+'/Webcam_photos/Classifier/' + file

        # gray_image = cv2.imread(out)  # open photo
        # img_grey = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        clahe_image = clahe.apply(out) # Aplying Contrast Limited Adaptive Histogram Equalization to create a grater contrast b/w on the photo
        data = get_landmarks(clahe_image,predictor,detector) #getting landmarks from photo

        if data['landmarks_vectorised'] == "error":
            print("no face detected on this one")
        else:
            # cv2.imwrite("/home/marta/Webcam/dataset/%s" % (clahe_image, out)) # Write image

            prediction_data.append(data['landmarks_vectorised'])
            clf1 = joblib.load(PATH+'/MODELOS/SVC_own_dataset_Classifier_py35.pkl')
            clf2 = joblib.load(PATH+'/MODELOS/RF_own_dataset_Classifier_py35.pkl')
            clf3 = joblib.load(PATH+'/MODELOS/ET_own_dataset_Classifier_py35.pkl')
            clf4 = joblib.load(PATH+'/MODELOS/GB_own_dataset_Classifier_py35.pkl')
            clf5 = joblib.load(PATH+'/MODELOS/DT_own_dataset_Classifier_py35.pkl')
            clf6 = joblib.load(PATH+'/MODELOS/BC_own_dataset_Classifier_py35.pkl')
            # clf7 = joblib.load(PATH+'/MODELOS/MLP__Classifier.pkl')
            # v_clf = joblib.load(PATH+'/MODELOS/VC2_Classifier_hard.pkl')
            # v_clf2 = joblib.load(PATH+'/MODELOS/VC_Classifier_soft.pkl')
            voted_classifier = joblib.load(PATH+'/MODELOS/MLP_own_dataset_prediction_Classifier_py35.pkl')

            prediction1 = int(clf1.predict(prediction_data))
            prediction2 = int(clf2.predict(prediction_data))
            prediction3 = int(clf3.predict(prediction_data))
            prediction4 = int(clf4.predict(prediction_data))
            prediction5 = int(clf5.predict(prediction_data))
            prediction6 = int(clf6.predict(prediction_data))
            voted_class_list.append([prediction1, prediction2, prediction3, prediction4, prediction5, prediction6])
            voted_classifier_prediction = voted_classifier.predict(voted_class_list)
            # prediction7 = clf7.predict(prediction_data)
            # v_prediction_hard = v_clf.predict(prediction_data)
            # v_prediction_soft = v_clf2.predict(prediction_data)

        print(prediction1)
        print(prediction2)
        print(prediction3)
        print(prediction4)
        print(prediction5)
        print(prediction6)
        # print(prediction7)
        # print(v_prediction_hard)
        # print(v_prediction_soft)
        print(voted_classifier_prediction)

        # emotions_trained_model=["anger"-0, "happy"-1, "neutral"-2, "sadness"-3, "surprise"-4]
        if voted_classifier_prediction == 0:
            emotion = "anger"
            print(emotion)
        elif voted_classifier_prediction == 1:
            emotion = "happy"
            print(emotion)
        elif voted_classifier_prediction == 2:
            emotion = "neutral"
            print(emotion)
        elif voted_classifier_prediction == 3:
            emotion = "sadness"
            print(emotion)
        elif voted_classifier_prediction == 4:
            emotion = "surprise"
            print(emotion)

        prediction_data.pop() #delating previous prediction appended to data_prediction array
        voted_class_list.pop()

        # write timestamp and emotion in csv file
        with open(PATH+'/BANCO_IMAGENES/IMAGENES_PROPIAS/emotions_landmarks.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=';', lineterminator='\n')
            writer.writerow([file, emotion_real, voted_classifier_prediction[0]])



#
#
#
#
# faceDet = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
# faceDet_two = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_alt2.xml")
# faceDet_three = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_alt.xml")
# faceDet_four = cv2.CascadeClassifier(PATH+"/OpenCV_FaceCascade/haarcascade_frontalface_alt_tree.xml")
#
# emotions = [ "anger", "happy", "neutral", "sadness", "surprise"]  # Emotion list
#
# # Go over detected faces, stop at first detected face, return empty if no face.
# # emotions = ["neutral", "anger", "happy", "sadness", "surprise"] #FISHER FACE EMOTIONS
# for emotion in emotions:
#
#     if emotion == "anger":
#         emotion_real = 1
#     elif emotion == "happy":
#         emotion_real = 2
#     elif emotion == "neutral":
#         emotion_real = 0
#     elif emotion == "sadness":
#         emotion_real = 3
#     elif emotion == "surprise":
#         emotion_real = 4
#
#     # Modify image for classification
#     files = glob.glob(PATH+"/BANCO_IMAGENES/IMAGENES_PROPIAS/%s/TEST/*.png" %emotion)
#     # for picture in files:
#     # print(picture)
#     # if picture == path_new_file[0]:
#     for file in files:
#         # picture = PATH+'/Webcam_photos/' + filename
#         frame = cv2.imread(file)  # Open image
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
#
#         face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#         face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#         face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#         face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#
#         if len(face) == 1:
#             facefeatures = face
#         elif len(face_two) == 1:
#             facefeatures = face_two
#         elif len(face_three) == 1:
#             facefeatures = face_three
#         elif len(face_four) == 1:
#             facefeatures = face_four
#         else:
#             facefeatures = ""
#
#         # Cut and save face
#         for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
#             print("face found in file: %s" % file)
#             gray = gray[y:y + h, x:x + w]  # Cut the frame to size
#             try:
#                 out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
#                 # cv2.imwrite(PATH+'/BANCO_IMAGENES/IMAGENES_PROPIAS/%s/GRAY_FILES/' + file, out) #(picture[19:]), out)  # Write image (picture[19:])
#             except:
#                  pass  # If error, pass file
#
#         fishface = cv2.face.FisherFaceRecognizer_create() #fishface = cv2.createFisherFaceRecognizer()-> python 2  # Initialize fisher face classifier
#         fishface.read(PATH+"/MODELOS/fishface_model_test.xml") #own_fisher_face_model_py35.xml # LOADING THE FISHER_RECOG ALREADY TRAINED AND SAVED (see face_emotion_recognition_classifier.py)
#
#         pred, conf = fishface.predict(out)
#         # emotions_trained_model=["neutral", "anger", "happy", "sadness", "surprise"]
#         if pred == 0:
#             emotion = "neutral"
#             print(emotion)
#         elif pred == 1:
#             emotion = "anger"
#             print(emotion)
#         elif pred == 2:
#             emotion = "happy"
#             print(emotion)
#         elif pred == 3:
#             emotion = "sadness"
#             print(emotion)
#         elif pred == 4:
#             emotion = "surprise"
#             print(emotion)
#
#         f = open((PATH+'/BANCO_IMAGENES/IMAGENES_PROPIAS/emotions.csv'), 'a')
#         f.write(str(file) + ";" + str(emotion_real) + ";" + str(pred) + '\n')  # Give your csv text here.
#         ## Python will convert \n to os.linesep
#         f.close()
