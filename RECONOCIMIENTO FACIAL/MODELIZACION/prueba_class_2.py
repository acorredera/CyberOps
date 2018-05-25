from sklearn.externals import joblib
import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.ensemble import VotingClassifier

# joblib.dump(clf, '/home/marta/PycharmProjects/untitled/pyAudioAnalysis/SVM_Classifier.pkl') #Save classifier
# clf = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/SVM_Classifier.pkl') #Load classifier

emotions = ["anger", "happy", "neutral", "sadness", "surprise"]  # Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Contrast Limited Adaptive Histogram Equalization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/marta/Descargas/shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file

data = {}  # Make dictionary for all values

prediction_data = []
prediction_labels = []


from statistics import mode
from sklearn.externals import joblib
from nltk.classify import ClassifierI
import nltk
import statistics

voted_class_list = []

def get_landmarks(image):

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

Webcam_folder_dataset = glob.glob("/home/marta/Webcam/dataset/*.png")

for gray_image in Webcam_folder_dataset:
    print(gray_image)
    gray_image = cv2.imread(gray_image) #open item
    img_grey = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    # out = cv2.resize(img_grey, (350, 350))
    clahe_image = clahe.apply(img_grey)
    get_landmarks(clahe_image)
    if data['landmarks_vectorised'] == "error":
        print("no face detected on this one")
    else:
        # cv2.imwrite("/home/marta/Webcam/dataset/%s" % (clahe_image, out)) # Write image
        # emotions = ["anger"-0, "happiness"-1, "neutral"-2, "sadness"-3, "surprise"-4]  # Emotion list
        prediction_data.append(data['landmarks_vectorised'])
        print(prediction_data)
        clf1 = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/SVM_Classifier_linear.pkl')
        clf2 = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/RF_Classifier.pkl')
        clf3 = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/ET_Classifier.pkl')
        clf4 = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/GB2_Classifier.pkl')
        clf5 = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/DT_Classifier.pkl')
        clf6 = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/Bagg_Classifier.pkl')

        v_clf = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/VC2_Classifier_hard.pkl')
        v_clf2 = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/VC_Classifier_soft.pkl')

        prediction1 = int(clf1.predict(prediction_data))
        prediction2 = int(clf2.predict(prediction_data))
        prediction3 = int(clf3.predict(prediction_data))
        prediction4 = int(clf4.predict(prediction_data))
        prediction5 = int(clf5.predict(prediction_data))
        prediction6 = int(clf6.predict(prediction_data))
        voted_class_list.append([prediction1, prediction2, prediction3, prediction4, prediction5, prediction6])

        ensemble = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/MLP_prediction_Classifier.pkl')
        ensemble_prediction = ensemble.predict(voted_class_list)

        v_emotion_hard = v_clf.predict(prediction_data)
        v_emotion_soft = v_clf2.predict(prediction_data)
        # voted_class_list = voted_class_list.append(emotion) #, emotion2[0], emotion3[0], emotion4[0], emotion5[0], emotion6[0], emotion7[0]
        prediction_data.pop()
        voted_class_list.pop()

    print(prediction1)
    print(prediction2)
    print(prediction3)
    print(prediction4)
    print(prediction5)
    print(prediction6)
    print(ensemble_prediction)
    print(v_emotion_hard)
    print(v_emotion_soft)


# clf.predict("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/dataset/anger/10.jpg")