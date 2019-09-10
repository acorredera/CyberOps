import cv2
import glob
import random
import math
import numpy as np
import dlib
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
import os

PATH = os.getcwd()

emotions = ["anger", "happy", "neutral", "sadness", "surprise"]  # Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Contrast Limited Adaptive Histogram Equalization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/marta/PycharmProjects/CYBEROPS/RECONOCIMIENTO_FACIAL/MODELOS/shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file

clf1 = joblib.load(PATH+'/SVC_Classifier_py35.pkl')
clf2 = joblib.load(PATH+'/RF_Classifier_py35.pkl')
clf3 = joblib.load(PATH+'/ET_Classifier_py35.pkl')
clf4 = joblib.load(PATH+'/GB_Classifier_py35.pkl')
clf5 = joblib.load(PATH+'/DT_Classifier_py35.pkl')
clf6 = joblib.load(PATH+'/BC_Classifier_py35.pkl')
# clf7 = joblib.load('/home/marta/PycharmProjects/untitled/pyAudioAnalysis/MLP__Classifier.pkl')
clf7 = MLPClassifier(activation='tanh', solver='lbfgs', ) #change activation ({identity, logistic, tanh, relu}, default relu), solver ({lbfgs, sgd, adam}, default adam)


data = {}  # Make dictionary for all values


# AAdata['landmarks_vectorised'] = []

def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("/home/marta/PycharmProjects/CYBEROPS/RECONOCIMIENTO_FACIAL/BANCO_IMAGENES/IMAGENES_PROPIAS/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


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


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    training_prediction_classifiers = []
    testing_prediction_classifiers =[]
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)
        # Append AAdata to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised'])  # List of landmarks results for the image, appended image array to training AAdata list
                prediction1 = int(clf1.predict(training_data))
                prediction2 = int(clf2.predict(training_data))
                prediction3 = int(clf3.predict(training_data))
                prediction4 = int(clf4.predict(training_data))
                prediction5 = int(clf5.predict(training_data))
                prediction6 = int(clf6.predict(training_data))
                training_prediction_classifiers.append([prediction1, prediction2, prediction3, prediction4, prediction5, prediction6])
                # print(training_prediction_classifiers)
                training_labels.append(emotions.index(emotion))
                training_data.pop()

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised']) # List of landmarks results for the image
                prediction1 = int(clf1.predict(prediction_data))
                prediction2 = int(clf2.predict(prediction_data))
                prediction3 = int(clf3.predict(prediction_data))
                prediction4 = int(clf4.predict(prediction_data))
                prediction5 = int(clf5.predict(prediction_data))
                prediction6 = int(clf6.predict(prediction_data))
                testing_prediction_classifiers.append([prediction1, prediction2, prediction3, prediction4, prediction5, prediction6])
                # print(testing_prediction_classifiers)
                prediction_labels.append(emotions.index(emotion)) # Label of the image
                prediction_data.pop()

    return training_prediction_classifiers, training_labels, testing_prediction_classifiers, prediction_labels

accur_lin = []
for i in range(0, 1): #range(0,10)
    print("Making sets %s" % i)  # Make sets by random sampling 80/20%
    training_prediction_classifiers, training_labels, testing_prediction_classifiers, prediction_labels = make_sets()

    npar_train = np.array(training_prediction_classifiers)  # Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training classifier %s" % i)  # train SVM
    clf7.fit(npar_train, training_labels)

    print("getting accuracies %s" % i)  # Use score() function to get accuracy
    npar_pred = np.array(testing_prediction_classifiers)
    pred_lin = clf7.score(npar_pred, prediction_labels)
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin)  # Store accuracy in a list

print("Mean value class: %s" % np.mean(accur_lin))  # FGet mean accuracy of the 10 runs

from sklearn.externals import joblib
joblib.dump(clf7, PATH+'/MLP_prediction_Classifier_py35.pkl') #Save classifier