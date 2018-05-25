import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

emotions = ["anger", "happy", "neutral", "sadness", "surprise"]  # Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Contrast Limited Adaptive Histogram Equalization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/marta/Descargas/shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file

# clf = SVC(kernel='linear', probability=True, tol=1e-3)  # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel
clf1 = SVC(kernel='linear', probability=True)
# clf = SVC(kernel='poly')
# clf = SVC(kernel='rbf')
# clf = SVC(kernel='sigmoid')
clf2 = GradientBoostingClassifier() #max_depth = 100
clf3 = RandomForestClassifier () #max_depth = 70
clf4 = ExtraTreesClassifier(n_estimators= 70, criterion='gini', max_depth=None) #gini or entropy
clf5 = DecisionTreeClassifier()
clf6 = BaggingClassifier()
# clf7 = MLPClassifier(activation='tanh', solver='lbfgs') #change activation ({identity, logistic, tanh, relu}, default relu), solver ({lbfgs, sgd, adam}, default adam)
# v_clf = VotingClassifier(estimators=[('svm', clf1), ('gb', clf2), ('rf', clf3), ('et', clf4), ('dt', clf5), ('bc', clf6)], voting='hard') #, weights=[1,1,2,1]

data = {}  # Make dictionary for all values


# data['landmarks_vectorised'] = []

def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("/home/marta/Own_dataset/Gray_dataset/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 1)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.05):]  # get last 20% of file list
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
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised'])  # append image array to training data list
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                print(prediction_data)
                # print(len(prediction_data))
                prediction_labels.append(emotions.index(emotion))
                print(prediction_labels)

    return training_data, training_labels, prediction_data, prediction_labels


accur_lin = []
for i in range(0, 1): #range(0,10)
    print("Making sets %s" % i)  # Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training classifier %s" % i)  # train SVM
    clf1.fit(npar_train, training_labels)

    print("getting accuracies %s" % i)  # Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf1.score(npar_pred, prediction_labels)
    print "linear: ", pred_lin
    accur_lin.append(pred_lin)  # Store accuracy in a list

print("Mean value class: %s" % np.mean(accur_lin))  # FGet mean accuracy of the 10 runs

from sklearn.externals import joblib
joblib.dump(clf1, '/home/marta/PycharmProjects/untitled/pyAudioAnalysis/SVC_own_dataset_Classifier.pkl') #Save classifier