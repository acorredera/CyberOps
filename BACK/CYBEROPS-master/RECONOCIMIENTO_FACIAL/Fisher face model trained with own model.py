import cv2
import glob
import random
import numpy as np
import os

PATH = os.getcwd() #/home/marta/PycharmProjects/CYBEROPS/RECONOCIMIENTO_FACIAL

emotions = [ "anger", "happy", "neutral", "sadness", "surprise"]  # Emotion list
fishface = cv2.face.FisherFaceRecognizer_create()  # Initialize fisher face classifier

data = {}

def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob(PATH+"/BANCO_IMAGENES/IMAGENES_PROPIAS/CascadeClassifier/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.9)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.1):]  # get last 20% of file list
    return training, prediction


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        # Append AAdata to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            training_data.append(gray)  # append image array to training AAdata list
            training_labels.append(emotions.index(emotion))

        for item in prediction:  # repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    print("training fisher face classifier")
    print("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))

    print("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100 * correct) / (correct + incorrect))


# Now run it
metascore = []
for i in range(0, 10):
    correct = run_recognizer()
    print("got", correct, "percent correct!")
    metascore.append(correct)

print("\n\nend score:", np.mean(metascore), "percent correct!")#72.2% of 354-281 (73) images

# Saving and loading the trained model

training_data, training_labels, prediction_data, prediction_labels = make_sets()
fishface.train(training_data, np.asarray(training_labels))
fishface.save(PATH+"/MODELOS/own_fisher_face_model_py35.xml")
# fishface.load("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/fishface_model.xml")

# The dataset used in this script is the CK+ dataset, based on the work of:

# Kanade, T., Cohn, J. F., & Tian, Y. (2000).
# Comprehensive database for facial expression analysis.
# Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG00), Grenoble, France, 46-53.

# Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010).
# The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression.
# Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.