# CITATION

#  van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset.
# A tech blog about fun things with Python and embedded electronics.
# Retrieved from:
# http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/


# From each image sequence of the dataset, extract two images: one neutral (the first image) and one with an emotional expression (the last image)

# import glob
# from shutil import copyfile
#
# emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotion order
# participants = glob.glob("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/source_emotion/*")  # Returns a list of all folders with participant numbers
# # print(participants)
# for x in participants:
#     part = "%s" % x[-4:]  # store current participant number
#     # print(part)
#     for sessions in glob.glob("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/source_emotion/%s/*" % part):  # Store list of sessions for current participant
#         # print(sessions)
#         for files in glob.glob("%s/*emotion.txt" % sessions):
#             # print(files)
#             current_session = files[73:-30]
#             # print(current_session)
#             file = open(files, 'r')
#
#             emotion = int(float(file.readline()))  # emotions are encoded as a float, readline as float, then convert to integer.
#
#             sourcefile_emotion = glob.glob("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/source_images/%s/%s/*" % (part, current_session))[-1]  # get path for last image in sequence, which contains the emotion
#             sourcefile_neutral = glob.glob("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/source_images/%s/%s/*" % (part, current_session))[0]  # do same for neutral image
#
#             # print(sourcefile_emotion[67:])
#             # print(sourcefile_neutral)
#
#             dest_neut = "/home/marta/PycharmProjects/untitled/pyAudioAnalysis/sorted_set/neutral/%s" % sourcefile_neutral[76:]  # Generate path to put neutral image
#             dest_emot = "/home/marta/PycharmProjects/untitled/pyAudioAnalysis/sorted_set/%s/%s" % (emotions[emotion], sourcefile_emotion[76:])  # Do same for emotion containing image
#
#             copyfile(sourcefile_neutral, dest_neut)  # Copy file
#             copyfile(sourcefile_emotion, dest_emot)  # Copy file
#
# # Extracting faces. The classifier will work best if the training and classification images are all of the same size and have only a face on them
# # We need to find the face on each image, convert to grayscale, crop it and save the image to the dataset
# # Use HAAR filter from OpenCV to automate face finding. OpenCV provides 4 pretrained classifiers, so to detect as many faces as possible lets use all of them in sequence, and abort the face search once we have found one.
#
# import glob
# import cv2
#
# faceDet = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
# faceDet_two = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_alt2.xml")
# faceDet_three = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_alt.xml")
# faceDet_four = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_alt_tree.xml")
#
# emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotions
#
#
# def detect_faces(emotion):
#     files = glob.glob("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/sorted_set/%s/*" % emotion)  # Get list of all images with emotion
#
#     filenumber = 0
#     for f in files:
#         frame = cv2.imread(f)  # Open image
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
#
#         # Detect face using 4 different classifiers
#         face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#         face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#         face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#         face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
#
#         # Go over detected faces, stop at first detected face, return empty if no face.
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
#             print "face found in file: %s" % f
#             gray = gray[y:y + h, x:x + w]  # Cut the frame to size
#
#             try:
#                 out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
#                 cv2.imwrite("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/dataset/%s/%s.jpg" % (emotion, filenumber), out)  # Write image
#             except:
#                 pass  # If error, pass file
#         filenumber += 1  # Increment image number
#
#
# for emotion in emotions:
#     detect_faces(emotion)  # Call function
#
# # Creating the training and classification set
# # The dataset has been organised and is ready to be recognized, but first we need to actually teach the classifier what certain emotions look like.
# # we randomly sample and train on 80% of the AAdata and classify the remaining 20%, and repeat the process 10 times

import cv2
import glob
import random
import numpy as np

emotions = ["neutral", "anger", "happy", "sadness", "surprise"]  # Emotion list
fishface = cv2.face.FisherFaceRecognizer_create()  # Initialize fisher face classifier

data = {}


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("/home/marta/PycharmProjects/CYBEROPS/RECONOCIMIENTO_FACIAL/BANCO_IMAGENES/IMAGENES_PROPIAS/%s/*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
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

print("\n\nend score:", np.mean(metascore), "percent correct!") #72.2% of 354-281 (73) images

# Saving and loading the trained model

# training_data, training_labels, prediction_data, prediction_labels = make_sets()
# fishface.train(training_data, np.asarray(training_labels))
# fishface.save("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/fishface_model.xml")
# fishface.load("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/fishface_model.xml")

# The dataset used in this script is the CK+ dataset, based on the work of:

# Kanade, T., Cohn, J. F., & Tian, Y. (2000).
# Comprehensive database for facial expression analysis.
# Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG00), Grenoble, France, 46-53.

# Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010).
# The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression.
# Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.

