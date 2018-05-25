# TAKE PHOTOS FOR EACH EMOTION AND TRAIN THE CLASSIFIERS WITH THOSE PHOTOS

import cv2
from datetime import datetime
import time
import glob

camera = cv2.VideoCapture(0)
flag = 1
emotion_list = ["anger", "happy", "neutral", "sadness", "surprise"]

for emotion in emotion_list:

    print("creating %s dataset...\n" %emotion)
    time.sleep(5.0)  # time in seconds between 2 photos

    for i in range(0,11):

        if flag == 1:
            print("Please look at the webcam")

            print("Taking a photo...\n")
            return_value, image = camera.read()
            filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".png"
            # path = glob.glob("/home/marta/Own_dataset/%s/" %emotion)
            cv2.imwrite(("/home/marta/Own_dataset/%s/" %emotion) + filename, image)
            camera.release()

            flag = 0  # Activate flag to enter in "else" loop after the first photo

        else:
            print("Please look at the webcam")
            time.sleep(5.0)  # time in seconds between 2 photos

            print("Taking a photo...\n")
            camera = cv2.VideoCapture(0)
            return_value, image = camera.read()
            filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".png"
            # path = glob.glob("/home/marta/Own_dataset/%s/" % emotion)
            cv2.imwrite(("/home/marta/Own_dataset/%s/" % emotion) + filename, image)
            camera.release()

# SIZING AND TRANSFORMING PICTURES TO GRAY SCALE

faceDet = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_alt_tree.xml")

def detect_faces(emotion):
    files = glob.glob("/home/marta/Own_dataset/%s/*" % emotion)  # Get list of all images with emotion
    print(files)

    filenumber = 0
    for f in files:
        frame = cv2.imread(f)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

        # Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        # Go over detected faces, stop at first detected face, return empty if no face.
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
            print "face found in file: %s" % f
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size

            try:
                out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                cv2.imwrite("/home/marta/Own_dataset/Gray_dataset/%s/%s.jpg" %(emotion, filenumber), out)  # Write image
            except:
                pass  # If error, pass file
        filenumber += 1  # Increment image number

for emotion in emotion_list:
    detect_faces(emotion)  # Call function





