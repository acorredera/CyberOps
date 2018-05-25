# TAKE PHOTOS EACH 1 MINUTE- PYTHON FILE

import cv2
import time
from datetime import datetime
import os
import glob
import csv

# LOADING AND DEFINING INITIAL VARIABLES

camera = cv2.VideoCapture(0)
flag=1

faceDet = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/OpenCV_FaceCascade/haarcascade_frontalface_alt_tree.xml")

# DEFINING LOOP TO TAKE AND ANALYZE PHOTOS

while(True):

    if flag==1:

        return_value, image = camera.read()
        filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".png"
        cv2.imwrite('/home/marta/Webcam/' + filename, image)
        camera.release()
        flag=0 # Activate flag to enter in "else" loop after the first photo

        path_new_file = glob.glob("/home/marta/Webcam/" + filename)
        path, dirs, files = os.walk("/home/marta/Webcam/").next()

        # Modify image for classification
        files = glob.glob("/home/marta/Webcam/*.png")
        for picture in files:
            # print(picture)
            if picture == path_new_file[0]:
                frame = cv2.imread(picture)  # Open image
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
                    print "face found in file: %s" % picture
                    gray = gray[y:y + h, x:x + w]  # Cut the frame to size
                    try:
                        out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                        cv2.imwrite("/home/marta/Webcam/dataset/%s" % (picture[19:]), out)  # Write image
                    except:
                        pass  # If error, pass file

        # Pass classifier
        gray_files = glob.glob("/home/marta/Webcam/dataset/*.png")
        for gray_picture in gray_files:
            if gray_picture == "/home/marta/Webcam/dataset/%s" %filename:
                fishface = cv2.createFisherFaceRecognizer()  # Initialize fisher face classifier
                fishface.load("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/own_fisher_face_model.xml") # LOADING THE FISHER_RECOG ALREADY TRAINED AND SAVED (see face_emotion_recognition_classifier.py)

                pred, conf = fishface.predict(out)
                # emotions_trained_model=["neutral", "anger", "happy", "sadness", "surprise"]
                if pred == 0:
                    emotion = "anger"
                    print(emotion)
                elif pred == 1:
                    emotion = "happy"
                    print(emotion)
                elif pred == 2:
                    emotion = "neutral"
                    print(emotion)
                elif pred == 3:
                    emotion = "sadness"
                    print(emotion)
                elif pred == 4:
                    emotion = "surprise"
                    print(emotion)

                # write timestamp and emotion in csv file
                with open('/home/marta/Webcam/results.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';', lineterminator='\n')
                    writer.writerow([filename[0:19], emotion])

        # Delete files when they exceed a number
        file_count = len(files)
        file_count_gray = len(gray_files)

        tupla=[]
        tupla_gray=[]

        if file_count > 3: # Defining the maximum number of saved photos (3)
            for picture in files:
                date = os.path.getctime(picture)
                array=[date, picture]
                tupla.append(array)
            older_file = sorted(tupla)
            deletepath=older_file[0][1]
            os.remove(deletepath)

        if file_count_gray > 3: # Defining the maximum number of saved photos (3)
            for picture in gray_files:
                date = os.path.getctime(picture)
                array_gray = [date, picture]
                tupla_gray.append(array_gray)
            older_gray_file = sorted(tupla_gray)
            deletepath_gray = older_gray_file[0][1]
            os.remove(deletepath_gray)

    else:

        time.sleep(15.0) #time in seconds between 2 photos
        camera = cv2.VideoCapture(0)
        return_value, image = camera.read()
        filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".png"
        cv2.imwrite('/home/marta/Webcam/' + filename, image)
        camera.release()
        path_new_file = glob.glob("/home/marta/Webcam/" + filename)
        path, dirs, files = os.walk("/home/marta/Webcam/").next()

        # Modify image for classification
        files = glob.glob("/home/marta/Webcam/*.png")
        for picture in files:
            if picture == path_new_file[0]:
                frame = cv2.imread(picture)  # Open image
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
                    print "face found in file: %s" % picture
                    gray = gray[y:y + h, x:x + w]  # Cut the frame to size
                    try:
                        out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                        cv2.imwrite("/home/marta/Webcam/dataset/%s" % (picture[19:]), out)  # Write image
                    except:
                        pass  # If error, pass file

        # Pass classifier
        gray_files = glob.glob("/home/marta/Webcam/dataset/*.png")
        for gray_picture in gray_files:
            if gray_picture == "/home/marta/Webcam/dataset/%s" % filename:
                fishface = cv2.createFisherFaceRecognizer()  # Initialize fisher face classifier
                fishface.load("/home/marta/PycharmProjects/untitled/pyAudioAnalysis/own_fisher_face_model.xml")

                pred, conf = fishface.predict(out)
                # emotions_trained_model=["neutral", "anger", "happy", "sadness", "surprise"]
                if pred == 0:
                    emotion = "anger"
                    print(emotion)
                elif pred == 1:
                    emotion = "happy"
                    print(emotion)
                elif pred == 2:
                    emotion = "neutral"
                    print(emotion)
                elif pred == 3:
                    emotion = "sadness"
                    print(emotion)
                elif pred == 4:
                    emotion = "surprise"
                    print(emotion)

                # write timestamp and emotion in csv file
                with open('/home/marta/Webcam/results.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';', lineterminator='\n')
                    writer.writerow([filename[0:19], emotion])

        # Delete files when they exceed a number
        file_count = len(files)
        file_count_gray = len(gray_files)

        tupla=[]
        tupla_gray=[]

        if file_count > 3: # Defining the maximum number of saved photos (3)
            for picture in files:
                date = os.path.getctime(picture)
                array=[date, picture]
                tupla.append(array)
            older_file = sorted(tupla)
            deletepath=older_file[0][1]
            os.remove(deletepath)

        if file_count_gray > 3: # Defining the maximum number of saved photos (3)
            for picture in gray_files:
                date = os.path.getctime(picture)
                array_gray = [date, picture]
                tupla_gray.append(array_gray)
            older_gray_file = sorted(tupla_gray)
            deletepath_gray = older_gray_file[0][1]
            os.remove(deletepath_gray)

del(camera)

