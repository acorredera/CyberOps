from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os

PATH = os.getcwd()

vokaturi_predicted_list=[]
deep_affect_predicted_list=[]
open_smile_predicted_list=[]

vokaturi_real_list=[]
deep_affect_real_list=[]
open_smile_real_list=[]

def merged(empty_list,list):
    for i in list:
        empty_list+=i
    return empty_list

# VOKATURI
emotion_vokaturi={"angry":[0], "fear":[1], "happiness":[2], "neutral":[3], "sadness":[4]}

# Confussion Matrix
angry_v = 115*emotion_vokaturi["angry"],10*emotion_vokaturi["happiness"],2*emotion_vokaturi["neutral"]
fear_v = 1*emotion_vokaturi["angry"], 61*emotion_vokaturi["fear"], 4*emotion_vokaturi["happiness"], 2*emotion_vokaturi["neutral"], 1*emotion_vokaturi["sadness"]
happiness_v = 7*emotion_vokaturi["angry"], 3*emotion_vokaturi["fear"], 61*emotion_vokaturi["happiness"]
neutral_v = 3*emotion_vokaturi["fear"], 4*emotion_vokaturi["happiness"], 68*emotion_vokaturi["neutral"], 4*emotion_vokaturi["sadness"]
sadness_v = 1*emotion_vokaturi["fear"], 2*emotion_vokaturi["neutral"], 59*emotion_vokaturi["sadness"]

emotion_predicted_vokaturi = angry_v + fear_v + happiness_v + neutral_v + sadness_v

# Real data
real_emotion_vokaturi = 127* emotion_vokaturi["angry"], 69*emotion_vokaturi["fear"], 71*emotion_vokaturi["happiness"], 79*emotion_vokaturi["neutral"], 62*emotion_vokaturi["sadness"]

true_vokaturi=merged(vokaturi_real_list,real_emotion_vokaturi)
prediction_vokaturi = merged(vokaturi_predicted_list,emotion_predicted_vokaturi)

y_true = true_vokaturi
y_pred = prediction_vokaturi
target_names = ["anger", "fear", "happiness", "neutral", "sadness"]

print("VOKATURI API")
print(classification_report(y_true, y_pred, target_names=target_names))

# DEEP-AFFECT
emotion_deep_affect={"angry":[0], "disgust":[1], "fear":[2], "happiness":[3], "neutral":[4], "pleasant":[5], "sadness":[6]}

# Confussion Matrix
angry_d = 28*emotion_deep_affect["disgust"],22*emotion_deep_affect["fear"],45*emotion_deep_affect["happiness"], 1*emotion_deep_affect["neutral"], 31*emotion_deep_affect["pleasant"]
disgust_d = 24*emotion_deep_affect["disgust"], 4*emotion_deep_affect["fear"], 2*emotion_deep_affect["happiness"], 4*emotion_deep_affect["pleasant"], 12*emotion_deep_affect["sadness"]
fear_d = 24*emotion_deep_affect["disgust"], 16*emotion_deep_affect["fear"], 12*emotion_deep_affect["happiness"], 1*emotion_deep_affect["neutral"], 8*emotion_deep_affect["pleasant"], 8*emotion_deep_affect["sadness"]
happiness_d = 17*emotion_deep_affect["disgust"], 16*emotion_deep_affect["fear"], 20*emotion_deep_affect["happiness"], 15*emotion_deep_affect["pleasant"], 3*emotion_deep_affect["sadness"]
neutral_d = 61*emotion_deep_affect["disgust"], 11*emotion_deep_affect["fear"], 3*emotion_deep_affect["happiness"], 1*emotion_deep_affect["neutral"], 3*emotion_deep_affect["sadness"]
sadness_d = 50*emotion_deep_affect["disgust"], 12*emotion_deep_affect["sadness"]

emotion_predicted_deep = angry_d + disgust_d + fear_d + happiness_d + neutral_d + sadness_d

# Real data
real_emotion_deep = 127* emotion_deep_affect["angry"], 46*emotion_deep_affect["disgust"], 69*emotion_deep_affect["fear"], 71*emotion_deep_affect["happiness"], 79*emotion_deep_affect["neutral"], 62*emotion_deep_affect["sadness"]

true_deep=merged(deep_affect_real_list,real_emotion_deep)
prediction_deep = merged(deep_affect_predicted_list,emotion_predicted_deep)

y_true = true_deep
y_pred = prediction_deep
target_names = ["anger", "disgust", "fear", "happiness", "neutral", "pleasant", "sadness"]

print("DEEP AFFECT API")
print(classification_report(y_true, y_pred, target_names=target_names))


# OPEN-SMILE
emotion_open_smile={"angry":[0], "boredom":[1], "disgust":[2], "fear":[3], "happiness":[4], "neutral":[5], "sadness":[6]}

# Confussion Matrix
angry_os = 103*emotion_open_smile["angry"], 4*emotion_open_smile["boredom"], 2*emotion_open_smile["disgust"], 7*emotion_open_smile["fear"], 7*emotion_open_smile["happiness"], 4*emotion_open_smile["neutral"]
boredom_os = 71*emotion_open_smile["boredom"], 2*emotion_open_smile["fear"], 1*emotion_open_smile["neutral"],7*emotion_open_smile["sadness"]
disgust_os = 7*emotion_open_smile["boredom"], 30*emotion_open_smile["disgust"], 4*emotion_open_smile["fear"], 4*emotion_open_smile["neutral"], 1*emotion_open_smile["sadness"]
fear_os = 5*emotion_open_smile["angry"], 8*emotion_open_smile["boredom"], 39*emotion_open_smile["fear"], 9*emotion_open_smile["happiness"], 6*emotion_open_smile["neutral"], 2*emotion_open_smile["sadness"]
happiness_os = 12*emotion_open_smile["angry"], 2*emotion_open_smile["boredom"], 7*emotion_open_smile["fear"], 45*emotion_open_smile["happiness"], 5*emotion_open_smile["neutral"]
neutral_os = 42*emotion_open_smile["boredom"], 2*emotion_open_smile["disgust"], 6*emotion_open_smile["fear"], 2*emotion_open_smile["happiness"], 25*emotion_open_smile["neutral"], 2*emotion_open_smile["sadness"]
sadness_os = 21*emotion_open_smile["boredom"], 7*emotion_open_smile["fear"], 3*emotion_open_smile["neutral"], 31*emotion_open_smile["sadness"]

emotion_predicted_os = angry_os + boredom_os + disgust_os + fear_os + happiness_os + neutral_os + sadness_os

# Real data
real_emotion_os = 127* emotion_open_smile["angry"], 81*emotion_open_smile["boredom"], 46*emotion_open_smile["disgust"], 69*emotion_open_smile["fear"], 71*emotion_open_smile["happiness"], 79*emotion_open_smile["neutral"], 62*emotion_open_smile["sadness"]

true_os=merged(open_smile_real_list,real_emotion_os)
prediction_os = merged(open_smile_predicted_list, emotion_predicted_os)

y_true = true_os
y_pred = prediction_os
target_names = ["anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness"]

print("OPEN SMILE API")
print(classification_report(y_true, y_pred, target_names=target_names))