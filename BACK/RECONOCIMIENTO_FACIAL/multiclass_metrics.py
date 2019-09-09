from sklearn.metrics import classification_report
import pandas as pd
import os

PATH = os.getcwd()

df=pd.read_csv(PATH+'/BANCO_IMAGENES/IMAGENES_PROPIAS/emotions.csv', sep=";", header='infer')

y_true = df["EMOTION"]
y_pred = df["PREDICTION"]
target_names = ["neutral", "anger", "happy", "sadness", "surprise"]

print("FISHER FACE MODEL")
print(classification_report(y_true, y_pred, target_names=target_names))

# -------------------------------------------------

df1=pd.read_csv(PATH+'/BANCO_IMAGENES/IMAGENES_PROPIAS/emotions_landmarks.csv', sep=";", header='infer')

y_true1 = df1["EMOTION"]
y_pred1 = df1["PREDICTION"]
target_names1 = ["anger", "happy", "neutral", "sadness", "surprise"]

print("LANDMARKS FACE MODEL")
print(classification_report(y_true1, y_pred1, target_names=target_names1))


