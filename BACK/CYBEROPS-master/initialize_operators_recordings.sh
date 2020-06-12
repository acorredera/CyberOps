#!/bin/bash
#En el entorno py35AndcvEnv y dentro de la carpeta de CyberOps ejecutar con bash + el nombre del archivo
python3 AUDIO/audio_emotion_predictor.py &   #metido el de coger ruido ambiente también
sleep 2
python3 RECONOCIMIENTO_FACIAL/photo_emotion_landmarksShapePredictor_2.py &
sleep 2
python3 ASISTENTE/pomodoro_google_forms.py &
sleep 4
python3 OPERADOR/pop_up_messages.py &
