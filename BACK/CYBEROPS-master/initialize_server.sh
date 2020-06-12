#!/bin/bash
python3 KAFKA/main_consumer_Alexa.py&
sleep 2
python3 KAFKA/main_consumer_emotion.py &
sleep 2
python3 KAFKA/main_consumer_hr.py &
sleep 2
python3 KAFKA/main_consumer_arousal.py &
sleep 2
python3 KAFKA/main_consumer_noise_level.py &
sleep 2
python3 KAFKA/main_consumer_temperature.py &
sleep 2
python3 KAFKA/main_consumer_maquinas.py &
sleep 2
#gnome-terminal --tab --title="anomalyDetector" --command="bash -c 'source ~/.virtualenvs/cyberops_py2/bin/activate;python ~/PycharmProjects/CYBEROPS/MAQUINAS/Anomaly/Anomaly_detection.py'" &
sleep 2
python3 SERVER_SENSOR_DATA/prueba_server.py &
sleep 2
python3 NewsCore/getterNoticiasServlet.py &


