#!/usr/bin/env bash
#******************************IMPORTANTE***********************************************
#* Si cambia la IP de mi ordena (10.40.38.30 en el labo), cambiar la IP en:
#*     -Código del sensor
#*     -Settings.py
#*     -La configuración de la app de peeble
#****************************************************************************************



#INITIALIZE LOCAL SERVICES
source ~/.virtualenvs/cyberops/bin/activate

#1. meterme en el gmail de marta para que la consola envie bien los datos y demás (pomodoro_google_forms)
#2.
#start kafka, cassandra, zookeeper
systemctl start cassandra
systemctl start zookeeper


gnome-terminal --tab --title="kafkaService" --command="bash -c 'cd ~/confluent-oss-5.0.1-2.11/confluent-5.0.1;bin/kafka-server-start etc/kafka/server.properties'" &


sleep 30 #lo que tarda en levantarse kafka
#systemctl start collectd #config de collectd en mi ordena: /home/cris/Documentos/collected_plugins/parser.py &&  /etc/collectd/collectd.conf


source ~/.virtualenvs/cyberops/bin/activate
sleep 10
#INITIALIZE CYBEROPS
bash initialize_server.sh &

#INITIALIZE GUI:
sleep 30
gnome-terminal --tab --title="GUI" --command="bash -c 'cd ~/PycharmProjects/CYBEROPS_EVA_GUI/CyberOpsOperator-master;npm config set prefix ~/.npm-global;export PATH=~/.npm-global/bin:$PATH;source ~/.profile;cd cyberopsoperator/;ionic serve'" &

sleep 40
bash initialize_operators_recordings.sh &