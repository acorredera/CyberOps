#!/usr/bin/env bash

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
sleep 10
bash initialize_operators_recordings.sh &

