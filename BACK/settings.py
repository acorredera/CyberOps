# -*- coding: latin-1 -*-
import datetime, os

camera_mode = 0
keyspace_cassandra = "cyberops"
employee_name = "Marta"


#COMENTAR UNO U OTRO SEGuN DESDE DONDE QUERAMOS COGER LOS DATOS
#----------------START DATOS DC/OS---------------------
# ip_running_programs_server = "10.40.38.26"#"10.40.38.26"#"localhost"#"10.40.38.26"
# ip_DCOS_cassandra = '10.40.39.33' #"localhost"#
# ip_kafka_DCOS = '10.40.39.33:9092' #"localhost"#
#----------------END DATOS DC/OS-----------------------

#--------------------------------freq & para datos enlatados-----------------------------------------------
ip_running_programs_server = "10.40.38.30"#"10.40.38.26"#"localhost"#"10.40.38.26"
ip_DCOS_cassandra = 'localhost' #"localhost"#
ip_kafka_DCOS = '10.40.39.33:9092' #"localhost"#
current_time = datetime.datetime.strptime('2018-06-29 08:15:27.000', '%Y-%m-%d %H:%M:%S.%f') #sólo para pruebas en GUI o para los pop-ups que necesitan saber el estad del usuario


#initial_time_ts = time.mktime(current_time.timetuple())
#final_time_ts = time.mktime((currentTime - datetime.timedelta(minutes=5)).timetuple()) #o lo que sea minutos ago

minutes_ago_5 = 5 # return previous 5 minutes information. Method: getOperatorsRoomSensorsData, getStatusDC, getStatusRacks, getStatusServersOfRack
minutes_ago_30 = 30 #OPERADOR/pop_up_messages.py
hours_ago_24 = 24 #return previous 24 hours data. Method: getSummaryStatusEmployees, getStatusEmployee

#------------------------------------end datos enlatados---------------------------------------------------