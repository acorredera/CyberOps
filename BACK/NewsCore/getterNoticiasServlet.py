import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import NewsCore.dao.NewsDAOImpl as DAO
import NewsCore.dao.OperatorsRoomStatusDAOImpl as DAO_Operator
import NewsCore.dao.AlexaAnswerDAOImpl as AlexaDao
import NewsCore.model.AlexaAnswer as profileQuestions
import NewsCore.dao.EmployeeDAOImpl as employeeStatusDao
import NewsCore.dao.ServersStatusDAOImpl as serverStatusDAO
import NewsCore.dao.MachinesDAOImplementation as machinesDAO
import NewsCore.dao.AnomaliesInMachinesDAOImpl as anomaliesDAO
from flask import Flask, redirect, url_for
from flask import request
from flask_cors import CORS
from flask_jsonpify import jsonify
from flask_restful import Api
from flask import render_template
import numpy as np
from kafka import KafkaProducer
from dateutil import parser
import MAQUINAS.processor.status_and_anomalies_detection as statusAndAnomDetect
import json
import datetime
import settings
import MAQUINAS.processor.status_and_anomalies_detection as status_machines_processor


app = Flask(__name__) #http://flask.pocoo.org/docs/1.0/api/
api = Api(app)
CORS(app)

#---DB variables
ip_cassandra = settings.ip_DCOS_cassandra
keyspace = settings.keyspace_cassandra
ip_kafka = settings.ip_kafka_DCOS



#---------------

#--------GET NEWS------------
@app.route("/")
def doGetNews():
    print("hola")
    # cambiar todo esto para que sea cargar database y que sea persistente
    # databaseNoticias =DAO.NewsDAOImpl(keyspace)
    # databaseNoticias.createsession()
    # databaseNoticias.setlogger()
    # rows = databaseNoticias.select_data(1)
    # news = []
    # for row in rows:
    #     #print(row.metrics, row.title, row.summary)
    #     news.append({"titular":row.title, "resumen":row.summary})
    # return jsonify({'Noticias': news})#jsonify(news)#{'Noticias': {'id':1, 'titular':'hola caracola','resumen':'blabla'}

#http://flask.pocoo.org/docs/0.12/quickstart/#variable-rules
#ejemplo de llamada desde ip_cassandra: http://127.0.0.1:5002/getNoticias/noticias?keywords=metricB,metricA,metricC  ....
@app.route('/getNoticias/<data>')#noticias?keywords=<any:keyword_list>
def doGetNewsBykeywords(data):#keyword_list
    keyword_list = request.args.get('keywords', '').split(",")
    databaseNoticias = DAO.NewsDAOImpl(keyspace)
    databaseNoticias.createsession()
    databaseNoticias.setlogger()
    print(keyword_list)
    rows = databaseNoticias.selectNews(keyword_list)
    news = []
    for row in rows:
        news.append({"titular":row.title, "resumen":row.summary})
        #print(row.metric, row.title, row.summary)
    return jsonify(news)
#--------END GET NEWS------------


# @app.route('/directory')
# def getDir():
#     dirpath = os.getcwd()
#     print(dirpath)
#     return dirpath

#llamada: http://localhost:3000/getErrors
@app.route('/getErrors')
def getErrors():
    return jsonify("1")

#-------------------OPERATORS ROOM SENSORS-------------------
#llamada:   http://localhost:3000/getOperatorsRoomSensorsData
#return:   [{"heat_index": 0.0, "humidity": 0.0, "noise": 0.0, "temperature": 0.0}]
@app.route('/getOperatorsRoomSensorsData')
def allData():
    databaseallData = DAO_Operator.OperatorsRoomStatusDAOImpl()
    databaseallData.createsession(ip_cassandra)
    databaseallData.setlogger()
    databaseallData.loadkeyspace(keyspace)

    currentTime = settings.current_time#datetime.datetime.now()
    timestamp_rounded = currentTime - datetime.timedelta(seconds=currentTime.second, microseconds=currentTime.microsecond)

    someMinutesAgo = timestamp_rounded - datetime.timedelta(minutes=settings.minutes_ago_5)
    allData = databaseallData.select_allData_inRange(end_timestamp=timestamp_rounded, initial_timestamp=someMinutesAgo ,room_id="1")
    return jsonify(allData)
#----------------------------------------------------------

#-------------------STATUS EMPLOYEE/S--------------------------------------
#llamada:  http://ip_cassandra:3000/getSummaryStatusEmployees
#example return: [{"color": "red", "name": "Marta", "profile": 0}, {"color": "red", "name": "cris", "profile": 0}, {"color": "red", "name": "juan", "profile": 0}]
@app.route('/getSummaryStatusEmployees')
def getSummaryStatusEmployees():
    #create session and load DB
    daoAlexaAnswers = AlexaDao.AlexaAnswerDAOImpl()
    daoAlexaAnswers.createsession(ip_cassandra)
    daoAlexaAnswers.setlogger()
    daoAlexaAnswers.loadkeyspace(keyspace)
    currentTime = settings.current_time#datetime.datetime.now()

    someMinutesAgo = currentTime - datetime.timedelta(hours=settings.hours_ago_24)
    list_employees_alias = daoAlexaAnswers.select_users()
    if(not list_employees_alias):
        return {}
    else:
        stateEmployeesSomeMinutesAgo = daoAlexaAnswers.select_avg_profiles_by_date_allUsers(list_employees_alias, someMinutesAgo, currentTime)
    return jsonify(stateEmployeesSomeMinutesAgo)

#Se podría pasar el date también si eso.
#llamada: http://ip_cassandra:3000/getStatusEmployee/stateEmployee?alias=juan
#example return: {"Coordination": 0, "Environment": 0, "Mood": 0, "Performance": {"color": "red", "punctuation": 0}, "Work": 0}
@app.route('/getStatusEmployee/<data>')
def getStatusEmployee(data): #poner corchetes en el return[]
    #create session and load DB
    employee_alias = request.args.get('alias', '').split(",")[0]
    daoAlexaAnswers = AlexaDao.AlexaAnswerDAOImpl()
    daoAlexaAnswers.createsession(ip_cassandra)
    daoAlexaAnswers.setlogger()
    daoAlexaAnswers.loadkeyspace(keyspace)
    currentTime = settings.current_time#datetime.datetime.now()
    someMinutesAgo = currentTime - datetime.timedelta(hours=settings.hours_ago_24)
    profile = daoAlexaAnswers.select_avg_profiles_by_alias_date(employee_alias, someMinutesAgo, currentTime)
    profile_employee = {
        "Mood":0,
        "Work":0,
        "Coordination":0,
        "Environment":0,
        "Performance":{"punctuation":0, "color":"red"}
    }
    for row in profile:
        profileAnswer = profileQuestions.QuestionProfile(row.profile_question).name
        profile_employee[profileAnswer]=row.system_avg_punctuation
    punctuation_avg, color = daoAlexaAnswers.select_avg_and_color_by_alias_date(employee_alias, someMinutesAgo, currentTime)
    profile_employee["Performance"]["color"]=color
    profile_employee["Performance"]["punctuation"] = punctuation_avg
    return jsonify([profile_employee])

#EXAMPLE HOW TO GENERATE TIMESTAMPS IN PYTHON:
#   date = datetime.datetime.now()
#   ts = time.mktime(date.timetuple())
#   convert to date again:   date = datetime.datetime.fromtimestamp(ts)
#llamada: http://ip_cassandra:3000/getHREmployee/HREmployee?alias=Cris&initial_time=1550059655.926982&final_time=1550059915.413911
#return: [] ó ....(probar)
@app.route('/getHREmployee/<data>')
def getHREmployee(data):
    employee_alias = request.args.get('alias', '').split(",")[0]
    #num_samples = int(request.args.get('num_samples', '').split(",")[0] or '50')
    initial_time = datetime.datetime.fromtimestamp(float(request.args.get('initial_time', '').split(",")[0]))
    final_time = datetime.datetime.fromtimestamp(float(request.args.get('final_time', '').split(",")[0]))
    daoEmployeeStatus = employeeStatusDao.EmployeeDAOImpl()
    daoEmployeeStatus.createsession(ip_cassandra)
    daoEmployeeStatus.setlogger()
    daoEmployeeStatus.loadkeyspace(keyspace)
    hr_list = daoEmployeeStatus.select_some_hr_inRange(employee_alias, initial_time, final_time, -1)
    hr_2_send = []
    for row in hr_list:
        hr_2_send.append({"date":row[0], "hr":row[1]})
    return jsonify(hr_2_send)


#llamada: http://ip_server:3000/getEmotionEmployee/EmotionEmployee?alias=Cris&initial_time=1550228820.0&final_time=1550235840.0
#return: [] ó ....(probar)
@app.route('/getEmotionEmployee/<data>')
def getEmotionEmpployee(data):
    employee_alias = request.args.get('alias', '').split(",")[0]
    #num_samples = int(request.args.get('num_samples', '').split(",")[0] or '50')
    initial_time = datetime.datetime.fromtimestamp(float(request.args.get('initial_time', '').split(",")[0]))
    final_time = datetime.datetime.fromtimestamp(float(request.args.get('final_time', '').split(",")[0]))
    daoEmployeeStatus = employeeStatusDao.EmployeeDAOImpl()
    daoEmployeeStatus.createsession(ip_cassandra)
    daoEmployeeStatus.setlogger()
    daoEmployeeStatus.loadkeyspace(keyspace)
    emotion_list = daoEmployeeStatus.select_some_emotion_inRange(employee_alias, initial_time, final_time, -1)
    emotion_2_send = []
    for row in emotion_list:
        emotion_2_send.append({"date":row[0], "emotion":row[1]})
    return jsonify(emotion_2_send)

#llamada: http://ip_server:3000/getArousalEmployee/ArousalEmployee?alias=Cris&initial_time=1550228820.0&final_time=1550235840.0
#return: [] o ....(probar)
@app.route('/getArousalEmployee/<data>')
def getArousalEmpployee(data):
    employee_alias = request.args.get('alias', '').split(",")[0]
    #num_samples = int(request.args.get('num_samples', '').split(",")[0] or '50')
    initial_time = datetime.datetime.fromtimestamp(float(request.args.get('initial_time', '').split(",")[0]))
    final_time = datetime.datetime.fromtimestamp(float(request.args.get('final_time', '').split(",")[0]))
    daoEmployeeStatus = employeeStatusDao.EmployeeDAOImpl()
    daoEmployeeStatus.createsession(ip_cassandra)
    daoEmployeeStatus.setlogger()
    daoEmployeeStatus.loadkeyspace(keyspace)
    parameter = "arousal"
    parameter_list = daoEmployeeStatus.select_some_parameter_inRange(employee_alias, initial_time, final_time, -1, parameter)
    parameter_2_send = []
    for row in parameter_list:
        parameter_2_send.append({"date":row[0], parameter:row[1]})
    return jsonify(parameter_2_send)

##--------------------END STATUS EMPLOYEE/S-----------------------------------

#---------------------SERVERS/RACK STATUS-----------------------------------------------

#llamada: http://ip_cassandra:3000/getStatusDC
@app.route('/getStatusDC')
def getStatusDC():
    daoServersStatus = machinesDAO.MachinesDAOImplementation()
    # daoServersStatus.create_table()
    daoServersStatus.create_session(ip_cassandra)
    daoServersStatus.set_logger()
    daoServersStatus.load_keyspace(keyspace)
    currentTime = settings.current_time#datetime.datetime.now()
    someMinutesAgo = currentTime - datetime.timedelta(minutes=settings.minutes_ago_5)

    #Room ids & rack ids:
    dict_rooms_racks = daoServersStatus.select_all_roomAndRacksIds()
    if (not dict_rooms_racks):
        return jsonify([])
    else:
        status_DC = 0
        num_racks = 0
        for room_id in dict_rooms_racks:
            set_racks_in_room = dict_rooms_racks[room_id]
            for rack_id in set_racks_in_room:
                # previous variables
                status_rack = 100
                set_servers = daoServersStatus.select_server_ids_by_roomAndRackid(room_id, rack_id)
                for server_id in set_servers:
                    dict_status, color, _, _ = status_machines_processor.get_status_servers(
                        daoServersStatus, None, room_id, rack_id, server_id, someMinutesAgo,
                        currentTime, metrics=["temperature", "energy", "utilization"])
                    if (status_rack > dict_status["total"]): #we keep the worst server status as the rack status
                        status_rack = dict_status["total"]
                #End of processing of servers -> We keep the value of the DC
                status_DC += status_rack
                num_racks += 1
        avg_DC = status_DC/num_racks #avg of the worst server of each rack
        color_DC = statusAndAnomDetect.color_status(avg_DC)
        return jsonify({"DC_status": avg_DC, "color":color_DC})


#llamada: http://ip_cassandra:3000/getStatusRacks
@app.route('/getStatusRacks')
def getStatusRacks():
    daoServersStatus = machinesDAO.MachinesDAOImplementation()
    #daoServersStatus.create_table()
    daoServersStatus.create_session(ip_cassandra)
    daoServersStatus.set_logger()
    daoServersStatus.load_keyspace(keyspace)
    #DAO anomalies
    daoServersAnomalies = anomaliesDAO.AnomaliesInMachinesDAOImpl()
    # daoServersAnomalies.create_table()
    daoServersAnomalies.create_session(ip_cassandra)
    daoServersAnomalies.set_logger()
    daoServersAnomalies.load_keyspace(keyspace)

    currentTime = settings.current_time#datetime.datetime.now()
    someMinutesAgo = currentTime - datetime.timedelta(minutes=settings.minutes_ago_5)#minutes=5
    dict_rooms_racks = daoServersStatus.select_all_roomAndRacksIds()
    list_status_racks = []
    profile_room = []

    if(not dict_rooms_racks):
        return jsonify([])
    else:
        list_of_rooms = []
        for room_id in dict_rooms_racks:
            #room_id_dict = {"room_id": room_id, "rack_status": {}}
            set_racks_in_room = dict_rooms_racks[room_id]
            #get servers in each rack
            list_of_racks_by_room = []
            for rack_id in set_racks_in_room:
                #previous variables
                status_rack = 100
                color_rack = "red"
                rack_anomaly_type = "None"
                rack_anomaly_value = False
                set_servers = daoServersStatus.select_server_ids_by_roomAndRackid(room_id, rack_id)
                for server_id in set_servers:
                    dict_status, color, anomaly_type, anomaly_value = status_machines_processor.get_status_servers(daoServersStatus, daoServersAnomalies, room_id, rack_id, server_id, someMinutesAgo, currentTime,metrics=["temperature", "energy", "utilization"])
                    if(status_rack > dict_status["total"]):
                        status_rack = dict_status["total"]
                        color_rack = color
                    if(anomaly_value):
                        rack_anomaly_type = anomaly_type #Para el rack no tiene tanto sentido porque cada server podría tener una diferente
                        rack_anomaly_value = anomaly_value
                list_of_racks_by_room.append({"rack_id": rack_id, "status": status_rack, "color": color_rack, "anomaly_type":rack_anomaly_type, "anomaly_value":rack_anomaly_value})
            list_of_rooms.append({"room_id":room_id, "rack_status":list_of_racks_by_room})
            #room_id_dict[room_id]["rack_status"].append({"rack_id": rack_id, "status": status_rack, "color": color_rack})
            #value = profile_room["room_status"]
            #value
            #list_servers = daoServersStatus.select_server_ids_by_rack_id(rack_id)
            #status_rack = daoServersStatus.select_statusRack_by_rack_servers_date_profile(rack_id,
            #list_servers, someMinutesAgo, currentTime,profile=[1, 2, 3, 4, 5])
        return jsonify(list_of_rooms)

#llamada: http://ip_cassandra:3000/getStatusServersOfRack/rack?rackId=1&roomId=1
@app.route('/getStatusServersOfRack/<data>')
def getStatusServers(data):
    rack_id = int(request.args.get('rackId', '').split(",")[0])
    room_id = int(request.args.get('roomId', '').split(",")[0])
    daoServersStatus = machinesDAO.MachinesDAOImplementation()
    #daoServersStatus = serverStatusDAO.ServersStatusDAOImpl()
    daoServersStatus.create_session(ip_cassandra)
    daoServersStatus.set_logger()
    daoServersStatus.load_keyspace(keyspace)
    # DAO anomalies
    daoServersAnomalies = anomaliesDAO.AnomaliesInMachinesDAOImpl()
    # daoServersAnomalies.create_table()
    daoServersAnomalies.create_session(ip_cassandra)
    daoServersAnomalies.set_logger()
    daoServersAnomalies.load_keyspace(keyspace)

    currentTime = settings.current_time#datetime.datetime.now()
    someMinutesAgo = currentTime - datetime.timedelta(minutes = settings.minutes_ago_5)#minutes=5
    set_servers = daoServersStatus.select_server_ids_by_roomAndRackid(room_id, rack_id)
    if(not set_servers):
        return jsonify([])
    else:
        list_servers = []
        metrics = [
            "temperature",
            "energy",
            "utilization"]
        avg_status_rack = {"temperature": 0, "energy":0, "utilization":0}
        number_of_servers_in_rack = len(set_servers)
        for server_id in set_servers:
            dict_status, color, anomaly_type, anomaly_value = status_machines_processor.get_status_servers(
                daoServersStatus,
                daoServersAnomalies,
                room_id, rack_id,
                server_id,
                someMinutesAgo,
                currentTime,
                metrics=metrics)
            for metric in metrics:
                avg_status_rack[metric] = avg_status_rack[metric]+(dict_status[metric]/number_of_servers_in_rack)
            list_servers.append({"server_id": server_id,
                                 "server_status":{"status":dict_status["total"], "color":color},
                                 "anomaly": {"type": anomaly_type, "value": anomaly_value}})
        rack_information = {"rack_id":rack_id,
                            "server_status": list_servers,
                            "avg_performance":avg_status_rack}
        return jsonify(rack_information) #fields: server_id, status, color

#http://ip_cassandra:3000/getRackEvolution/rack?rackId=1&roomId=1&initial_time=1554878600.0&final_time=1557758600.0
@app.route('/getRackEvolution/<data>') #OJO -> Sólo se puede crear 1 spark context por petición, si se piden más, se cae
def getStatusRackInTime(data):
    rack_id = int(request.args.get('rackId', '').split(",")[0])
    room_id = int(request.args.get('roomId', '').split(",")[0])
    # num_samples = int(request.args.get('num_samples', '').split(",")[0] or '50')
    initial_time = datetime.datetime.fromtimestamp(float(request.args.get('initial_time', '').split(",")[0]))
    final_time = datetime.datetime.fromtimestamp(float(request.args.get('final_time', '').split(",")[0]))
    daoServersStatus = machinesDAO.MachinesDAOImplementation()
    # daoServersStatus = serverStatusDAO.ServersStatusDAOImpl()
    daoServersStatus.create_session(ip_cassandra)
    daoServersStatus.set_logger()
    daoServersStatus.load_keyspace(keyspace)
    daoServersStatus.create_spark_context(ip_cassandra)
    daoServersStatus.load_spark_table()
    metric = "temperature"
    set_servers = daoServersStatus.select_server_ids_by_roomAndRackid(room_id, rack_id)
    results = daoServersStatus.select_MAXmetric_and_date(room_id,rack_id, list(set_servers), initial_time, final_time, metric_type=metric) #select_metricAndDate_by_roomAndRackAndServerIds_andComponent
    metric_2_send = []
    if (not results):
        return jsonify([])
    else:
        for row in results.rdd.collect():
            metric_2_send.append({"date": row["new_date"], metric:row["max(metric_value)"]})
        return jsonify(metric_2_send)
# llamada: http://ip_cassandra:3000/getStatusSingleServer/server?rack_id=1&server_id=groot
# @app.route('/getStatusSingleServer/<data>')
# def getStatusSingleServer(data):
#     #ver si eva nos puede guardar el estado del servidor (media de los profiles) para evitar hacer otra llamada a la base de datos
#     rack_id = request.args.get('rack_id', '').split(",")[0]
#     server_id = request.args.get('server_id', '').split(",")[0]
#     daoServersStatus = serverStatusDAO.ServersStatusDAOImpl()
#     daoServersStatus.createsession(ip_cassandra)
#     daoServersStatus.setlogger()
#     daoServersStatus.loadkeyspace(keyspace)
#     currentTime = datetime.datetime.now()
#     someMinutesAgo = currentTime - datetime.timedelta(hours=24)#minutes=5
#     status_server = daoServersStatus.select_statusServer_by_rack_id_server_date(rack_id=rack_id,server_id=server_id,
#                                                                                 initial_timestamp=someMinutesAgo, end_timestamp=currentTime)
#     return jsonify(status_server) #fields: profile,status


#-------------END SERVER/RACKS STATUS------------------------------------------



# @app.route('/index', methods=['GET', 'POST'])
# def lionel():
#     return render_template('index.html')




#------------PEBBLE APP POST--------------
@app.route("/post/", methods=['POST'])
def doPOST_HR3():
    topic_hr = 'cyberops_HR'
    if request.method == 'POST':
        data = request.json['data']
        metricsObtained = data.split(',')
        timestamp = metricsObtained[0]
        HR = metricsObtained[1]
        employee = metricsObtained[-1]
        #kafka producer that sends data to consumer
        producer = KafkaProducer(bootstrap_servers=ip_kafka) #, api_version=(0, 10, 1)
        data2send = {"timestamp": timestamp,
             "hr": HR,
             "employeeAlias": employee}
        producer.send(topic_hr, json.dumps(data2send).encode('utf-8'))
        producer.close()
    return ""
#------------ END PEBBLE APP POST--------------




if __name__ == '__main__':
   app.run(host='0.0.0.0',port=3000)