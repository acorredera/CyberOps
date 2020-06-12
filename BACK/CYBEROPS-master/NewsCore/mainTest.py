import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import NewsCore.model.News as newsModel
from cassandra import ReadTimeout
# import NewsCore.dao.AlexaQuestionDAOImpl as alexaDAO
import NewsCore.dao.OperatorsRoomStatusDAOImpl as daoStatusRoom
import NewsCore.dao.EmployeeDAOImpl as daoEmployee
import NewsCore.dao.AnomaliesInMachinesDAOImpl as daoAnomaly
import NewsCore.dao.MachinesDAOImplementation as machiniesDAO
import datetime
import time
from dateutil import tz


if __name__ == "__main__":
    ip_dcos = "10.40.39.33"
    keyspace = "cyberops"
    #
    # #CREATION OF TABLES
    # #create employeeStatedb
    # daoStatus = daoEmployee.EmployeeDAOImpl()
    # daoStatus.createsession(ip_dcos)
    # daoStatus.setlogger()
    # daoStatus.loadkeyspace(keyspace)
    # daoStatus.create_table()
    #
    # #create alexaAnswerdb
    # daoAlexaQuestions = alexaDAO.AlexaQuestionDAOImpl()
    # daoAlexaQuestions.createsession(ip_dcos)
    # daoAlexaQuestions.setlogger()
    # daoAlexaQuestions.loadkeyspace(keyspace)
    # daoAlexaQuestions.create_table()
    #
    # # create employeeStatedb
    # daoOperatorsRoom = daoStatusRoom.OperatorsRoomStatusDAOImpl()
    # daoOperatorsRoom.createsession(ip_dcos)
    # daoOperatorsRoom.setlogger()
    # daoOperatorsRoom.loadkeyspace(keyspace)
    # daoOperatorsRoom.create_table()






    # daoAlexaQuestions = alexaDAO.AlexaQuestionDAOImpl()
    # daoAlexaQuestions.createsession("10.40.39.33")
    # daoAlexaQuestions.setlogger()
    # daoAlexaQuestions.loadkeyspace('prueba')
    # daoAlexaQuestions.create_table()
    #
    # q2 = alexaQuestion.AlexaQuestion(2,'Como andas?', alexaQuestion.QuestionProfile.Mood)
    # daoAlexaQuestions.insert_question(q2)
    # #daoAlexaQuestions.update_question_text(1,'holaa')
    # #daoAlexaQuestions.delete_table()
    # #daoAlexaQuestions.delete_keyspace('prueba')
    #
    # print(daoAlexaQuestions.select_question(1))
    #
    # daoStatus = dao.EmployeeDAOImpl()
    # daoStatus.createsession('localhost')
    # daoStatus.setlogger()
    # daoStatus.loadkeyspace('cyberOps')
    # currentTime = datetime.datetime.now()
    # someMinutesAgo = currentTime - datetime.timedelta(minutes=5)
    # print(currentTime)
    # print(someMinutesAgo)
    # daoStatus.select_hr_inRange('Cris', someMinutesAgo, currentTime)

    daoServerStatus = machiniesDAO.MachinesDAOImplementation()  # dao.MachinesDAOImplementation()
    daoServerStatus.create_session('10.40.39.33')  # ip_dcos'10.40.39.33'
    daoServerStatus.set_logger()
    daoServerStatus.load_keyspace(keyspace)
    daoServerStatus.create_table()
    daoServerStatus.create_spark_context('10.40.39.33')#ip_dcos
    daoServerStatus.load_spark_table()
    currentTime = datetime.datetime.now() #+ datetime.timedelta(hours=2)
    cuTime = time.time()
    someMinutesAgo = currentTime - datetime.timedelta(minutes=20)
    ts = time.mktime(currentTime.timetuple())
    print(time.mktime(currentTime.timetuple()))

    # newDate = currentTime - datetime.timedelta(minutes=1)
    # metric_log = [0,'test','Core5', 'temperature', 3,  newDate]#.strftime("%Y-%m-%d %H:%M:%S")
    # daoServerStatus.insert_metric(1, 1,metric_log=metric_log)

    a =daoServerStatus.select_MAXmetric_and_date(1, 1, ['gamora.lsi.die'], initial_timestamp=someMinutesAgo, end_timestamp=currentTime, metric_type="temperature", server_component="")
    # b = daoServerStatus.select_metricAndDate_by_roomAndRackAndServerIds(1, 1, ['gamora.lsi.die'], someMinutesAgo,
    #                                                 currentTime, metric_type="temperature")
    #print(a)


    # daoServerStatus.insert_metric(1, 1, 'gamora.lsi.die', currentTime, temperature_anomaly=False, energy_anomaly=False,
    #                               utilization_anomaly=False)
    # daoServerStatus.update(1, 1, 'gamora.lsi.die', currentTime, type_anomaly="utilization")
    # daoServerStatus.select_anomaly_by_roomAndRackAndServerIds(1, 1, 'gamora.lsi.die',  someMinutesAgo,
    #                                                           currentTime, anomaly_type="temperature")
    #
    # daoServerStatus.select_summary_anomaly_by_roomAndRackAndServerIds(1, 1, 'gamora.lsi.die', someMinutesAgo,
    #                                                                   currentTime, anomaly_type="temperature")
    #

