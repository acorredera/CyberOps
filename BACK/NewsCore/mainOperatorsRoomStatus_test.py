import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import NewsCore.model.AlexaAnswer as aAnswers
import NewsCore.dao.OperatorsRoomStatusDAOImpl as dao
import datetime

if __name__ == "__main__":
    # daoAlexaQuestions = alexaDAO.AlexaQuestionDAOImpl()
    # daoAlexaQuestions.createsession("10.40.39.33")
    # daoAlexaQuestions.setlogger()
    # daoAlexaQuestions.loadkeyspace('prueba')
    # daoAlexaQuestions.create_table()
    #
    # q2 = alexaQuestion.AlexaQuestion(2,'¿Cómo andas?', alexaQuestion.QuestionProfile.Mood)
    # daoAlexaQuestions.insert_question(q2)
    # #daoAlexaQuestions.update_question_text(1,'holaa')
    # #daoAlexaQuestions.delete_table()
    # #daoAlexaQuestions.delete_keyspace('prueba')
    #
    # print(daoAlexaQuestions.select_question(1))

    daoOperatorsRoom = dao.OperatorsRoomStateDAOImpl()
    daoOperatorsRoom.createsession('localhost')
    daoOperatorsRoom.setlogger()
    daoOperatorsRoom.loadkeyspace('mydb')
    daoOperatorsRoom.create_table()

    #creation of AAdata
    currentTime = datetime.datetime.now()
    daoOperatorsRoom.insert_new_row(temperature=25,noise_level=110.5,date=currentTime,room_id='1', humidity=23, heat_index=24)
    daoOperatorsRoom.insert_new_row(temperature=26,noise_level=111.5,date=currentTime,room_id='1', humidity=24, heat_index=25)
    daoOperatorsRoom.insert_new_row(temperature=27,noise_level=112.5,date=currentTime,room_id='1', humidity=25, heat_index=26)
    daoOperatorsRoom.insert_new_row(temperature=28,noise_level=113.5,date=currentTime,room_id='1', humidity=26, heat_index=27)

    someMinutesAgo = currentTime - datetime.timedelta(hours=800)
    print(currentTime)
    print(someMinutesAgo)
    # print("temperature: ", daoOperatorsRoom.select_temperature_inRange('1', someMinutesAgo, currentTime))
    # print("noise level: ", daoOperatorsRoom.select_noise_level_inRange('1', someMinutesAgo, currentTime))
    data = daoOperatorsRoom.select_allData_inRange('1', someMinutesAgo, currentTime)
    daoOperatorsRoom.insert_temp_hum_heatIndex(100, 50, 12, currentTime, "1")
    data = daoOperatorsRoom.select_allData_inRange('1', someMinutesAgo, currentTime)
    print(data)

