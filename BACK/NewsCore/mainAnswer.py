import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import NewsCore.model.AlexaAnswer as aAnswers
import NewsCore.dao.AlexaAnswerDAOImpl as dao
import datetime
from flask_jsonpify import jsonify

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

    daoAlexaAnswers = dao.AlexaAnswerDAOImpl()
    daoAlexaAnswers.createsession('10.40.39.33')
    daoAlexaAnswers.setlogger()
    daoAlexaAnswers.loadkeyspace('cyberOps')
    daoAlexaAnswers.create_table()

    #creation of AAdata
    #currentTime = datetime.datetime.now()
    # a1 = aAnswers.AlexaAnswer(1, '5', 'cris', currentTime, aAnswers.QuestionProfile.Mood, 15, aAnswers.QuestionProfile.NoProfile)
    # a2 = aAnswers.AlexaAnswer(2, '6', 'cris', currentTime+datetime.timedelta(seconds=1), aAnswers.QuestionProfile.Mood, 18, aAnswers.QuestionProfile.NoProfile)
    # a3 = aAnswers.AlexaAnswer(4, '4', 'cris', currentTime+datetime.timedelta(seconds=2), aAnswers.QuestionProfile.Environment, 10, aAnswers.QuestionProfile.NoProfile)
    # a7 = aAnswers.AlexaAnswer(4, '7', 'cris', currentTime + datetime.timedelta(seconds=5),aAnswers.QuestionProfile.Penalization, 10, aAnswers.QuestionProfile.Mood)
    # a4 = aAnswers.AlexaAnswer(3, '8', 'juan', currentTime+datetime.timedelta(seconds=3), aAnswers.QuestionProfile.Coordination, 40, aAnswers.QuestionProfile.NoProfile)
    # a5 = aAnswers.AlexaAnswer(4, '1', 'juan', currentTime+datetime.timedelta(seconds=4), aAnswers.QuestionProfile.Coordination, 50, aAnswers.QuestionProfile.NoProfile)
    # a6 = aAnswers.AlexaAnswer(4, '1', 'juan', currentTime+datetime.timedelta(seconds=5), aAnswers.QuestionProfile.Mood, 60, aAnswers.QuestionProfile.NoProfile)
    # a8 = aAnswers.AlexaAnswer(4, '7', 'juan', currentTime + datetime.timedelta(seconds=6),aAnswers.QuestionProfile.Penalization, 10, aAnswers.QuestionProfile.Coordination)
    # a9 = aAnswers.AlexaAnswer(4, '7', 'juan', currentTime + datetime.timedelta(seconds=7),aAnswers.QuestionProfile.Penalization, -8, aAnswers.QuestionProfile.Mood)
    # # #
    # daoAlexaAnswers.insert_answer(a1)
    # daoAlexaAnswers.insert_answer(a2)
    # daoAlexaAnswers.insert_answer(a3)
    # daoAlexaAnswers.insert_answer(a4)
    # daoAlexaAnswers.insert_answer(a5)
    # daoAlexaAnswers.insert_answer(a6)
    # daoAlexaAnswers.insert_answer(a7)
    # daoAlexaAnswers.insert_answer(a8)
    #daoAlexaAnswers.insert_answer(a9)

    # someMinutesAgo = currentTime - datetime.timedelta(hours=800)
    # print(currentTime)
    # print(someMinutesAgo)
    # # AAdata = daoAlexaAnswers.select_avg_profiles_by_alias_date('juan', someMinutesAgo, currentTime)
    # # datos = daoAlexaAnswers.select_sum_penalization_by_alias_date('juan', someMinutesAgo, currentTime)
    # # datos2 = daoAlexaAnswers.select_punctuation_by_profile_date_employee('juan', 3, someMinutesAgo, currentTime)
    # # datos3 = daoAlexaAnswers.select_avg_profiles_by_date_allUsers(someMinutesAgo, currentTime)
    # datos4 = daoAlexaAnswers.select_avg_and_color_by_alias_date('cris', someMinutesAgo, currentTime)
    # list_users = daoAlexaAnswers.select_users()
    #
    # #probar esto mañana:
    # datps = daoAlexaAnswers.select_avg_profiles_by_date_allUsers(list_users,someMinutesAgo, currentTime)
    # print('end')