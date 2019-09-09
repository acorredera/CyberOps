from datetime import datetime
import uuid
from enum import Enum

class QuestionProfile(Enum):
    Mood = 1
    Work = 2
    Coordination = 3
    Environment = 4
    Penalization = 5
    Null = 6


# class Animal(Enum):
#     DOG = 1
#     CAT = 2
#
# print(Animal.DOG)
# # <Animal.DOG: 1>
#
# print(Animal.DOG.value)
# # 1
#
# print(Animal.DOG.name)
# # "DOG"

class AlexaAnswer:
    def __init__(self, id_question, answer, alias_employee,date, profile_question, punctuation, penalization):
        self.__id_answer = uuid.uuid1()
        self.__id_question = id_question
        self.__answer = answer
        self.__alias_employee = alias_employee
        self.__date = date
        self.__profile_question = profile_question
        self.__punctuation = punctuation
        self.__penalization = penalization


    def get_id_answer(self):
        return self.__id_answer

    def get_id_question(self):
        return self.__id_question

    def get_answer(self):
        return self.__answer

    def get_alias_employee(self):
        return self.__alias_employee

    def get_date(self):
        return self.__date

    def get_profile_question_string(self):
        return self.__profile_question.name

    def get_profile_question_int(self):
        return self.__profile_question.value

    def get_punctuation(self):
        return self.__punctuation

    def get_penalization_string(self):
        return self.__penalization.name

    def get_penalization_int(self):
        return self.__penalization.value


    def set_id_answer(self, id_answer):
        self.__id_answer = id_answer

    def set_id_question(self, id_question):
        self.__id_question = id_question

    def set_answer(self, answer):
        self.__answer = answer

    def set_alias_employee(self, alias_employee):
        self.__alias_employee = alias_employee

    def set_date(self, date):
        self.__date = date

    def set_profile_question(self, profile_question):
        self.__profile_question = profile_question

    def set_punctuation(self, punctuation):
        self.__punctuation = punctuation

    def set_penalization(self, penalization):
        self.__penalization = penalization
