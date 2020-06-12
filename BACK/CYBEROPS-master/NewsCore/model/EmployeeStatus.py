from datetime import datetime

def current_time():
    return datetime.now()  # .strftime('%Y-%m-%d %H:%M:%S')


class EmployeeStatus:
    def __init__(self, alias, hr=0, emotion='', timestamp=current_time):
        self.__alias =alias
        self.__hr = hr
        self.__emotion = emotion
        self.__timestamp = timestamp

    def get_alias(self):
        return self.__alias

    def get_hr(self):
        return self.__hr

    def get_emotion(self):
        return self.__emotion

    def get_timestamp(self):
        return self.__timestamp


    def set_alias(self, alias):
        self.__alias = alias

    def set_hr(self, hr):
        self.__hr = hr

    def set_emotion(self, emotion):
        self.__emotion = emotion

    def set_timestamp(self, timestamp):
        self.__timestamp = timestamp





