from enum import Enum

class statusProfile(Enum):
    temperature = 1 #celsius
    utilization = 2 #%
    power = 3 #W
    voltage = 4 #V
    clock = 5
    fanspeed = 6 #RPM
    Null = 7


class ServerStatus:
    def __init__(self, rack_id, metric_id, server_id,date, server_component, profile, value, punctuation=0.2):
        self.__rack_id = rack_id
        self.__metric_id = metric_id
        self.__server_component = server_component
        self.__server_id = server_id
        self.__date = date
        self.__profile = profile
        self.__value = value
        self.__punctuation = punctuation



    def get_rack_id(self):
        return self.__rack_id

    def get_metric_id(self):
        return self.__metric_id

    def get_server_component(self):
        return self.__server_component

    def get_server_id(self):
        return self.__server_id

    def get_date(self):
        return self.__date

    def get_profile_string(self):
        return self.__profile.name

    def get_profile_int(self):
        return self.__profile.value

    def get_value(self):
        return self.__value

    def get_punctuation(self):
        return self.__punctuation


    def set_rack_id(self, rack_id):
        self.__rack_id = rack_id

    def set_metric_id(self, metric_id):
        self.__metric_id = metric_id

    def set_server_component(self, server_component):
        self.__server_component = server_component

    def set_server_id(self, server_id):
        self.__server_id = server_id

    def set_date(self, date):
        self.__date = date

    def set_profile(self, profile):
        self.__profile = profile

    def set_value(self, value):
        self.__value = value

    def set_punctuation(self, punctuation):
        self.__punctuation = punctuation