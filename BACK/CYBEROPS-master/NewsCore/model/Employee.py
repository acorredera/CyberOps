from datetime import datetime
import uuid

def current_time():
    return datetime.now() #.strftime('%Y-%m-%d %H:%M:%S')

class Employee:
    def __init__(self, alias, profile):
        self.__alias = alias
        self.__profile = profile

    def get_alias(self):
        return self.__alias

    def get_profile(self):
        return self.__profile


    def set_alias(self, alias):
        self.__alias = alias

    def set_profile(self, profile):
        self.__profile = profile



