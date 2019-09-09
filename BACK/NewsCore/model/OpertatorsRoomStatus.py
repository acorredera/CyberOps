from datetime import datetime

def current_time():
    return datetime.now()  # .strftime('%Y-%m-%d %H:%M:%S')


class OperatorsRoomStatus:
    def __init__(self, room_id,temperature,noise_level='', timestamp=current_time):
        self.__room_id =room_id
        self.__temperature = temperature
        self.__noise_level = noise_level
        self.__timestamp = timestamp

    def get_room_id(self):
        return self.__room_id

    def get_temperature(self):
        return self.__temperature

    def get_noise_level(self):
        return self.__noise_level

    def get_timestamp(self):
        return self.__timestamp


    def set_room_id(self, room_id):
        self.__room_id = room_id

    def set_temperature(self, temperature):
        self.__temperature = temperature

    def set_noise_level(self, noise_level):
        self.__noise_level = noise_level

    def set_timestamp(self, timestamp):
        self.__timestamp = timestamp





