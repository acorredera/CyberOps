import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import audioop
import datetime
import json
import math
import datetime
import pyaudio
import time
from kafka import KafkaProducer
import settings as settings


# PATH = os.getcwd() #/home/marta/PycharmProjects/CYBEROPS/AUDIO


def passiveListen(producer, topic_emotion, stream, CHUNK):

    tm = datetime.datetime.now()
    timestamp_rounded = tm - datetime.timedelta(seconds=tm.second, microseconds=tm.microsecond)

    # starts passive listening for disturbances

    input = stream.read(CHUNK)
    rms_value = audioop.rms(input, 2)
    db = int(round(20 * math.log(rms_value,10)))
    print(db)

    data2send = {"timestamp": timestamp_rounded.strftime("%Y-%m-%d %H:%M:%S"),  # redondeado
                 "db": db,
                 "room_id": "1"}

    producer.send(topic_emotion, json.dumps(data2send).encode('utf-8'))

    return True

if __name__ == "__main__":
    producer = KafkaProducer(bootstrap_servers=settings.ip_kafka_DCOS, api_version=(0, 10, 1)) #localhost:9092
    # convert rms to decibel scale decibel = 20 * log10(rms)
    # http://www.sengpielaudio.com/calculator-db-volt.htm   ----> dB en dBV
    THRESHOLD1 = 3162  # 100-317 rms (40-50 db); 70 dB = 3162 rms
    THRESHOLD2 = 10000  # 80 dB = 10000 rms
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    LISTEN_TIME = 5  # listening time in seconds = record time

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while (True):
        passiveListen(producer=producer, topic_emotion="cyberops_noise_level", CHUNK=CHUNK, stream=stream)
        time.sleep(60.0) #time in seconds
    producer.close()