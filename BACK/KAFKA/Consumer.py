import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kafka import KafkaConsumer
import NewsCore.model.AlexaAnswer as aAnswers
import json
import datetime
from dateutil import parser
import multiprocessing


class Consumer(multiprocessing.Process):
    def __init__(self, topic, field2Extract, DAO, ip_kafka_DCOS):
        multiprocessing.Process.__init__(self)
        self.stop_event = multiprocessing.Event()
        self._topic = topic
        self._field2Extract = field2Extract
        self._DAO = DAO
        self._ip_kafka_DCOS = ip_kafka_DCOS


    #parser to extract message information:
    def parse_message(self,tupla):
        tuplaFields = {"toppic":[], "partition":[], "offset":[], "timestamp":[], "timestamp_type":[], "key":[], "value":[], "checksum":[], "serialized_key_size":[], "serialized_value_size":[]}
        toppic = (str(tupla[0])).replace('"','') #toppic
        partition = int(tupla[1]) #partition
        offset =int(tupla[2]) #offset
        timestamp = datetime.datetime.fromtimestamp(int(tupla[3]) / 1e3) #timestamp -> without seconds: the_time = the_time.replace(second=0, microsecond=0)
        timestamp_type = int(tupla[4]) #timestamp_type
        key = str(tupla[5]) #key
        value = json.loads(tupla[6].decode('utf-8'))#(str(tupla[6])).replace('"','') #la opción comentada para caso de envio de una sola cosa
        checksum = str(tupla[7]) #checksum
        serialized_key_size = str(tupla[8]) #serialized_key_size
        serialized_value_size = str(tupla[9]) #serialized_value_size
        tuplaFields["toppic"]=toppic
        tuplaFields["partition"] = partition
        tuplaFields["offset"] = offset
        tuplaFields["timestamp"] = timestamp
        tuplaFields["timestamp_type"] = timestamp_type
        tuplaFields["key"] = key
        tuplaFields["value"] = value
        tuplaFields["checksum"] = checksum
        tuplaFields["serialized_key_size"] = serialized_key_size
        tuplaFields["serialized_value_size"] = serialized_value_size
        return tuplaFields

    def stop(self):
        self.stop_event.set()

    def run(self):
        #creation of the consumer
        #casandra: 10.40.39.33:9042
        #kafka: 10.40.39.33:1025
        consumer = KafkaConsumer(bootstrap_servers=self._ip_kafka_DCOS,
                                 auto_offset_reset='latest')
        consumer.subscribe([self._topic])

        # Always listening new messages, when they arrive, they are parsed and introduced into Cassandra
        while not self.stop_event.is_set():
            for message in consumer:
                #print(message) #values in [6]
                entry = self.parse_message(message)
                sentData = entry["value"]
                print(sentData)
                # Obtain AAdata and introduce it into cassandraDB
                #newTimeStamp = parser.parse(sentData["timestamp"])  # option1-introduce all the info
                dataOfSensor = sentData[self._field2Extract]
                if(self._field2Extract=="hr"):
                    newTimeStamp = parser.parse(sentData["timestamp"][:-3] + '00Z')  # option1-introduce just minutes
                    self._DAO.insert_HR(int(dataOfSensor), newTimeStamp, sentData["employeeAlias"])
                elif(self._field2Extract=="emotion"):
                    newTimeStamp = parser.parse(sentData["timestamp"][:-2] + '00Z')  # option1-introduce just minutes
                    self._DAO.insert_emotion(dataOfSensor, newTimeStamp, sentData["employeeAlias"])
                elif(self._field2Extract=="answer"):
                    for question_index in range(0, len(sentData["index"]) - 1):
                        if(sentData["profile"][question_index] == 'Finish'):
                            continue
                        elif(sentData["profile"][question_index] == 'Penalization'):
                            punctuation = -0.1
                        else:
                            punctuation = float(sentData["answer"][question_index]) / 10
                        dateFormat = datetime.datetime.fromtimestamp((sentData["timestamp"][question_index])/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        newTimeStamp = parser.parse(dateFormat + 'Z')
                        a1 = aAnswers.AlexaAnswer(id_question=sentData["index"][question_index],
                                                  answer=sentData["answer"][question_index],
                                                  alias_employee=sentData["name"],
                                                  date=newTimeStamp,#sentData["timestamp"][question_index],
                                                  profile_question=aAnswers.QuestionProfile[
                                                      sentData["profile"][question_index]],
                                                  punctuation=punctuation,
                                                  penalization=aAnswers.QuestionProfile[sentData["penalization"][
                                                      question_index]])  # aAnswers.QuestionProfile[sentData["penalization"][question_index]].value
                        #   ts = time.mktime(date.timetuple())
                        #   convert to date again:   date = datetime.datetime.fromtimestamp(ts)
                        self._DAO.insert_answer(a1)
                elif(self._field2Extract=="db"):
                    newTimeStamp = parser.parse(sentData["timestamp"][:-2] + '00Z')
                    self._DAO.insert_noise_level(dataOfSensor, newTimeStamp, sentData["room_id"])
                elif (self._field2Extract == "arousal"):
                    newTimeStamp = parser.parse(sentData["timestamp"][:-2] + '00Z')
                    print(newTimeStamp)
                    self._DAO.insert_parameter(float(dataOfSensor), newTimeStamp, sentData["employeeAlias"], parameter="arousal")
                elif (self._field2Extract == "temperature"):
                    dateFormat = datetime.datetime.utcfromtimestamp(int(sentData["timestamp"])).strftime('%Y-%m-%d %H:%M:%S')
                    newTimeStamp = parser.parse(dateFormat[:-2] + '00Z')  # option1-introduce just minutes
                    self._DAO.insert_temp_hum_heatIndex(float(sentData["temperature"]),float(sentData["humidity"]), float(sentData["heat_index"]), newTimeStamp, sentData["room_id"])
                print("--------------------------------------------------")
                if self.stop_event.is_set():
                    break
        consumer.close()


