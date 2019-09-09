from kafka import KafkaConsumer
import json, multiprocessing
from datetime import datetime
from dateutil import parser

class Consumer(multiprocessing.Process):
    def __init__(self, topic, field2Extract, DAO, ip_kafka_DCOS):
        multiprocessing.Process.__init__(self)
        self.stop_event = multiprocessing.Event()
        self._topic = topic
        self._field2Extract = field2Extract
        self._DAO = DAO
        self._ip_kafka_DCOS = ip_kafka_DCOS

    def parse_message(self, log):
        try:
            log6 = log[6].decode("utf-8")
        except AttributeError:
            log6 = log[6]
            pass
        log = json.loads(log6)[0]
        date = datetime.fromtimestamp(log["time"]).isoformat().replace("T", " ")[:-3]
        date = parser.parse(date[:-6] + '00Z') #sentData["timestamp"][:-3] + '00Z'
        #newTimeStamp = parser.parse(sentData["timestamp"][:-2] + '00Z')  # option1-introduce just minutes
        instance = [log["meta"]["index"], log["host"], log["plugin"], log["type"], log["values"][0], date]
        return instance

    def get_ids(self, server_name):
        if('gamora' in server_name or 'fury' in server_name):
            room_id = 1
            rack_id = 1
        elif ('nebula' in server_name):
            room_id = 1
            rack_id = 2
        elif('groot' in server_name or 'hulk' in server_name):
            room_id = 2
            rack_id = 1
        else:
            room_id = 3
            rack_id = 1
        return room_id, rack_id

    def run(self):
        #creation of the consumer
        #ip_DCOS_service = '10.40.39.32:1027' #casandra: 10.40.39.33:9042 kafka: 10.40.39.33:1025
        consumer = KafkaConsumer(bootstrap_servers=self._ip_kafka_DCOS,auto_offset_reset='latest')
        consumer.subscribe([self._topic])
        # Always listening new messages, when they arrive, they are parsed and introduced into Cassandra
        while not self.stop_event.is_set():
            for message in consumer:
                entry = self.parse_message(message)
                room_id, rack_id = self.get_ids(entry[1])

                #print(entry)
                self._DAO.insert_metric(room_id, rack_id,entry) #Insert into Cassandra
                #print("--------------------NEW LOG IN THE DATABASE---------------------")
                if self.stop_event.is_set():
                    break
        consumer.close()

    def stop(self):
        self.stop_event.set()
