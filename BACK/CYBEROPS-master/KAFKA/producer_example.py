from kafka import KafkaProducer
import json

producer = KafkaProducer(bootstrap_servers='10.40.39.33:1025', api_version=(0, 10, 1))
# Asynchronous by default
date = "2018-11-05T17:41:00Z"
emotion = 'happy'
employee = 'Cris'

#send information as a dictionary
d = {'timestamp':date,
     "emotion":emotion,
     "employeeAlias":employee}
print(d)
producer.send('hi',json.dumps(d).encode('utf-8'))



# time.sleep(1)
producer.close()

# https://github.com/dpkp/kafka-python/blob/master/example.py