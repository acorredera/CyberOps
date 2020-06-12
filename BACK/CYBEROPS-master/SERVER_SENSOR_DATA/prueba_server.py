import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import settings
'''
    Simple udp socket server
'''

#CLIENT = SEND INFO
import socket, sys
from kafka import KafkaProducer
import json
UDP_IP = settings.ip_running_programs_server#"10.40.38.30"#"127.0.0.1"#
UDP_PORT = 2004



print("UDP target IP:", UDP_IP)
print("UDP target port:", UDP_PORT)

#SERVER = RECEIVE INFO
sock = socket.socket(socket.AF_INET, # Internet
                    socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

producer = KafkaProducer(bootstrap_servers=settings.ip_kafka_DCOS, api_version=(0, 10, 1))  # localhost:9092
topic_temperature = "cyberops_temperature"
while True:
    print (sys.stderr, '\nwaiting to receive message')
    data, address = sock.recvfrom(4096)

    print (sys.stderr, 'received %s bytes from %s' % (len(data), address))
    print (sys.stderr, data)
    data2send = eval(data.decode("utf-8"))
    #producer kafka
    producer.send(topic_temperature, json.dumps(data2send).encode('utf-8'))
    if data:
        sent = sock.sendto(data, address)
        print (sys.stderr, 'sent %s bytes back to %s' % (sent, address))
