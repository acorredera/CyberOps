# Create credentials json
# Creating a service account (From the Role drop-down list, select Project > Owner)
# https://cloud.google.com/docs/authentication/getting-started

# Make requests to Cloud Database
# https://firebase.google.com/docs/firestore/quickstart?hl=es-419

#Installations: pip install --upgrade firebase-admin
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from kafka import KafkaProducer

import settings as settings

def remove_and_upload_firebase_data(previous_path = os.getcwd()+"/ASISTENTE"):
    # Use a service account

    if(not len(firebase_admin._apps)):
        cred = credentials.Certificate(os.path.join(previous_path, "Cyberops-firebase-v1-project-owner.json"))
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    # Read Data from collection
    users_ref = db.collection(u'dialogflow')
    docs = users_ref.get()
    topic = "cyberops_alexaAnswer"
    producer = KafkaProducer(bootstrap_servers=settings.ip_kafka_DCOS, api_version=(0, 10, 1))
    for doc in docs:
        registro = u'{}'.format(doc.to_dict())
        registro_dict=eval(registro)
        #print(registro_dict['profile'])

        #save/insert answers in cassandra

        producer.send(topic, json.dumps(registro_dict).encode('utf-8'))
        doc.reference.delete()

    producer.close()

    # with open('registros_firebase.csv', 'a') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for doc in docs:
    #         registro = u'{} [ {}]'.format(doc.id, doc.to_dict())
    #         for key, value in registro:
    #             writer.writerow([key, value])