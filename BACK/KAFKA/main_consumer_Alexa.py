import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import KAFKA.Consumer as consumer
import NewsCore.dao.AlexaAnswerDAOImpl as dao
import settings as settings



if __name__ == "__main__":
    ip_DCOS_cassandra = settings.ip_DCOS_cassandra
    keyspace = settings.keyspace_cassandra
    topic = "cyberops_alexaAnswer"
    field2Extract="answer"

    #Open and prepare cassandra:
    daoAlexaAnswers = dao.AlexaAnswerDAOImpl()
    daoAlexaAnswers.createsession(ip_DCOS_cassandra)
    daoAlexaAnswers.setlogger()
    daoAlexaAnswers.loadkeyspace(keyspace)
    daoAlexaAnswers.create_table()

    consumer_alexaAnswer = consumer.Consumer(topic=topic, field2Extract=field2Extract, DAO=daoAlexaAnswers, ip_kafka_DCOS=settings.ip_kafka_DCOS)
    consumer_alexaAnswer.run()
