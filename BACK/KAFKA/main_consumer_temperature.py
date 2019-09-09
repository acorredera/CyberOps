import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import KAFKA.Consumer as consumer
import NewsCore.dao.OperatorsRoomStatusDAOImpl as dao
import settings as settings

if __name__ == "__main__":
    ip_DCOS_cassandra = settings.ip_DCOS_cassandra
    keyspace = settings.keyspace_cassandra
    topic = "cyberops_temperature"
    field2Extract="temperature"

    #Open and prepare cassandra:
    dao = dao.OperatorsRoomStatusDAOImpl()
    dao.createsession(ip_DCOS_cassandra)
    dao.setlogger()
    dao.loadkeyspace(keyspace)
    dao.create_table()

    consumer_alexaAnswer = consumer.Consumer(topic=topic, field2Extract=field2Extract, DAO=dao, ip_kafka_DCOS=settings.ip_kafka_DCOS)
    consumer_alexaAnswer.run()
