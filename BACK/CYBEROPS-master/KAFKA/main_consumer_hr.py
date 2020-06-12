import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import NewsCore.dao.EmployeeDAOImpl as dao
import KAFKA.Consumer as consumer
import settings as settings


if __name__ == "__main__":
    #variables
    ip_DCOS_cassandra = settings.ip_DCOS_cassandra
    keyspace = settings.keyspace_cassandra
    topic = 'cyberops_HR'
    field2Extract = 'hr'
    #loading of the cassandra seesion, creatiopn of table (if needded)
    daoStatus = dao.EmployeeDAOImpl()
    daoStatus.createsession(ip_DCOS_cassandra)
    daoStatus.setlogger()
    daoStatus.loadkeyspace(keyspace)
    daoStatus.create_table() #only if table is not created previously
    #Run consumer for HR:
    consumer_hr = consumer.Consumer(topic=topic, field2Extract=field2Extract, DAO=daoStatus, ip_kafka_DCOS=settings.ip_kafka_DCOS)
    consumer_hr.run()
