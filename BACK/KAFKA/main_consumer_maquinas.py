import os
import sys
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import NewsCore.dao.MachinesDAOImplementation as dao
import machines_consumer as consumer
import settings as settings


# nodetool tablestats -H proteus_prueba  -> Table info (esp. byte size) 

if __name__ == "__main__":
    ### Variables ###
    ip_DCOS_cassandra = settings.ip_DCOS_cassandra#"localhost"#settings.ip_DCOS_cassandra#'localhost'#
    keyspace = settings.keyspace_cassandra#'cyberops'#
    topic = 'collectd'
    field2Extract = 'machines'

    strategy = 'SimpleStrategy'
    replication_factor = 2
    ### Cassandra ###
    daoStatus = dao.MachinesDAOImplementation()
    daoStatus.create_session(ip_DCOS_cassandra)
    daoStatus.set_logger()
    daoStatus.load_keyspace(keyspace)

    #daoStatus.create_table()  # only if table is not created previously
    ## Kafka Consumer ###
    consumer_machines = consumer.Consumer(topic=topic, field2Extract=field2Extract, DAO=daoStatus, ip_kafka_DCOS=settings.ip_kafka_DCOS)
    consumer_machines.run()

    #daoStatus.select_all_roomAndRacksIds()
    # end = datetime.datetime.now()
    # ini = end - datetime.timedelta(hours=5000)
    # # rows = daoStatus.select_metricAndDate_by_roomAndRackAndServerIds(1,1, "gamora.lsi.die", ini, end, metric_type="temperature")
    # print("done")
    #daoStatus.test(1,1, "gamora.lsi.die", ini, end, metric_type="temperature",server_component="core")