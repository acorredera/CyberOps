import logging
from cassandra.cluster import Cluster, BatchStatement
import cassandra
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from pyspark.sql import SQLContext
#from dateutil import tz
import datetime
import os, time
#import pandas as pd

class MachinesDAOImplementation:

    __instance =None

    def __init__(self): #In a variable, underscore means it's private. Keyspace: cluster name where tables are stored
        self.cluster = None
        self.session = None
        self.keyspace = None
        self.log = None
        self.spark_table = None
        self.sqlContext = None
        self.sc = None

    def __del__(self):
        self.cluster.shutdown()

    def create_spark_context(self,ip, sparkMaster="local",appName = "cyberopsMAchinesDAO"):
        # Load spark context -> check this in future for clusters ...
        os.environ['PYSPARK_SUBMIT_ARGS'] = \
            "--packages com.datastax.spark:spark-cassandra-connector_2.11:2.3.0 " \
            "--conf spark.cassandra.connection.host=" + ip + " "\
            "--conf spark.sql.session.timeZone=GMT " \
            "--conf spark.executor.extraJavaOptions=-Duser.timezone=GMT "\
            "--conf spark.driver.extraJavaOptions=-Duser.timezone=GMT"+\
            " pyspark-shell"  # "--conf spark.mesos.executor.docker.image=mesosphere/spark:2.4.0-2.2.1-3-hadoop-2.6 "\

        # os.environ['SPARK_LOCAL_IP'] = "10.40.39.30:"
        conf = SparkConf().setMaster(sparkMaster).setAppName(appName)
        sc = SparkContext.getOrCreate(conf)  # master="mesos://10.40.39.24:5050"
        #sc.conf.set("spark.sql.session.timeZone", "GMT")
        self.sqlContext = SQLContext(sc)
        self.sc = sc


    def create_session(self, ip):
        self.cluster = Cluster([ip]) #localhost or IPs
        self.session = self.cluster.connect(self.keyspace)


    def get_session(self):
        return self.session

    #Log info for troubleshooting
    def set_logger(self):
        log = logging.getLogger()
        log.setLevel('INFO')
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        log.addHandler(handler)
        self.log = log

    #Load Keyspace by parameter
    def load_keyspace(self, keyspace):
        self.keyspace = keyspace.lower()
        rows = self.session.execute("SELECT keyspace_name FROM system_schema.keyspaces")
        if self.keyspace in [row[0] for row in rows]:
            self.log.info("Loading keyspace...")
            self.session.set_keyspace(self.keyspace)
            self.log.info("Keyspace set!")
        else:
            self.log.info("This keyspace does not exist! Try creating it first")


    def create_keyspace(self, keyspace, strategy, rep_factor):
        self.keyspace = keyspace.lower()
        rows = self.session.execute("SELECT keyspace_name FROM system_schema.keyspaces")
        if self.keyspace in [row[0] for row in rows]:
            self.log.info("This keyspace already exists! Try dropping it first")
        else:
            self.log.info("creating keyspace...")
            self.session.execute("CREATE KEYSPACE " + keyspace + " WITH replication = { 'class': '" + strategy + "', 'replication_factor': '" + str(rep_factor) + "' }")
            self.log.info("setting keyspace...")
            self.session.set_keyspace(self.keyspace)
            self.log.info("Keyspace set!")

    def delete_keyspace(self, keyspace):
        self.keyspace = keyspace.lower()
        rows = self.session.execute("SELECT keyspace_name FROM system_schema.keyspaces")
        if self.keyspace in [row[0] for row in rows]:
            self.log.info("Deleting keyspace...")
            self.session.execute("DROP KEYSPACE " + self.keyspace)
            self.log.info("The keyspace has been deleted successfully")
        else:
            self.log.info("This keyspace does not exist! Check the spelling")

    def create_table(self):
        self.log.info("Creating Proteus table")
        #Variable no se puede modificar si lo ponemos a frozen.
        c_sql = "CREATE TABLE IF NOT EXISTS machinesdb (room_id int, rack_id int, server_id varchar, metric_type varchar, server_component varchar, metric_value int, date timestamp, PRIMARY KEY ((room_id,rack_id),server_id, metric_type,date, server_component)) WITH CLUSTERING ORDER BY (server_id ASC, metric_type ASC,date ASC);"
        self.session.execute(c_sql)
        self.log.info("Proteus table created!")
        #Load spark table
        # self.load_spark_table()

    def load_spark_table(self):
        table_df = self.sqlContext.read \
            .format("org.apache.spark.sql.cassandra") \
            .options(table="machinesdb", keyspace=self.keyspace) \
            .load()
        self.spark_table = table_df

    def insert_metric(self,room_id, rack_id, metric_log):
        #self.log.info("Inserting data into machinesdb")
        insert_sql = self.session.prepare("INSERT INTO machinesdb (room_id, rack_id, server_id, server_component, metric_type, metric_value, date) VALUES (?,?,?,?,?,?,?)")
        batch = BatchStatement()
        batch.add(insert_sql, (room_id, rack_id, metric_log[1], metric_log[2], metric_log[3], int(metric_log[4]), metric_log[5]))
        self.session.execute(batch)
        #self.log.info('Data insertion completed in machinesdb successfully')

    def select_all_roomAndRacksIds(self):
        query = self.session.prepare('SELECT DISTINCT room_id,rack_id FROM machinesdb;')
        rows = self.session.execute(query)
        print(rows)
        dict_rooms_racks = {}
        for room_id,rack_id in rows:
            if(room_id in dict_rooms_racks):
                previous_racks_set = dict_rooms_racks[room_id]
                previous_racks_set.add(rack_id)
                dict_rooms_racks[room_id] = previous_racks_set
            else:
                new_set = set()
                new_set.add(rack_id)
                dict_rooms_racks[room_id] = new_set
        return dict_rooms_racks # key: room_id, value: set(racks_ids)

    def select_server_ids_by_roomAndRackid(self, room_id, rack_id):
        query = self.session.prepare('SELECT server_id FROM machinesdb WHERE room_id =? AND rack_id=?;')# cqlsh 3.4.0 ->  group by rack_id,server_id
        rows = self.session.execute(query, parameters=(room_id,rack_id,))
        set_servers = set()
        for server in rows:
            set_servers.add(server.server_id)
        return set_servers

    def select_server_metric_by_roomAndRackAndServerIds(self, room_id,rack_id, server_id, initial_timestamp, end_timestamp, metric_type="temperature"):
        query_str = 'SELECT avg(metric_value) FROM machinesdb WHERE room_id=? AND rack_id=? AND server_id =? AND metric_type = ? AND date >= ? AND date <= ?;'
        parametros = [room_id] + [rack_id] + [server_id] + [metric_type, initial_timestamp, end_timestamp]
        query = self.session.prepare(query_str)
        rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
        print('AVG metric DONE!!')
        metric_result = rows._current_rows[0][0]
        return metric_result



    def select_metricAndDate_by_roomAndRackAndServerIds(self, room_id, rack_id, server_ids_list, initial_timestamp,
                                                        end_timestamp, metric_type="temperature"):

        query_str = 'SELECT date, metric_value FROM machinesdb WHERE room_id=? AND rack_id=? AND server_id IN (? '
        server_ids = list(server_ids_list)
        for server_id in range(len(server_ids) - 1):
            query_str += ",?"
        query_str += ") AND metric_type = ? AND date >= ? AND date <= ?;"
        parametros = [room_id] + [rack_id] + server_ids + [metric_type, initial_timestamp, end_timestamp]
        print('AVG metric DONE!!')
        query = self.session.prepare(query_str)
        rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
        # metric_result = rows._current_rows[0][0] -> rows.current_rows[i].metric_value & rows.current_rows[i].date
        return rows._current_rows

    def select_metricAndDate_by_roomAndRackAndServerIds_andComponent(self, room_id, rack_id, server_ids_list, initial_timestamp,
                                                        end_timestamp, metric_type="temperature", server_component="core"):
        # query_str = 'SELECT date,metric_value FROM machinesdb WHERE room_id=? AND rack_id=? AND server_id =? AND metric_type = ? AND date >= ? AND date <= ?;'
        # parametros = [room_id] + [rack_id] + [server_id] + [metric_type, initial_timestamp, end_timestamp]
        # query = self.session.prepare(query_str)
        # rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
        query_str = 'SELECT date, metric_value, server_component FROM machinesdb WHERE room_id=? AND rack_id=? AND server_id IN (? '
        server_ids = list(server_ids_list)
        for server_id in range(len(server_ids) - 1):
            query_str += ",?"
        query_str += ") AND metric_type = ? AND date >= ? AND date <= ? AND server_component;"
        parametros = [room_id] + [rack_id] + server_ids + [metric_type, initial_timestamp, end_timestamp]
        print('AVG metric DONE!!')
        query = self.session.prepare(query_str)
        rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
        # metric_result = rows._current_rows[0][0] -> rows.current_rows[i].metric_value & rows.current_rows[i].date
        return rows._current_rows

    def select_MAXmetric_and_date(self, room_id, rack_id, server_id, initial_timestamp, end_timestamp, metric_type="temperature", server_component=""):
        initial_timestamp +=datetime.timedelta(hours=2)
        end_timestamp +=datetime.timedelta(hours=2)
        if(server_component==""):
            rows = self.spark_table.selectExpr("date as new_date", "server_id", "metric_value"). \
                where("room_id=" + str(room_id)).\
                where("rack_id=" + str(rack_id)). \
                where(F.col("server_id").isin(server_id)). \
                where("metric_type='" + metric_type + "'"). \
                where((F.col("new_date") > initial_timestamp) & (F.col("new_date") < end_timestamp)). \
                groupBy("new_date").max().orderBy("new_date")                 #where(q1). \

        else:
            #server_component = "Core"
            # where((F.col("date") > initial_timestamp) & (F.col("date") < end_timestamp)). \
            ##
            rows = self.spark_table.selectExpr("date as new_date","server_id", "metric_value").\
                where("room_id=" + str(room_id)).\
                where("rack_id=" + str(rack_id)). \
                where(F.col("server_id").isin(server_id)).\
                where("metric_type='" + metric_type + "'"). \
                where((F.col("new_date") > initial_timestamp) & (F.col("new_date") < end_timestamp)). \
                where("server_component LIKE('" + server_component + "%')").\
                groupBy("new_date").max().orderBy("new_date")
        rows.show()
        # for row in rows.rdd.collect():
        #     print(row["new_date"])
        #     print(row["max(metric_value)"])
        return rows #iterate doing :   for row in rows.rdd.collect():
                                         #print(row["new_date"])
                                         #print(row["max(metric_value)"])



    # def select_metric_by_roomAndRackIds(self, room_id, rack_id, initial_timestamp,
    #                                              end_timestamp, metric_type="temperature", servers_list=[]):
    #     #select servers in rack & room
    #     if(servers_list==[]):
    #         servers_list = list(self.select_server_ids_by_roomAndRackid(room_id, rack_id))
    #     query_str = 'SELECT avg(metric_value) FROM machinesdb WHERE room_id=? AND rack_id=? AND server_id IN (? '
    #     for server_id in range(len(servers_list) - 1):
    #         query_str += ",?"
    #     query_str += ") AND metric_type=?"
    #     parametros = parametros = [room_id] + [rack_id] + servers_list + [metric_type, initial_timestamp, end_timestamp]
    #     query_str += " AND date >= ? AND date <= ?;"  # tambien podemos poner profile fijo: profile IN (1,2,...)
    #     query = self.session.prepare(query_str)
    #     rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
    #     metric_result = rows._current_rows[0][0]
    #     return metric_result, servers_list


        #
        # query_str = 'SELECT avg(metric_value) FROM machinesdb WHERE room_id=? AND rack_id=? AND server_id =? AND metric_type = ? AND date >= ? AND date <= ?;'
        # parametros = [room_id] + [rack_id] + [server_id] + [metric_type, initial_timestamp, end_timestamp]
        # query = self.session.prepare(query_str)
        # rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
        # print('AVG metric DONE!!')
        # metric_result = rows._current_rows[0][0]
        # return metric_result


    def update_temperature_anomaly(self, anomaly, room_id, rack_id, server_id, timestamp, metric_type="temperature"):
        query_str = 'UPDATE machinesdb SET temperature_anomaly=? WHERE room_id=? AND rack_id=? AND server_id =? AND metric_type = ? AND date = ? ;'
        parametros = [room_id] + [rack_id] + [server_id] + [metric_type, timestamp, anomaly]
        query = self.session.prepare(query_str)
        rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
        # batch = BatchStatement()
        # batch.add(query_str, (anomaly, room_id, rack_id, server_id, timestamp, metric_type))
        # self.session.execute(batch)
        print('Anomaly UPDATED!!')

    def update_data(self):
        pass

    def delete_data(self):
        pass

    def delete_table(self):
        pass

    def delete_keyspace(self, keyspace):
        self.session.execute("DROP KEYSPACE " + self.keyspace)

    # def select_all_by_id(self, id):
    #     query = self.session.prepare('SELECT * FROM machinesdb WHERE id = ?;')
    #     rows = self.session.execute(query, parameters=(id,))
    #     return rows
    #
    # def select_all_by_name(self, name):
    #     query = self.session.prepare('SELECT * FROM machinesdb WHERE name = ?;')
    #     rows = self.session.execute(query, parameters=(name,))
    #     return rows
    #
    # def select_range_by_name(self, name, limit):
    #     query = self.session.prepare('SELECT * FROM machinesdb WHERE name = ? LIMIT ?;')
    #     rows = self.session.execute(query, parameters=(name,limit,))
    #     return rows
    #
    # def select_all_by_host(self, host):
    #     query = self.session.prepare('SELECT * FROM machinesDB WHERE host = ?;')
    #     rows = self.session.execute(query, parameters=(host,))
    #     return rows
    #
    # def select_all_by_type(self, type):
    #     query = self.session.prepare('SELECT * FROM machinesdb WHERE type = ?;')
    #     rows = self.session.execute(query, parameters=(type,))
    #     return rows
    #
    # def select_all_by_date(self,id , host,  initial_timestamp, end_timestamp):
    #     query = self.session.prepare('SELECT * FROM machinesdb WHERE id=? AND host=? WHERE date >= ? AND date <= ?;')
    #     rows = self.session.execute(query, parameters=(id, host, initial_timestamp, end_timestamp,))
    #     return rows
    #
    # def select_range_by_date(self, initial_timestamp, end_timestamp):
    #     query = self.session.prepare('SELECT * FROM machinesdb WHERE date >= ? AND date <= ?;')
    #     rows = self.session.execute(query, parameters=(initial_timestamp,end_timestamp,))
    #     return rows
    #
    # def select_all(self):
    #     query = self.session.prepare('SELECT * FROM machinesdb ;')
    #     rows = self.session.execute(query)
    #     return rows
    #
    # def select_range(self, limit):
    #     query = self.session.prepare('SELECT * FROM machinesdb LIMIT ? ;')
    #     rows = self.session.execute(query, parameters=(limit,))
    #     return rows
