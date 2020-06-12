import logging
from cassandra.cluster import Cluster, BatchStatement
import cassandra

class AnomaliesInMachinesDAOImpl:
    __instance =None
    def __init__(self): #las barra bajas significan que es privado, keyspace = name of cluster where the tables live
        self.cluster = None
        self.session = None
        self.keyspace = None
        self.log = None

    def __del__(self):
        self.cluster.shutdown()

    def create_session(self, ip):
        self.cluster = Cluster([ip])#localhost o IPs
        self.session = self.cluster.connect(self.keyspace)
        #self.session.row_factory = dict_factory
    def get_session(self):
        return self.session
    # How about Adding some log info to see what went wrong
    def set_logger(self):
        log = logging.getLogger()
        log.setLevel('INFO')
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        log.addHandler(handler)
        self.log = log
    # Create Keyspace based on Given Name
    def load_keyspace(self, keyspace):
        """
        :param keyspace:  The Name of Keyspace to be created
        :return:
        """
        # Before we create new lets check if exiting keyspace; we will drop that and create new
        self.keyspace = keyspace.lower()
        rows = self.session.execute("SELECT keyspace_name FROM system_schema.keyspaces")
        if self.keyspace in [row[0] for row in rows]:
            self.log.info("Loading keyspace...")
        else:
            self.log.info("creating keyspace...")
            self.session.execute("""
                            CREATE KEYSPACE %s
                            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
                            """ % self.keyspace)
        self.log.info("setting keyspace...")
        self.session.set_keyspace(self.keyspace)
    def create_keyspace(self, keyspace):
        """
        :param keyspace:  The Name of Keyspace to be created
        :return:
        """
        # Before we create new lets check if exiting keyspace; we will drop that and create new
        self.keyspace = keyspace.lower()
        rows = self.session.execute("SELECT keyspace_name FROM system_schema.keyspaces")
        if self.keyspace in [row[0] for row in rows]:
            self.log.info("dropping existing keyspace...")
            self.session.execute("DROP KEYSPACE " + self.keyspace)
        self.log.info("creating keyspace...")
        self.session.execute("""
                CREATE KEYSPACE %s
                WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
                """ % self.keyspace)
        self.log.info("setting keyspace...")
        self.session.set_keyspace(self.keyspace)

    def create_table(self):
        c_sql = "CREATE TABLE IF NOT EXISTS AnomaliesMachinesDB (room_id int, rack_id int, server_id varchar, date timestamp, temperature_anomaly boolean, energy_anomaly boolean, utilization_anomaly boolean, PRIMARY KEY ((room_id,rack_id),server_id,date)) WITH CLUSTERING ORDER BY (server_id ASC, date ASC);"
        self.session.execute(c_sql)
        self.log.info("News Table AnomaliesMachinesDB Created !!!")

    def insert_metric(self, room_id, rack_id, server_id, datetime, temperature_anomaly=False, energy_anomaly=False, utilization_anomaly=False):
        self.log.info("Inserting AAdata into ProteusDB")
        insert_sql = self.session.prepare(
            "INSERT INTO AnomaliesMachinesDB (room_id, rack_id, server_id, date, temperature_anomaly, energy_anomaly, utilization_anomaly) VALUES (?,?,?,?,?,?,?)")
        batch = BatchStatement()
        batch.add(insert_sql, (room_id, rack_id, server_id, datetime, temperature_anomaly, energy_anomaly, utilization_anomaly,))
        self.session.execute(batch)
        self.log.info('Data insertion completed in AnomaliesMachinesDB successfully')

    def update(self, room_id, rack_id, server_id, datetime, type_anomaly="temperature", value_anomlay=True):
        print("update")
        update_sql = "UPDATE AnomaliesMachinesDB SET "+type_anomaly+"_anomaly=? WHERE room_id=? AND rack_id=? AND server_id =? AND date=? ;"
        update_sql = self.session.prepare(update_sql)
        parametros = [value_anomlay] + [room_id] + [rack_id] + [server_id] + [datetime]
        self.session.execute(update_sql, parameters=(cassandra.query.ValueSequence(parametros)))
        self.log.info('anomaly update completed in AnomaliesMachinesDB')



    def select_all_roomAndRacksIds(self):
        query = self.session.prepare('SELECT DISTINCT room_id,rack_id FROM AnomaliesMachinesDB;')
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
        query = self.session.prepare(
            'SELECT server_id FROM AnomaliesMachinesDB WHERE room_id =? AND rack_id=?;')  # cqlsh 3.4.0 ->  group by rack_id,server_id
        rows = self.session.execute(query, parameters=(room_id, rack_id,))
        set_servers = set()
        for server in rows:
            set_servers.add(server.server_id)
        return set_servers

    def select_anomaly_by_roomAndRackAndServerIds(self, room_id, rack_id, server_id, initial_timestamp,
                                                        end_timestamp, anomaly_type="temperature"):
        query_str = 'SELECT max('+anomaly_type+'_anomaly) FROM AnomaliesMachinesDB WHERE room_id=? AND rack_id=? AND server_id =? AND date >= ? AND date <= ?;'
        parametros = [room_id] + [rack_id] + [server_id] + [initial_timestamp, end_timestamp]
        query = self.session.prepare(query_str)
        rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
        print('max metric DONE!!')
        metric_result = rows._current_rows[0][0]
        print(metric_result)
        return metric_result

    def select_summary_anomaly_by_roomAndRackAndServerIds(self, room_id, rack_id, server_id, initial_timestamp,
                                                        end_timestamp, anomaly_type="temperature"):
        query_str = 'SELECT max(temperature_anomaly), max(energy_anomaly), max(utilization_anomaly) FROM AnomaliesMachinesDB WHERE room_id=? AND rack_id=? AND server_id =? AND date >= ? AND date <= ?;'
        parametros = [room_id] + [rack_id] + [server_id] + [initial_timestamp, end_timestamp]
        query = self.session.prepare(query_str)
        rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
        metric_result = rows._current_rows[0]
        anomaly_detected = False
        for metric in metric_result:
            anomaly_detected = metric or anomaly_detected
        print(anomaly_detected)
        return anomaly_detected

    def delete_table(self):
        query = self.session.prepare('DROP TABLE AnomaliesMachinesDB;')
        self.session.execute(query)

    def delete_keyspace(self, keyspace):
        self.session.execute("DROP KEYSPACE " + keyspace)
        self.log.info('KEYSPACE DELETED!')

