import logging
from cassandra.cluster import Cluster, BatchStatement
import cassandra
from flask_jsonpify import jsonify
from datetime import datetime




class ServersStatusDAOImpl:
    __instance =None
    def __init__(self): #las barra bajas significan que es privado, keyspace = name of cluster where the tables live
        self.cluster = None
        self.session = None
        self.keyspace = None
        self.log = None

    def __del__(self):
        self.cluster.shutdown()

    def createsession(self, ip):
        self.cluster = Cluster([ip])#localhost o IPs
        self.session = self.cluster.connect(self.keyspace)
        #self.session.row_factory = dict_factory
    def getsession(self):
        return self.session
    # How about Adding some log info to see what went wrong
    def setlogger(self):
        log = logging.getLogger()
        log.setLevel('INFO')
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        log.addHandler(handler)
        self.log = log
    # Create Keyspace based on Given Name
    def loadkeyspace(self, keyspace):
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
    def createkeyspace(self, keyspace):
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
        c_sql = "CREATE TABLE IF NOT EXISTS ServerStatusDB (rack_id varchar, server_id varchar, date timestamp, server_component varchar, profile int, value int,punctuation double, PRIMARY KEY ((rack_id),server_id,profile,date)) WITH CLUSTERING ORDER BY (server_id ASC,profile ASC, date DESC);"  # variable no se puede modificar si lo ponemos a frozen.
        self.session.execute(c_sql)
        self.log.info("News Table ServerStatusDB Created !!!")

    def insert_metric(self, server_status_log):
        self.log.info("Inserting AAdata into ProteusDB")
        insert_sql = self.session.prepare(
            "INSERT INTO ServerStatusDB (rack_id, server_id, date, server_component, profile, value, punctuation) VALUES (?,?,?,?,?,?,?)")
        batch = BatchStatement()
        batch.add(insert_sql,
                  (server_status_log.get_rack_id(), server_status_log.get_server_id(), server_status_log.get_date(), server_status_log.get_server_component(), server_status_log.get_profile_int(), server_status_log.get_value(), server_status_log.get_punctuation()))
        self.session.execute(batch)
        self.log.info('Data insertion completed in ServerStatusDB successfully')


    def select_rack_ids(self):
        query = self.session.prepare('SELECT DISTINCT rack_id FROM ServerStatusDB;')
        rows = self.session.execute(query)
        print(rows)
        list_racks = []
        for rack in rows:
            list_racks.append(rack.rack_id)
        return list_racks


    def select_server_ids_by_rack_id(self, rack_id):
        query = self.session.prepare('SELECT server_id FROM ServerStatusDB WHERE rack_id=? group by rack_id,server_id;')
        rows = self.session.execute(query, parameters=(rack_id,))
        list_servers = []
        for server in rows:
            list_servers.append(server.server_id)
        return list_servers

    def select_statusRack_by_rack_servers_date_profile(self, rack_id, server_ids, initial_timestamp, end_timestamp, profile=[1,2,3,4,5]):
        query_str = 'SELECT avg(punctuation), min(punctuation) FROM ServerStatusDB WHERE rack_id=? AND server_id IN (? '
        server_ids = list(server_ids)
        for server_id in range(len(server_ids)-1):
            query_str+=",?"
        query_str += ") AND profile "
        if(len(profile)>1):
            query_str += "IN (?"
            for profile_id in range(len(profile) - 1):
                query_str += ",?"
            query_str+=")"
            parametros = parametros = [rack_id] + server_ids + profile+[initial_timestamp, end_timestamp]
        else:
            query_str+="=?"
            parametros = parametros = [rack_id] + server_ids + [profile[0], initial_timestamp, end_timestamp]
        query_str += " AND date >= ? AND date <= ?;" # también podemos poner profile fijo: profile IN (1,2,...)
        query = self.session.prepare(query_str)
        rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
        print('AVG punctuation DONE!!')
        punctuation = {"avg":rows._current_rows[0][0],
                       "min":rows._current_rows[0][1],
                       "color": self.get_color(rows._current_rows[0][0])}
        return punctuation #ahora estamos devolviendo la media pero quizás lo ideal sería devolver media y mínimo y si alguna es 0, entonces marcar en rojo porque hay algún server que no hace nada...


    def select_statusServers_by_rack_servers_date_profile(self, rack_id, server_ids, initial_timestamp, end_timestamp, profile=[1,2,3,4,5]):
        query_str = 'SELECT server_id, avg(punctuation), min(punctuation) FROM ServerStatusDB WHERE rack_id=? AND server_id IN (? '
        server_ids = list(server_ids)
        for server_id in range(len(server_ids)-1):
            query_str+=",?"
        query_str += ") AND profile "
        if(len(profile)>1):
            query_str += "IN (?"
            for profile_id in range(len(profile) - 1):
                query_str += ",?"
            query_str+=")"
            parametros = parametros = [rack_id] + server_ids + profile+[initial_timestamp, end_timestamp]
        else:
            query_str+="=?"
            parametros = parametros = [rack_id] + server_ids + [profile[0], initial_timestamp, end_timestamp]
        query_str += " AND date >= ? AND date <= ? group by rack_id, server_id"
        query = self.session.prepare(query_str)
        rows = self.session.execute(query, parameters=(cassandra.query.ValueSequence(parametros)))
        avg_status_servers=[]
        for data in rows:
            color= self.get_color(data.system_avg_punctuation)
            avg_status_servers.append({"server_id": data.server_id, "status": data.system_avg_punctuation, "color": color})
        return avg_status_servers #ahora estamos devolviendo la media pero quizás lo ideal sería devolver media y mínimo y si alguna es 0, entonces marcar en rojo porque hay algún server que no hace nada...

    def select_statusServer_by_rack_id_server_date(self, rack_id, server_id, initial_timestamp, end_timestamp):
        query_str = 'SELECT profile, avg(punctuation), min(punctuation) FROM ServerStatusDB ' \
                    'WHERE rack_id=? AND server_id =? AND profile IN (1,2,3,4,5) ' \
                    'AND date >= ? AND date <= ? group by rack_id, server_id, profile;'
        query = self.session.prepare(query_str)
        rows = self.session.execute(query, parameters=(rack_id, server_id, initial_timestamp, end_timestamp))
        status_server=[]
        for data in rows:
            status_server.append({"profile": data.profile, "status": data.system_avg_punctuation})
        return status_server #ahora estamos devolviendo la media pero quizás lo ideal sería devolver media y mínimo y si alguna es 0, entonces marcar en rojo porque hay algún server que no hace nada...


    def delete_table(self):
        query = self.session.prepare('DROP TABLE ServerStatusDB;')
        self.session.execute(query)

    def delete_keyspace(self, keyspace):
        self.session.execute("DROP KEYSPACE " + keyspace)
        self.log.info('KEYSPACE DELETED!')

    def get_color(self, avg):
        color = "green"
        if (avg <= 0.33):
            color = "red"
        elif (avg > 0.33 and avg <= 0.66):
            color = "yellow"
        else:
            color = "green"
        return color