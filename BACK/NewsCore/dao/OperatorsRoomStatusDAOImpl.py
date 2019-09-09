import logging
from cassandra.cluster import Cluster, BatchStatement
from flask_jsonpify import jsonify
from datetime import datetime




class OperatorsRoomStatusDAOImpl:
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
        c_sql = "CREATE TABLE IF NOT EXISTS OperatorsRoomStatusDB (room_id varchar, date timestamp, temperature double, humidity double, heat_index double, noise_level double, PRIMARY KEY (room_id,date)) WITH CLUSTERING ORDER BY (date ASC);"  # variable no se puede modificar si lo ponemos a frozen.
        self.session.execute(c_sql)
        self.log.info("News Table Employee state Created !!!")

    def insert_new_row(self,temperature, noise_level, humidity, heat_index, date, room_id):
        insert_sql = self.session.prepare(
            "INSERT INTO OperatorsRoomStatusDB (room_id, date, temperature,humidity, heat_index, noise_level) VALUES (?,?,?,?,?,?)")  # ,(1, single_news.get_title(), single_news.get_summary(), single_news.get_metric())
        batch = BatchStatement()
        batch.add(insert_sql, (room_id, date, temperature, humidity, heat_index, noise_level))
        self.session.execute(batch)
        self.log.info('Data insertion completed in OperatorsRoomStatusDB')

    def insert_temperature(self, temperature, date, room_id):
        update_sql = self.session.prepare("UPDATE OperatorsRoomStatusDB SET temperature=? WHERE room_id=? AND date=?;")
        batch = BatchStatement()
        batch.add(update_sql, (temperature, room_id, date))
        self.session.execute(batch)
        self.log.info('temperature insertion completed in OperatorsRoomStatusDB')

    def insert_temp_hum_heatIndex(self, temperature,humidity, heatIndex, date, room_id):
        update_sql = self.session.prepare("UPDATE OperatorsRoomStatusDB SET temperature=?, humidity=?, heat_index=? WHERE room_id=? AND date=?;")
        batch = BatchStatement()
        batch.add(update_sql, (temperature,humidity, heatIndex, room_id, date))
        self.session.execute(batch)
        self.log.info('temperature insertion completed in OperatorsRoomStatusDB')

    def insert_noise_level(self, noise_level, date, room_id):
        update_sql = self.session.prepare("UPDATE OperatorsRoomStatusDB SET noise_level=? WHERE room_id=? AND date=?;")
        batch = BatchStatement()
        batch.add(update_sql, (noise_level, room_id, date))
        self.session.execute(batch)
        self.log.info('emoiton insertion completed in OperatorsRoomStatusDB')


    def select_data(self, room_id):
        query = self.session.prepare('SELECT * FROM OperatorsRoomStatusDB WHERE room_id=? limit 10;')
        rows = self.session.execute(query, parameters=(room_id,))
        return rows

    def select_temperature(self, room_id, date):
        query = self.session.prepare('SELECT temperature FROM OperatorsRoomStatusDB WHERE room_id=? AND date=?limit 10;')
        rows = self.session.execute(query, parameters=(room_id,date,))
        return rows

    def select_noise_level(self, room_id, date):
        query = self.session.prepare('SELECT noise_level FROM OperatorsRoomStatusDB WHERE room_id=? AND date=?limit 10;')
        rows = self.session.execute(query, parameters=(room_id,date,))
        return rows

    def select_temperature_inRange(self, room_id, initial_timestamp, end_timestamp):
        query = self.session.prepare('SELECT avg(temperature) FROM OperatorsRoomStatusDB WHERE room_id=? AND date >= ? AND date <= ?;')
        rows = self.session.execute(query, parameters=(room_id, initial_timestamp, end_timestamp,))
        print(rows.was_applied)
        print('AVG temperature DONE!!')
        return rows.was_applied

    def select_noise_level_inRange(self, room_id, initial_timestamp, end_timestamp):
        query = self.session.prepare(
            'SELECT avg(noise_level) FROM OperatorsRoomStatusDB WHERE room_id=? AND date >= ? AND date <= ?;')
        rows = self.session.execute(query, parameters=(room_id, initial_timestamp, end_timestamp,))
        print(rows.was_applied)
        print('AVG temperature DONE!!')
        return rows.was_applied

    def select_allData_inRange(self, room_id, initial_timestamp, end_timestamp):
        query = self.session.prepare(
            'SELECT avg(noise_level),avg(temperature), avg(humidity), avg(heat_index) FROM OperatorsRoomStatusDB WHERE room_id=? AND date >= ? AND date <= ?;')
        rows = self.session.execute(query, parameters=(room_id, initial_timestamp, end_timestamp,))
        print('AVG DONE!!')
        data = []
        data.append({"noise":rows.current_rows[0][0], "temperature": rows.current_rows[0][1], "humidity":rows.current_rows[0][2], "heat_index":rows.current_rows[0][3]})
        return data

    def update_data(self):
        pass
    def delete_data(self):
        pass
    def delete_table(self):
        pass

    def delete_keyspace(self, keyspace):
        self.session.execute("DROP KEYSPACE " + self.keyspace)


