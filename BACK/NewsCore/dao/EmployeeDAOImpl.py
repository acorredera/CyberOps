import logging
from cassandra.cluster import Cluster, BatchStatement
import pandas as pd



class EmployeeDAOImpl:
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
        c_sql = "CREATE TABLE IF NOT EXISTS EmployeeStateDB (alias varchar, date timestamp, HR int, emotion varchar, arousal float, PRIMARY KEY (alias,date)) WITH CLUSTERING ORDER BY (date ASC);"  # variable no se puede modificar si lo ponemos a frozen.
        self.session.execute(c_sql)
        self.log.info("News Table Employee state Created !!!")


    def insert_parameter(self, data2insert, date, employee_alias, parameter="HR"):
        command = "UPDATE EmployeeStateDB SET "+parameter+"=? WHERE alias=? AND date=?;"
        update_sql = self.session.prepare(command)
        batch = BatchStatement()
        batch.add(update_sql, (data2insert, employee_alias, date))
        self.session.execute(batch)
        self.log.info(parameter+' insertion completed in EmployeeStateDB')

    def insert_HR(self, hrData, date, employee_alias):
        update_sql = self.session.prepare("UPDATE EmployeeStateDB SET HR=? WHERE alias=? AND date=?;")
        batch = BatchStatement()
        batch.add(update_sql, (hrData, employee_alias, date))
        self.session.execute(batch)
        self.log.info('HR insertion completed in EmployeeStateDB')


    def insert_emotion(self, emotion, date, employee_alias):
        update_sql = self.session.prepare("UPDATE EmployeeStateDB SET emotion=? WHERE alias=? AND date=?;")
        batch = BatchStatement()
        batch.add(update_sql, (emotion, employee_alias, date))
        self.session.execute(batch)
        self.log.info('emoiton insertion completed in EmployeeStateDB')

    def insert_audio_arousal(self, emotion, date, employee_alias):
        update_sql = self.session.prepare("UPDATE EmployeeStateDB SET arousal=? WHERE alias=? AND date=?;")
        batch = BatchStatement()
        batch.add(update_sql, (emotion, employee_alias, date))
        self.session.execute(batch)
        self.log.info('emoiton insertion completed in EmployeeStateDB')

    def select_data(self, employee_alias):
        query = self.session.prepare('SELECT * FROM EmployeeStateDB WHERE alias=? limit 10;')
        rows = self.session.execute(query, parameters=(employee_alias,))
        return rows

    def select_hr(self, employee_alias, date):
        query = self.session.prepare('SELECT HR FROM EmployeeStateDB WHERE alias=? AND date=?;')
        rows = self.session.execute(query, parameters=(employee_alias,date,))
        return rows

    def select_emotion(self, employee_alias, date):
        query = self.session.prepare('SELECT emotion FROM EmployeeStateDB WHERE alias=? AND date=?;')
        rows = self.session.execute(query, parameters=(employee_alias,date,))
        return rows

    def select_hr_inRange(self, employee_alias, initial_timestamp, end_timestamp):
        query = self.session.prepare('SELECT avg(HR) FROM EmployeeStateDB WHERE alias=? AND date >= ? AND date <= ?;')
        rows = self.session.execute(query, parameters=(employee_alias, initial_timestamp, end_timestamp,))
        print(rows.was_applied)
        print('AVG HR DONE!!')
        return rows.was_applied

    def select_some_hr_inRange(self, employee_alias, initial_timestamp, end_timestamp, number_samples):
        #query = self.session.prepare('SELECT date,HR FROM EmployeeStateDB WHERE alias=? AND date >= ? AND date <= ? limit ?;')
        query = self.session.prepare('SELECT date,HR FROM EmployeeStateDB WHERE alias=? AND date >= ? AND date <= ?;')
        rows = self.session.execute(query, parameters=(employee_alias, initial_timestamp, end_timestamp,))
        print(rows._current_rows)
        return rows._current_rows

    def select_some_emotion_inRange(self, employee_alias, initial_timestamp, end_timestamp, number_samples):
        #query = self.session.prepare('SELECT date,HR FROM EmployeeStateDB WHERE alias=? AND date >= ? AND date <= ? limit ?;')
        query = self.session.prepare('SELECT date,emotion FROM EmployeeStateDB WHERE alias=? AND date >= ? AND date <= ?;')
        rows = self.session.execute(query, parameters=(employee_alias, initial_timestamp, end_timestamp,))
        print(rows._current_rows)
        return rows._current_rows

    def select_some_parameter_inRange(self, employee_alias, initial_timestamp, end_timestamp, number_samples, parameter="arousal"):
        # query = self.session.prepare('SELECT date,HR FROM EmployeeStateDB WHERE alias=? AND date >= ? AND date <= ? limit ?;')
        command = 'SELECT date,'+parameter+' FROM EmployeeStateDB WHERE alias=? AND date >= ? AND date <= ?;'
        query = self.session.prepare(command)
        rows = self.session.execute(query, parameters=(employee_alias, initial_timestamp, end_timestamp,))
        print(rows._current_rows)
        return rows._current_rows

    def select_dominant_emotion(self, employee_alias, initial_timestamp, end_timestamp):
        data = []
        query = self.session.prepare('SELECT emotion FROM EmployeeStateDB WHERE alias=? AND date>=? AND date<=?;')
        rows = self.session.execute(query, parameters=(employee_alias, initial_timestamp, end_timestamp,))
        for row in rows:
            # print(row)
            data.append(row)

        df = pd.DataFrame(data)

        try:
            emotion = df.loc[:, "emotion"].mode()
            del data[:]
            del df

            print(emotion[0])
            print('DOMINANT EMOTION DONE!!')
            return emotion[0]
        except:
            return 0
            print("NO EMOTION DETECTED")

    def update_data(self):
        pass
    def delete_data(self):
        pass
    def delete_table(self):
        pass

    def delete_keyspace(self, keyspace):
        self.session.execute("DROP KEYSPACE " + self.keyspace)


