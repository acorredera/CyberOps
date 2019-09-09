import logging
from cassandra.cluster import Cluster, BatchStatement
from flask_jsonpify import jsonify
from cassandra.query import dict_factory
from dotmap import DotMap

class AlexaAnswerDAOImpl:
    __instance =None
    def __init__(self): #las barra bajas significan que es privado, keyspace = name of cluster where the tables live
        self.cluster = None
        self.session = None
        self.keyspace = None
        self.log = None
        #self.results_as_dict=results_as_dict

    def __del__(self):
        self.cluster.shutdown()

    def createsession(self, ip):
        self.cluster = Cluster([ip])#localhost o IPs
        self.session = self.cluster.connect(self.keyspace)
        # if(self.results_as_dict):
        #     self.session.row_factory = dict_factory
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
        # if((self.results_as_dict and self.keyspace in [row['keyspace_name'] for row in rows])
        #    or (not self.results_as_dict and self.keyspace in [row[0] for row in rows])):
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
        # if ((self.results_as_dict and self.keyspace in [row['keyspace_name'] for row in rows])
        #     or (not self.results_as_dict and self.keyspace in [row[0] for row in rows])):
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
        c_sql = "CREATE TABLE IF NOT EXISTS AlexaAnswerDB (id UUID,id_question int, answer varchar,alias_employee varchar, date timestamp,profile_question int,penalization_profile int, punctuation double, PRIMARY KEY ((alias_employee),profile_question,penalization_profile, date)) WITH CLUSTERING ORDER BY (profile_question ASC, penalization_profile ASC, date ASC);"#WITH CLUSTERING ORDER BY (date ASC, profile_question ASC);"  # variable no se puede modificar si lo ponemos a frozen.
        self.session.execute(c_sql)
        self.log.info("News Table Employee state Created !!!")


    def insert_answer(self,alexaAnswer):
        insert_sql = self.session.prepare(
            "INSERT INTO AlexaAnswerDB (id, id_question, answer,alias_employee, date,profile_question, penalization_profile, punctuation) VALUES (?,?,?,?,?,?,?,?)")  # ,(1, single_news.get_title(), single_news.get_summary(), single_news.get_metric())
        batch = BatchStatement()
        batch.add(insert_sql, (alexaAnswer.get_id_answer(), alexaAnswer.get_id_question(), alexaAnswer.get_answer(),alexaAnswer.get_alias_employee(), alexaAnswer.get_date(), alexaAnswer.get_profile_question_int(), alexaAnswer.get_penalization_int(), alexaAnswer.get_punctuation()))
        self.session.execute(batch)
        self.log.info('Data insertion completed in AlexaAnswerDB')

    def select_punctuation_by_profile_date_employee(self, employee_alias, profile, initial_timestamp,end_timestamp):
        query = self.session.prepare('SELECT avg(punctuation) FROM AlexaAnswerDB '
                                     'WHERE alias_employee=? '
                                     'AND profile_question =? '
                                     'AND penalization_profile=6 '
                                     'AND date >= ? AND date <= ?;')
        rows = self.session.execute(query, parameters=(employee_alias, profile, initial_timestamp, end_timestamp,))
        punctuation = rows.was_applied
        print('AVG punctuation DONE!!')
        print(punctuation)
        if(punctuation<0):
            punctuation = 0
        return punctuation

    def select_sum_penalization_by_alias_date(self,employee_alias, initial_timestamp, end_timestamp):
        query = self.session.prepare(
            'select alias_employee,penalization_profile, sum(punctuation), date from alexaanswerdb '
            'WHERE alias_employee=? '
            'AND profile_question=5 '
            'AND penalization_profile IN (1,2,3,4) '
            'AND date>=? AND date <=? '
            'GROUP BY alias_employee, profile_question, penalization_profile;') #sum(punctuation)
        rows = self.session.execute(query, parameters=(employee_alias, initial_timestamp, end_timestamp,))
        print(rows)
        #ver cómo lo devolvemos como json
        return rows

    def select_avg_profiles_by_alias_date(self,employee_alias, initial_timestamp, end_timestamp):
        query = self.session.prepare(
            'select alias_employee, profile_question, avg(punctuation), date from alexaanswerdb '
            'WHERE alias_employee=? '
            'AND profile_question=? '
            'AND penalization_profile=6 '
            'AND date>=? AND date <=?;') #sum(punctuation)
        # 'select alias_employee, profile_question, avg(punctuation), date from alexaanswerdb '
        # 'WHERE alias_employee=? '
        # 'AND profile_question IN(1,2,3,4) '
        # 'AND penalization_profile=6 '
        # 'AND date>=? AND date <=? '
        # 'GROUP BY alias_employee, profile_question;')  # sum(punctuation)
        rows_list = []
        for profile_index in (1,2,3,4):
            rows = self.session.execute(query, parameters=(employee_alias,profile_index, initial_timestamp, end_timestamp,))
            if (not rows.current_rows or (rows.current_rows[0].alias_employee == None)):
                rows_list.append(DotMap({"alias_employee":employee_alias, "profile_question":profile_index, "system_avg_punctuation":0, "date":initial_timestamp}))
            else:
                rows_list.append(rows.current_rows[0])
        return rows_list

    def select_avg_and_color_by_alias_date(self, employee_alias, initial_timestamp, end_timestamp):
        query = self.session.prepare(
            'select alias_employee, avg(punctuation) from alexaanswerdb WHERE alias_employee=? AND profile_question IN(1,2,3,4) AND penalization_profile=6 AND date>=? AND date <=?;') #sum(punctuation) #'AND date>=? AND date <=? ''GROUP BY alias_employee
        rows = self.session.execute(query, parameters=(employee_alias, initial_timestamp, end_timestamp,))
        print(rows)
        #ver cómo lo devolvemos como json
        if(not rows.current_rows):
            result = 0
            color = "red"
        else:
            result = rows.current_rows[0][1]
            if (result <= 0.33):
                color = "red"
            elif (result > 0.33 and result <= 0.66):
                color = "yellow"
            else:
                color = "green"
        return result,color

    def select_avg_profiles_by_date_allUsers(self,list_users, initial_timestamp, end_timestamp):
        avg_status_user = []
        for user in list_users:
            profile,color = AlexaAnswerDAOImpl.select_avg_and_color_by_alias_date(self, user, initial_timestamp, end_timestamp)
            avg_status_user.append({"name":user, "profile": round(profile,2), "color":color})
            #CÓMO AÑADIMOS LA PENALIZACIÓN A ESTO???
        #ver cómo lo devolvemos como json
        return avg_status_user#jsonify({'Employees': avg_status_user})

    def select_users(self):
        query = self.session.prepare(
            'select distinct alias_employee from alexaanswerdb;')  # sum(punctuation)
        rows = self.session.execute(query)
        print(rows)
        # ver cómo lo devolvemos como json
        list_users = []
        for user in rows:
            list_users.append(user.alias_employee)
        return list_users

    def delete_answer(self, id_question, alias, timestamp):
        query = self.session.prepare("DELETE FROM AlexaQuestionDB WHERE id_question=? AND alias_employee=? AND date=?;")
        batch = BatchStatement()
        batch.add(query, (id_question, alias,timestamp,))
        self.session.execute(batch)
        self.log.info('ROW DELETED!')

    def delete_table(self):
        query = self.session.prepare('DROP TABLE AlexaAnswerDB;')
        self.session.execute(query)


    def delete_keyspace(self, keyspace):
        self.session.execute("DROP KEYSPACE " + keyspace)
        self.log.info('KEYSPACE DELETED!')

