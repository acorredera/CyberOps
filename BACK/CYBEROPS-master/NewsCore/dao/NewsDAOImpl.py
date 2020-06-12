import logging
from cassandra.cluster import Cluster, BatchStatement



class NewsDAOImpl:
    __instance =None
    def __init__(self, keyspace=None): #las barra bajas significan que es privado, keyspace = name of cluster where the tables live
        self.cluster = None
        self.session = None
        self.keyspace = keyspace
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
        rows = self.session.execute("SELECT keyspace_name FROM system_schema.keyspaces")
        if keyspace in [row[0] for row in rows]:
            self.log.info("Loading keyspace...")
        else:
            self.log.info("creating keyspace...")
            self.session.execute("""
                            CREATE KEYSPACE %s
                            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
                            """ % keyspace)
        self.log.info("setting keyspace...")
        self.session.set_keyspace(keyspace)
    def createkeyspace(self, keyspace):
        """
        :param keyspace:  The Name of Keyspace to be created
        :return:
        """
        # Before we create new lets check if exiting keyspace; we will drop that and create new
        rows = self.session.execute("SELECT keyspace_name FROM system_schema.keyspaces")
        if keyspace in [row[0] for row in rows]:
            self.log.info("dropping existing keyspace...")
            self.session.execute("DROP KEYSPACE " + keyspace)
        self.log.info("creating keyspace...")
        self.session.execute("""
                CREATE KEYSPACE %s
                WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
                """ % keyspace)
        self.log.info("setting keyspace...")
        self.session.set_keyspace(keyspace)


    def select_data(self, nameDB, numNews):
        c_sql = "SELECT * FROM "+nameDB+" limit ?;"
        query = self.session.prepare(c_sql)
        rows = self.session.execute(query, parameters=(numNews,))
        # for row in rows:
        #     print(row.metric, row.title, row.summary)
        return rows

    def update_data(self):
        pass
    def delete_data(self):
        pass


    def create_table(self):
        c_sql = "CREATE TABLE IF NOT EXISTS NoticiasDB (id UUID, date timestamp, news int, typology varchar, punctuation int, title varchar, summary varchar, graph varchar, side varchar, metrics frozen<list<text>>, PRIMARY KEY (id,punctuation)) WITH CLUSTERING ORDER BY (punctuation DESC);" #variable no se puede modificar si lo ponemos a frozen.
        self.session.execute(c_sql)
        self.log.info("News Table Noticias Created !!!")
        c_sql = "CREATE TABLE IF NOT EXISTS MetricsDB (metric varchar PRIMARY KEY, news_id set<UUID>);"  # variable no se puede modificar si lo ponemos a frozen.
        self.session.execute(c_sql)
        self.log.info("News Table Metrics Created !!!")

    # lets do some batch insert
    def insert_data(self, single_news):
        #INSERT DATA IN NOTICASDB
        insert_sql = self.session.prepare("INSERT INTO NoticiasDB (id, date, news, typology, punctuation, title, summary, graph, side, metrics) VALUES (?,?,?,?,?,?,?,?,?,?)") #,(1, single_news.get_title(), single_news.get_summary(), single_news.get_metric())
        batch = BatchStatement()
        batch.add(insert_sql, (single_news.get_id(), single_news.get_date(), single_news.get_news_number(), single_news.get_typology(), single_news.get_punctuation(), single_news.get_title(), single_news.get_summary(), single_news.get_graph(), single_news.get_side(), single_news.get_metrics()))
        self.session.execute(batch)
        self.log.info('Data insertion completed in NoticiasDB')

        #INSERT DATA IN METRICSDB
        identificatorNew = single_news.get_id()
        for metric in single_news.get_metrics():
            insert_sql = self.session.prepare("UPDATE MetricsDB SET news_id=news_id+? WHERE metric=?")  # ,(1, single_news.get_title(), single_news.get_summary(), single_news.get_metric())
            toAdd = set({identificatorNew})
            batch = BatchStatement()
            batch.add(insert_sql, (toAdd,metric))
            self.session.execute(batch)
            self.log.info('Data insertion completed in MetricsDB')


    #QUIZÁS LO OLPIMO SERÍA O HACERLO ASÍNCRONO O LIMITAR A X NÚMERO DE NOTICIAS
    #***ALLOW FILTERING ISSUES: (https://www.datastax.com/dev/blog/allow-filtering-explained-2)
    # 1 ) If your table contains for example a 1 million rows and 95% of them have the requested value
    # for the metrics column, the query will still be relatively efficient and you should use ALLOW FILTERING.
    # 2) On the other hand, if your table contains 1 million rows and only 2 rows contain the requested value for the metrics column, your query is extremely inefficient.
    #  Cassandra will load 999, 998 rows for nothing. If the query is often used, it is probably better to ADD AN INDEX on the metrics column.
    def selectNews(self, keywords): # ver si esto se puede hacer asíncrono yyy cómo devolver datos, ahora mismo tipo Row(title,summary)

        #q2 = "SELECT title,summary FROM NoticiasDB WHERE metrics CONTAINS 'pue' OR  ALLOW FILTERING;"
        #OPCION1: SELECCIONAMOS metrics QUE CONTENGAN LAS KEYWORDS QUE QUEREMOS PERO TIENE QUE SER CON ANDs
        # query = 'SELECT title,summary FROM NoticiasDB WHERE metrics CONTAINS \"'+keywords[0]+'\"'
        # for keyw_index in range(1,len(keywords)):
        #     query += ' AND metrics CONTAINS \"'+keywords[keyw_index] +'\"'
        # query+=" ALLOW FILTERING DISTINCT ;"
        # AAdata = self.session.execute(query)  # , parameters=(keywords,)
        # results_idsNews = []
        #query = "SELECT id,title,summary FROM NoticiasDB WHERE metrics CONTAINS 'pue' AND metrics CONTAINS 'coste energía refrigeración' ALLOW FILTERING DISTINCT ;" #si contiene 'x' y contiene 'y' y contiene 'z'...
        #OPCION2: SELECCIONAMOS TODAS LAS NOTICIAS QUE TENGAN 1 VARIABLE, HACEMOS TANTAS LLAMADAS COMO KEYWORDS??? ESTO ES UNA BURRADA PERO BUENO...
        # Y LUEGO ELIMINAMOS SOLAPAMIENTOS EN NOTICIAS, POR ID.
        idNews = []
        #for word in keywords:
        query = self.session.prepare('SELECT news_id FROM MetricsDB WHERE metric IN ?;')
        idNews.append(self.session.execute(query, parameters=(keywords,)))
        #query = "SELECT title,summary FROM NoticiasDB WHERE metrics IN (['pue','coste energía refrigeración']) ALLOW FILTERING;" # Cuando se quiere EXACTAMENTE lo que viene después del IN
        #query = "SELECT title,summary FROM NoticiasDB WHERE metrics CONTAINS 'pue' OR metrics CONTAINS 'coste energía refrigeración' ALLOW FILTERING;"
        #query =self.session.prepare("SELECT title,summary FROM NoticiasDB WHERE metrics CONTAINS 'pue';") #("SELECT title,summary FROM NoticiasDB WHERE metrics=? ALLOW FILTERING;") CONTAINS #("SELECT title,summary FROM NoticiasDB WHERE metrics CONTAINS ? ALLOW FILTERING;") ---ORDER BY punctuation DESC
        # for word in keywords:
        #     AAdata.append(self.session.execute(query, parameters=(word,)))
        #OBTENER IDS
        results_idsNews = set({})
        for idSubResponse in idNews:
            for ids in idSubResponse:
                results_idsNews = results_idsNews.union(ids[0])

        #OBTENER NOTICAS DESDE SUS IDS:
        finalNews = []
        results_news = []
        #self.session.execute("PAGING OFF")
        query = self.session.prepare('SELECT punctuation,title,summary FROM NoticiasDB WHERE id IN ? ORDER BY punctuation DESC;')
        query.fetch_size = None
        finalNews.append(self.session.execute(query, parameters=(list(results_idsNews),)))
        for newsPart in finalNews:
            for news in newsPart:
                results_news.append(news)
        #try:
            #print(results_idsNews)
        #except ReadTimeout:
        #   print("Query timed out:")
        return results_news

