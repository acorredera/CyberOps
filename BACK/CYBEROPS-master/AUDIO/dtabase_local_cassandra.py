from cassandra.cluster import Cluster
import random
import datetime
import time

# The connect() method takes an optional keyspace argument which sets the default keyspace for all queries made through that Session:

cluster = Cluster()
session = cluster.connect('mydb')

# You can always change a Session's keyspace using set_keyspace() or by executing a USE <keyspace> query
#
# session.set_keyspace('users')
# # or you can do this instead
# session.execute('USE users')

# Now that we have a Session we can begin to execute queries. The simplest way to execute a query is to use execute()
#
# rows = session.execute('SELECT id, title, year FROM books')
# for user_row in rows:
#     print user_row.title, user_row.year

def create_table():
    c_sql = "CREATE TABLE IF NOT EXISTS noiseLevel (id int, db int, PRIMARY KEY (id));"  # variable no se puede modificar si lo ponemos a frozen.
    session.execute(c_sql)
    print("Noise Table state Created !!!")

create_table()

#closing my connection
session.shutdown()