import os,sys
if os.name=="nt":
    operative_system="\\"
else: operative_system="/"
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))).rsplit(operative_system,1)[0])
print (os.path.join(os.path.dirname(__file__)))
print (os.path.dirname(os.path.dirname(os.path.realpath(__file__))).rsplit(operative_system,1)[0])
print(sys.version)
from Anomaly_detection_model import createModel, runAnomaly
import NewsCore.dao.MachinesDAOImplementation as DAO_Maquinas
import NewsCore.dao.AnomaliesInMachinesDAOImpl as daoAnomaly
import datetime
import pandas as pd
import time
import settings
from datetime import timedelta
os.environ['TZ'] = 'UTC' #con esto se soluciona el tema de rescatar bien la hora - estamos diciendo a python
                         #  que coja esa hora para este script (y los uqe dependen de el) (FALTA COMPROBAR QUE FUNCIONE BIEN TMABIeN CON EL MODELO DE ANOMALIAS Y QUE GUARDE LOS DATOS BIEN EN LA TABLA DE ANOMALiAS DE MANERA QUE SE ACCEDA BIEN DESDE LA GUI , ES DEXIR, QUE SE COJA LA HORA UQE ES BIEN) -initial y current time van a estas como "adelantadas ,2horas menos," pero asi no tenemos que estar restando y demas

PATH=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))).rsplit(settings.operative_system,1)[0],"MAQUINAS", "Anomaly")
# import model_params
ip_cassandra = settings.ip_DCOS_cassandra
keyspace = settings.keyspace_cassandra

databaseallData = DAO_Maquinas.MachinesDAOImplementation()
databaseallData.create_session(ip_cassandra)
databaseallData.set_logger()
databaseallData.load_keyspace(keyspace)
databaseallData.create_table()
databaseallData.create_spark_context(ip_cassandra)
databaseallData.load_spark_table()

daoServerStatus = daoAnomaly.AnomaliesInMachinesDAOImpl()  # dao.MachinesDAOImplementation()
daoServerStatus.create_session(ip_cassandra)  # '10.40.39.33'
daoServerStatus.set_logger()
daoServerStatus.load_keyspace(keyspace)
daoServerStatus.create_table()

metrics = ["temperature", "utilization"] #, "utilization" temperature
servers = ["gamora", "fury", "nebula", "groot", "hulk","cris-X550VXK","DESKTOP-V89UFRB.localdomain"]
dict= {"gamora":{"room_id":1, "rack_id":1, "server_id":"gamora.lsi.die"}, #10.40.39.30
       "fury":{"room_id":1, "rack_id":1, "server_id":"fury"}, #10.40.39.29
       "nebula": {"room_id":1, "rack_id":2, "server_id":"nebula"}, #10.40.39.34
       "groot": {"room_id":2, "rack_id":1, "server_id":"groot"}, #10.40.39.31
       "hulk": {"room_id":2, "rack_id":1, "server_id":"hulk"},#10.40.39.33
       "cris-X550VXK": {"room_id":2, "rack_id":3, "server_id":"cris-X550VXK"},
       "DESKTOP-V89UFRB.localdomain": {"room_id":2, "rack_id":3, "server_id":"DESKTOP-V89UFRB.localdomain"}}

while(True):
    currentTime=datetime.datetime.now()
    currentTime = currentTime.replace(second=0, microsecond=0)  #+ timedelta(hours=2) suma de 2h por error horas Spark, para poder coger resgistros actuales guardados con 2h mas

    initial_timestamp= currentTime-datetime.timedelta(minutes=20)
    end_timestamp= currentTime

    for metric_type in metrics:
        if metric_type=="utilization" or metric_type=="temperature": #temperature
            base_file = (PATH + settings.operative_system+"Data_Center"+settings.operative_system+"%s_train.csv") %metric_type
            aux_file  = (PATH + settings.operative_system+"Data_Center"+settings.operative_system+"%s_aux.csv") %metric_type
            for server in servers:
                room_id=dict[server]["room_id"]
                rack_id=dict[server]["rack_id"]
                server_id=dict[server]["server_id"]
                if server!="gamora":
                    # with open(aux_file, "wb") as aux, open(base_file, "r") as base:
                    #     csvWriter = csv.writer(aux)
                    #     csvReader = csv.reader(base)
                    #     for row in csvReader:
                    #         csvWriter.writerow(row)
                    df_base_file=pd.read_csv(base_file, sep=",")
                    df_aux= df_base_file
                    df_aux.to_csv(aux_file, index=False)
                    training_file=aux_file
                else:
                    training_file=base_file
                print(server_id)
                print(initial_timestamp)
                print(end_timestamp)
                new_data = databaseallData.select_MAXmetric_and_date(room_id, rack_id, server_id, initial_timestamp, end_timestamp, metric_type,"Core")
                print("anomalie: "+str(new_data))
                # new_lines=len(new_data)
                # df = pd.read_csv(training_file, sep=",")
                # df_drop=df.drop(df.index[2:(new_lines + 2)])
                # lines=len(df_drop)+1
                # with open(base_file) as base:
                #     lines = sum(1 for line in base) - 3
                # df_drop.to_csv(training_file, index=False)
                # for record in new_data.rdd.collect():
                # print(row["new_date"])
                # print(row["max(metric_value)"])
                model= createModel(metric_type)
                for record in new_data.rdd.collect():
                    record=pd.DataFrame([record])
                    record=record.rename(index=str, columns={0: "c0", 1: "c1"})
                    df_train = pd.read_csv(training_file, sep=",")
                    df_train = pd.concat([df_train, record])
                    df_drop = df_train.drop(df_train.index[0:1])
                    df_drop.to_csv(training_file, index=False)
                    print(df_drop)
                    anomaly, timestamp = runAnomaly(model, df_drop)
                    print(anomaly)
                    print(timestamp)
                    print("real: "+str(end_timestamp))
                    if anomaly==1:
                        anomaly=True
                    else: anomaly=False
                    daoServerStatus.insert_metric(room_id, rack_id, server_id, timestamp, temperature_anomaly=anomaly, energy_anomaly=False,
                                      utilization_anomaly=False) #timestamp -> end_timestamp
                    # with open(training_file, "a") as train:
                    #     csvWriter = csv.writer(train)
                    #     csvWriter.writerow(record)
                    #     anomaly=runAnomaly(metric_type, training_file) #, lines
                    #     # csvWriter.writerow(["timestamp", "value", "anomaly_score", "anomaly_likehood", "label"])
                    #     # actualizar cassandra
                    #     if metric_type=="temperature":
                    #         databaseallData.update_temperature_anomaly(anomaly, room_id, rack_id, server_id, timestamp)
                    #     print("write in csv and pass anomaly_model")
                    #     print(anomaly)
                    #     # lines += 1
    time.sleep(1200) #time in seconds