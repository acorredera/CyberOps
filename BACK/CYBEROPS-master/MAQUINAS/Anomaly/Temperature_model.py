import csv
import datetime
import logging
from nupic.frameworks.opf.common_models.cluster_params import (getScalarMetricWithTimeOfDayAnomalyParams)
from nupic.algorithms import anomaly_likelihood
import NewsCore.dao.MaquinasDAOImplementation as DAO_Maquinas
import settings
from pkg_resources import resource_filename

from nupic.frameworks.opf.model_factory import ModelFactory

# Archivo de entrenamiento
# Temperatura (en F) ---> /home/marta/PycharmProjects/CYBEROPS/MAQUINAS/Labeled_data/ambient_temperature_system_failure_labeled.csv

# import model_params
ip_cassandra = settings.ip_DCOS_cassandra
keyspace = settings.keyspace_cassandra

# databaseallData = DAO_Maquinas.Maquinas_ProteusDAOImplementation()
# databaseallData.create_session(ip_cassandra)
# databaseallData.set_logger()
# databaseallData.load_keyspace(keyspace)

# new_data = DAO_Maquinas.select_utilization()
# currentTime = datetime.datetime.now()
# currentTime= currentTime - datetime.timedelta(hours=24)

_LOGGER = logging.getLogger(__name__)


# _TRAINING_DATA_FILE = "/home/marta/PycharmProjects/CYBEROPS/MAQUINAS/Labeled_data/ambient_temperature_system_failure_labeled (copia).csv"
_INPUT_DATA_FILE = "celsius_temperature_train.csv"
new_data = ["2019-04-24 14:22:00", 63]

with open (_INPUT_DATA_FILE, 'a') as f:
    csvWriter = csv.writer(f)
    csvWriter.writerow(new_data)

with open(_INPUT_DATA_FILE) as f:
    lines=sum(1 for line in f)-3

# with open (_TRAINING_DATA_FILE, "r") as train, open(_INPUT_DATA_FILE, 'w') as new:
#     csvReader=csv.reader(train)
#     csvWriter = csv.writer(new)
#     new_data = ["2014-02-28 14:22:00", 45]
#     # flag = 0
#     for row in csvReader:
#         csvWriter.writerow(row)
#     # for i in new_data:
#     #     if flag==0:
#     #         i=str(i)
#     #         data = i[0:-7]
#     #         flag=1
#     #     else:
#     #         value=i
#     # row=data+","+str(value)
#     csvWriter.writerow(new_data)
#
# with open(_INPUT_DATA_FILE) as f:
#     lines=sum(1 for line in f)-3

_OUTPUT_PATH = "anomaly_scores_temperature.csv"

_ANOMALY_THRESHOLD = 0.9


def createModel(modelParams):
  """
  Given a model params dictionary, create a CLA Model. Automatically enables
  inference for kw_energy_consumption.
  :param modelParams: Model params dict
  :return: OPF Model object
  """
  model = ModelFactory.create(modelParams)
  model.enableInference({"predictedField": "c1"})
  return model

def runAnomaly():
    params = getScalarMetricWithTimeOfDayAnomalyParams(
        metricData=[0],  # just dummy data unless you want to send in some real data here
        minVal=30,
        maxVal=45,
        minResolution=0.001,  # you may need to tune this #0.001
        tmImplementation="cpp") #cpp
    model = createModel(params["modelConfig"])
    # model.enableInference({'predictedField': 'c1'})
    with open (_INPUT_DATA_FILE) as fin:
        reader = csv.reader(fin)
        csvWriter = csv.writer(open(_OUTPUT_PATH,"a"))
        # csvWriter.writerow(["timestamp", "value", "anomaly_score", "anomaly_likehood", "label"])
        headers = reader.next()
        reader.next()
        reader.next()
        anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(historicWindowSize=1152) #, learningPeriod=1152
        for i, record in enumerate(reader, start=1):
            modelInput = dict(zip(headers, record))
            modelInput["c1"] = float(modelInput["c1"])
            value = modelInput["c1"]
            modelInput["c0"] = datetime.datetime.strptime(modelInput["c0"], "%Y-%m-%d %H:%M:%S")
            timestamp = modelInput["c0"]
            result = model.run(modelInput)
            anomalyScore = result.inferences['anomalyScore']
            anomalyLikelyhood2 = anomalyLikelihood.anomalyProbability(value, anomalyScore, timestamp)
            if i == lines:
                if anomalyLikelyhood2 > _ANOMALY_THRESHOLD:
                    _LOGGER.info("Anomaly detected at [%s]. Anomaly score: %f.", result.rawInput["c0"], anomalyScore)
                    anomaly=1
                else: anomaly=0
                csvWriter.writerow([timestamp, value, anomalyScore, anomalyLikelyhood2, anomaly])
                return anomaly
            # else:
            #     csvWriter.writerow([timestamp, value, anomalyScore, anomalyLikelyhood2, modelInput["label"]])
    print ("Anomaly scores have been written to " + _OUTPUT_PATH)

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
runAnomaly()