import os,sys
if os.name=="nt":
    operative_system="\\"
else: operative_system="/"
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))).rsplit(operative_system,1)[0])
print (os.path.join(os.path.dirname(__file__)))
print (os.path.dirname(os.path.dirname(os.path.realpath(__file__))).rsplit(operative_system,1)[0])
import datetime
from nupic.frameworks.opf.common_models.cluster_params import (getScalarMetricWithTimeOfDayAnomalyParams)
from nupic.algorithms import anomaly_likelihood
from nupic.frameworks.opf.model_factory import ModelFactory
from datetime import timedelta

metrics={"temperature":{"min": 30, "max": 45},
      "utilization":{"min": 38, "max": 55}}

def createModel(metric):
    min = metrics[metric]["min"]
    max = metrics[metric]["max"]
    params = getScalarMetricWithTimeOfDayAnomalyParams(
        metricData=[0],  # just dummy data unless you want to send in some real data here
        minVal=min,
        maxVal=max,
        minResolution=0.001,  # you may need to tune this #0.001
        tmImplementation="cpp")  # cpp
    model = ModelFactory.create(params["modelConfig"])
    model.enableInference({"predictedField": "c1"})
    return model

def runAnomaly(model,training_file, anomaly_threshold=0.9): #lines
    anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(historicWindowSize=144) #, learningPeriod=1152
    for index, record in training_file.iterrows():
        modelInput = {"c0": str(record[0]), "c1": record[1]}
        value = modelInput["c1"]
        modelInput["c0"] = datetime.datetime.strptime(modelInput["c0"], "%Y-%m-%d %H:%M:%S") #- timedelta(hours=2) #resta de 2h por error horas Spark, para devolver a su hora original
        timestamp = modelInput["c0"]
        result = model.run(modelInput)
        anomalyScore = result.inferences['anomalyScore']
        anomalyLikelyhood2 = anomalyLikelihood.anomalyProbability(value, anomalyScore, timestamp)
    print("anScoreANTERIOR: ", anomalyScore, ", anLikelihood2ANTERIOR: ", anomalyLikelyhood2)
    #version sin for:
    # modelInput = {"c0": "", "c1": 0}
    # timestamp = training_file._values[-1][0]
    # value = training_file._values[-1][1]
    # modelInput["c0"] = datetime.datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")
    # modelInput["c1"] = value
    # result = model.run(modelInput)
    # anomalyScore = result.inferences['anomalyScore']
    # anomalyLikelyhood3 = anomalyLikelihood.anomalyProbability(value, anomalyScore, modelInput["c0"])
    # print("anScoreSINFOR: ", anomalyScore, ", anLikelihood2SINFOR: ", anomalyLikelyhood3)

    #RESULTADOS: VARIAN PERO NO DE MANERA MUY SIGNIFICATIVA DIRIA, LO QUE PODEMOS HACER ES REDUCIR EL NUMERO DE TRAINING DATA QUE SE GUARDAN EN LOS FICHEROS PARA LAS PRUEBAS DEL PROTOTIO,
    # PODEMOS PONER UNA VENTANA DESIZANTE E IR BORRANDO DATOS APRA QUE NO SE HAGA ENORME EL CSV Y ADEMAS QUE PODEMOS SUPONER QUE LO NORMAL ES LO QUE OCURRE MAS RECIENTEMENTE, Y SI HA HABIDO ALGUNA ANOMALIA SE HABRIA INFORMADO EN CASOS ANTERIORES
    # (ESTO SI QUEREMOS PORUQE PARA LAS PRUEBAS NO NOS PEDIRAN NADA)

    # ('anScoreANTERIOR: ', 0.0, ', anLikelihood2ANTERIOR: ', 0.54721270106335917)
    # ('anScoreSINFOR: ', 0.0, ', anLikelihood2SINFOR: ', 0.56844323632835281)
    #
    # ('anScoreANTERIOR: ', 0.25, ', anLikelihood2ANTERIOR: ', 0.9821310895610571)
    # ('anScoreSINFOR: ', 0.075000003, ', anLikelihood2SINFOR: ', 0.98621086501000965)
    #
    # ('anScoreANTERIOR: ', 0.0, ', anLikelihood2ANTERIOR: ', 0.91637213745567192)
    # ('anScoreSINFOR: ', 0.0, ', anLikelihood2SINFOR: ', 0.91637213745567192)
    #
    # ('anScoreANTERIOR: ', 0.0, ', anLikelihood2ANTERIOR: ', 0.91637213745567192)
    # ('anScoreSINFOR: ', 0.0, ', anLikelihood2SINFOR: ', 0.91637213745567192)
    #
    # ('anScoreANTERIOR: ', 0.0, ', anLikelihood2ANTERIOR: ', 0.9999107159364139)
    # ('anScoreSINFOR: ', 0.0, ', anLikelihood2SINFOR: ', 0.9999107159364139)
    if anomalyLikelyhood2 > anomaly_threshold:
        anomaly=1
    else: anomaly=0
    return anomaly, timestamp