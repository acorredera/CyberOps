# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
A simple client to create a HTM anomaly detection model for hotgym.
The script prints out all records that have an abnormally high anomaly
score.
"""

import csv
import datetime
import logging
from nupic.frameworks.opf.common_models.cluster_params import (getScalarMetricWithTimeOfDayAnomalyParams)
from nupic.algorithms import anomaly_likelihood

from pkg_resources import resource_filename

from nupic.frameworks.opf.model_factory import ModelFactory

# import model_params

_LOGGER = logging.getLogger(__name__)

_INPUT_DATA_FILE = "/home/marta/PycharmProjects/CYBEROPS/MAQUINAS/Labeled_data/ec2_cpu_utilization_5f5533_labeled (copia).csv"
_OUTPUT_PATH = "anomaly_scores.csv"

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

def runHotgymAnomaly():
    params = getScalarMetricWithTimeOfDayAnomalyParams(
        metricData=[0],  # just dummy data unless you want to send in some real data here
        minVal=60,
        maxVal=75,
        minResolution=0.001,  # you may need to tune this #0.001
        tmImplementation="cpp") #cpp
    model = createModel(params["modelConfig"])
    # model.enableInference({'predictedField': 'c1'})
    with open (_INPUT_DATA_FILE) as fin:
        reader = csv.reader(fin)
        csvWriter = csv.writer(open(_OUTPUT_PATH,"wb"))
        csvWriter.writerow(["timestamp", "value", "anomaly_score", "anomaly_likehood", "label"])
        headers = reader.next()
        reader.next()
        reader.next()
        anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(historicWindowSize=288) #, learningPeriod=1152
        for i, record in enumerate(reader, start=1):
            modelInput = dict(zip(headers, record))
            modelInput["c1"] = float(modelInput["c1"])
            value = modelInput["c1"]
            modelInput["c0"] = datetime.datetime.strptime(modelInput["c0"], "%Y-%m-%d %H:%M:%S")
            timestamp = modelInput["c0"]
            result = model.run(modelInput)
            anomalyScore = result.inferences['anomalyScore']
            anomalyLikelyhood2 = anomalyLikelihood.anomalyProbability(value, anomalyScore, timestamp)
            csvWriter.writerow([timestamp, value, anomalyScore, anomalyLikelyhood2, modelInput["label"]])

            if anomalyScore > _ANOMALY_THRESHOLD:
                _LOGGER.info("Anomaly detected at [%s]. Anomaly score: %f.", result.rawInput["c0"], anomalyScore)

    print ("Anomaly scores have been written to " + _OUTPUT_PATH)

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
runHotgymAnomaly()