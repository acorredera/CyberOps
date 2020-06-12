from pprint import pprint
from nupic.frameworks.opf.common_models.cluster_params import (getScalarMetricWithTimeOfDayAnomalyParams)
import importlib
import sys
import csv
import datetime
from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.model_factory import ModelFactory
import nupic_anomaly_output2
import matplotlib as plot

GYM_NAME = "/home/marta/PycharmProjects/CYBEROPS/MAQUINAS/Labeled_data/ec2_disk_write_bytes_c0d644_labeled (copia).csv"
DATA_DIR = "."
MODEL_PARAMS_DIR = "/home/marta/PycharmProjects/untitled/Files/MAQUINAS/model_params.py"
# '7/2/10 0:00'
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

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

params = getScalarMetricWithTimeOfDayAnomalyParams(
  metricData=[0], # just dummy data unless you want to send in some real data here
  minVal=0,
  maxVal=100,
  minResolution=0.001, # you may need to tune this
  tmImplementation="cpp")

# Here, you can print out the params cause its is just a dict, and change
# them to suit your needs. Here, I'll just print them out so you can see them:
pprint(params)
# Now use these params to create a model
model = createModel(params["modelConfig"])


# # Open the file to loop over each row
# with open ("/home/marta/PycharmProjects/CYBEROPS/MAQUINAS/Labeled_data/ec2_disk_write_bytes_c0d644_labeled (copia).csv") as fileIn:
#   reader = csv.reader(fileIn)
#   # The first three rows are not data, but we'll need the field names when
#   # passing data into the model.
#   headers = reader.next()
#   reader.next()
#   reader.next()
#
#   for record in reader:
#     # Create a dictionary with field names as keys, row values as values.
#     modelInput = dict(zip(headers, record))
#     # Convert string consumption to float value.
#     modelInput["c1"] = float(modelInput["c1"])
#     # Convert timestamp string to Python datetime.
#     modelInput["c0"] = datetime.datetime.strptime(
#       modelInput["c0"], "%Y-%m-%d %H:%M:%S"
#     )
#     # Push the data into the model and get back results.
#     result = model.run(modelInput)
#
#     print(result)

# def getModelParamsFromName(gymName):
#   """
#   Given a gym name, assumes a matching model params python module exists within
#   the model_params directory and attempts to import it.
#   :param gymName: Gym name, used to guess the model params module name.
#   :return: OPF Model params dictionary
#   """
#   importName = "model_params.%s_model_params" % (
#     gymName.replace(" ", "_").replace("-", "_")
#   )
#   print "Importing model params from %s" % importName
#   try:
#     importedModelParams = importlib.import_module(importName).MODEL_PARAMS
#   except ImportError:
#     raise Exception("No model params exist for '%s'. Run swarm first!"
#                     % gymName)
#   return importedModelParams


def runIoThroughNupic(inputData, model, GYM_NAME, plot):
  """
  Handles looping over the input data and passing each row into the given model
  object, as well as extracting the result object and passing it into an output
  handler.
  :param inputData: file path to input data CSV
  :param model: OPF Model object
  :param gymName: Gym name, used for output handler naming
  :param plot: Whether to use matplotlib or not. If false, uses file output.
  """
  inputFile = open(inputData, "rb")
  csvReader = csv.reader(inputFile)
  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  shifter = InferenceShifter()
  if plot:
    output = nupic_anomaly_output2.NuPICPlotOutput(GYM_NAME)
  else:
    output = nupic_anomaly_output2.NuPICFileOutput(GYM_NAME)

  counter = 0
  for row in csvReader:
    counter += 1
    if (counter % 100 == 0):
      print "Read %i lines..." % counter
    timestamp = datetime.datetime.strptime(row[0], DATE_FORMAT)
    value = float(row[1])
    result = model.run({
      "c0": timestamp,
      "c1": value
    })
    print(result)

    if plot:
      result = shifter.shift(result)

    prediction = result.inferences["anomalyScore"] #multiStepBestPredictions
    anomalyScore = result.inferences["anomalyScore"]
    output.write(timestamp, value, prediction, anomalyScore)

  inputFile.close()
  output.close()

runIoThroughNupic(GYM_NAME, model, GYM_NAME, plot)

# from nupic.algorithms import anomaly_likelihood
# anomalyLikelihood = self.anomalyLikelihoodHelper.anomalyProbability(
#         value, anomalyScore, timestamp
#       )
#       outputRow = [timestamp, value, predicted, anomalyScore, anomalyLikelihood]



# def runModel(gymName, plot=False):
#   """
#   Assumes the gynName corresponds to both a like-named model_params file in the
#   model_params directory, and that the data exists in a like-named CSV file in
#   the current directory.
#   :param gymName: Important for finding model params and input CSV file
#   :param plot: Plot in matplotlib? Don't use this unless matplotlib is
#   installed.
#   """
#   print "Creating model from %s..." % gymName
#   model = createModel(getModelParamsFromName(gymName))
#   inputData = "%s/%s.csv" % (DATA_DIR, gymName.replace(" ", "_"))
#   runIoThroughNupic(inputData, model, gymName, plot)
#
#
# if __name__ == "__main__":
#   plot = False
#   args = sys.argv[1:]
#   if "--plot" in args:
#     plot = True
# runModel(GYM_NAME, plot=plot)
#
#
# # # import json
# #
# # with open('data.json', 'w') as fp:
# #     json.dump(params, fp)
#
#
