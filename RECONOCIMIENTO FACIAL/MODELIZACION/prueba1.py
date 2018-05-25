import sys
sys.path.append("/home/marta/pyAudioAnalysis/")
import audioTrainTest as aT
aT.featureAndTrain(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
aT.fileClassification("data/doremi.wav", "svmSMtemp","svm")



# import os
# import sys
# lib_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'lib'))
# sys.path.append("/home/marta/pyAudioAnalysis/")
# import audioTrainTest


import sys
sys.path.insert(0, '/home/marta/pyAudioAnalysis/')

import audioTrainTest as aT
aT.featureAndTrain(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
aT.fileClassification("data/doremi.wav", "svmSMtemp","svm")
