import os
from builtins import type
import UTILS.dataSaver as saver
import numpy as np
import UTILS.dataLoader as loader
#import settings.Constraints as const
import matplotlib as mlp
import settings
#import UTILS.setCreator as filesCreator
mlp.use('Agg')
# if(const.display):
    #import matplotlib.pyplot as plt
#from PIL import Image


#SPECTRUM NORMALIZATION
def addData2Lists(newData, dataList,newName, nameList):
    if (dataList == []):
        dataList = newData
    else:
        dataList = np.hstack([dataList, newData])
    nameList.append(newName)
    return dataList, nameList

def get_statistics_from_data(data, axis_attributes=0):
    avg = np.nanmean(data, axis=axis_attributes)
    std = np.nanstd(data, axis=axis_attributes)
    min = np.nanmin(data, axis=axis_attributes)
    max = np.nanmax(data, axis=axis_attributes)
    return avg, std, min, max

def getNormalizedData(data2Norm,minimumValue,maximumValue,newSupValue=255,newInfValue=0):
    if(maximumValue==minimumValue):
        if(maximumValue==0):
            maximumValue = 1
        normalized_data = (data2Norm/maximumValue)*newInfValue #remove the relevance that these fragments could have
    else:
        normalized_data = (((data2Norm - minimumValue) / (maximumValue - minimumValue)) * (newSupValue - newInfValue)) + newInfValue
        # check if some number surphase the maximum and minimum (newSupValue & newInfValue)
    normalized_data[normalized_data > newSupValue] = newSupValue
    normalized_data[normalized_data < newInfValue] = newInfValue
    return normalized_data

def getZscoreData(data2Zscore, meanValue, stdValue,zscoreThreshold=None, withLabels=False):
    if(isinstance(stdValue, float)):
        if(stdValue==0):
            stdValue==1
        elif(len(stdValue.shape)>1):
            stdValue[stdValue==0.0] = 1.0
    elif(isinstance(stdValue, (list, tuple, np.ndarray))):
        stdValue[stdValue==0.0]=1.0
    if(withLabels):
        labels = data2Zscore[:,-1]
        data2Zscore = data2Zscore[:,:-1]
    zscore_data = (data2Zscore - meanValue) / (stdValue)
    if (not zscoreThreshold == None):
        zscore_data[zscore_data > zscoreThreshold] = zscoreThreshold
        zscore_data[zscore_data < (-1 * zscoreThreshold)] = (-1 * zscoreThreshold)
    if(withLabels):
        zscore_data = np.column_stack((zscore_data,labels))
    return zscore_data

def getZscoreNormData(data2Zscore, meanValue, stdValue, minimumValue, maximumValue, newSupValue=255, newInfValue=0, zscoreThreshold=None):
    zscore_data = getZscoreData(data2Zscore, meanValue, stdValue,zscoreThreshold=zscoreThreshold)
    if(minimumValue==[]): #if minimum and maximum are null, it means that we are in the training set, so we can obtain min/max values
        if(isinstance(meanValue, np.ndarray)):#normalization type4
            minimumValue = zscore_data.min(1)
            maximumValue = zscore_data.max(1)
            minimumValue = np.reshape(minimumValue, [-1, 1])
            maximumValue = np.reshape(maximumValue, [-1, 1])
        else:#normalization type3
            minimumValue = zscore_data.min()
            maximumValue = zscore_data.max()
    zscoreNormalized_data = getNormalizedData(zscore_data, minimumValue, maximumValue, newSupValue=newSupValue, newInfValue=newInfValue)
    return zscoreNormalized_data, minimumValue, maximumValue

def saveNormalizedFunctionals4allUsers(listOfUsersNames, sizeSingleUserSamples, normalizedFunctionalsOfAllUsers, outputPath, matInternalName ='new_functionals', text2add = ''):
    for k in range(1, len(listOfUsersNames) + 1):
        indexStart = (k - 1) * sizeSingleUserSamples
        indexEnd = (k) * sizeSingleUserSamples
        new_functionals = normalizedFunctionalsOfAllUsers[indexStart: indexEnd, :]
        if('.arff' in listOfUsersNames[k - 1]):
            nameNormalizedFile = listOfUsersNames[k - 1].split('.arff')[0]
        else:
            nameNormalizedFile = listOfUsersNames[k - 1]
        new_functionals = new_functionals.transpose() #guardamos el traspuesta con features como filas para que la creación de imágenes sea más fácil
        savePath = "".join([outputPath, nameNormalizedFile,text2add])
        saver.saveMatrixAsMat(new_functionals, savePath, matInternalName)

def saveNormalizedFiles4AllUsers(listOfUsersNames, sizeSingleUserSpectrum, normalizedSpectrumOfAllUsers, outputPath, matInternalName ='fixed_spectrum_matrix'):
    for k in range(1, len(listOfUsersNames) + 1):
        indexStart = (k - 1) * sizeSingleUserSpectrum[1]
        indexEnd = (k) * sizeSingleUserSpectrum[1]
        new_spectrum = normalizedSpectrumOfAllUsers[:, indexStart: indexEnd]
        savePath = "".join([outputPath, listOfUsersNames[k - 1]])
        saver.saveMatrixAsMat(new_spectrum, savePath, matInternalName)

def normalizationPerUser(input_file_path,typeOfNormalization,newSupValue,newInfValue,outputPath, text2add='_norm0255.mat', name = 'fixed_spectrum_matrix', matInternalName = 'fixed_spectrum_matrix'):
    functionals_data = (loader.loadMatFiles(input_file_path))[name]
    if (typeOfNormalization == 1):  # total of the whole audio file
        minimumValue = functionals_data.min()
        maximumValue = functionals_data.max()
    elif (typeOfNormalization == 2):  # min/max per frequency (160, 224 or 672 depending on number of FFT points)
        minimumValue = functionals_data.min(1)
        maximumValue = functionals_data.max(1)
        minimumValue = np.reshape(minimumValue, [-1, 1])
        maximumValue = np.reshape(maximumValue, [-1, 1])
    else:
        print("Error in type of normalization")
        return -1
    new_spectrum = getNormalizedData(functionals_data,minimumValue,maximumValue,newSupValue,newInfValue)
    #save data
    output_spectrum_name = (input_file_path.split('/')[-1].split('.mat')[0]) + text2add
    savePath = "".join([outputPath, output_spectrum_name])
    saver.saveMatrixAsMat(new_spectrum, savePath, matInternalName)
    return minimumValue,maximumValue

def normPercentilePerUser(input_file_path,typeOfNormalization,newSupValue,newInfValue,outputPath, text2add='_norm0255.mat', percentile = 25):
    functionals_data = (loader.loadMatFiles(input_file_path))['fixed_spectrum_matrix']
    minimumValue=[]
    maximumValue=[]
    percentileValue = np.percentile(functionals_data, percentile)
    functionals_data[functionals_data < percentileValue] = percentileValue
    if (typeOfNormalization == 5):  # total of the whole audio file
        minimumValue = functionals_data.min()
        maximumValue = functionals_data.max()
    else:
        print("Error in type of normalization")
        return -1
    output_spectrum_name = (input_file_path.split('/')[-1].split('.mat')[0]) + text2add
    new_spectrum = getNormalizedData(functionals_data, minimumValue, maximumValue, newSupValue, newInfValue)
    #img = Image.fromarray(new_spectrum[:, :].astype('uint8'), 'L')
    # imgOutputPath = '/home/cris/PycharmProjects/MasterProject/data/RESULTS/otrasPruebas/norm.png'
    # img.save(imgOutputPath, "PNG")
    savePath = "".join([outputPath, output_spectrum_name])
    saver.saveMatrixAsMat(new_spectrum, savePath, 'fixed_spectrum_matrix')

def zscorePerUser(input_file_path,typeOfNormalization,newSupValue,newInfValue,outputPath,zscoreThreshold=None, text2add='_zscore0255.mat', name = 'fixed_spectrum_matrix', newName ='fixed_spectrum_matrix' ):
    functionals_data = (loader.loadMatFiles(input_file_path))[name]
    meanValue = []
    stdValue = []
    minimumValue = []
    maximumValue = []
    if (typeOfNormalization == 3):  # total of the whole audio file
        meanValue = functionals_data.mean()
        stdValue = functionals_data.std()
    elif (typeOfNormalization == 4):  # min/max per frequency (224 or 672 depending on number of FFT points)
        meanValue = functionals_data.mean(1)
        stdValue = functionals_data.std(1)
        meanValue = np.reshape(meanValue, [-1, 1])
        stdValue = np.reshape(stdValue, [-1, 1])
    else:
        print("Error in type of normalization")
        return -1
    output_spectrum_name = (input_file_path.split(settings.operative_system)[-1].split('.mat')[0]) + text2add
    new_spectrum,minimumValue,maximumValue = getZscoreNormData(functionals_data, meanValue, stdValue, minimumValue, maximumValue, newSupValue, newInfValue, zscoreThreshold)
    savePath = "".join([outputPath, output_spectrum_name])
    saver.saveMatrixAsMat(new_spectrum, savePath, newName)
    return meanValue, stdValue, minimumValue, maximumValue


def normPercentileZscorePerUser(input_file_path, typeOfNormalization, newSupValue, newInfValue, outputPath,text2add, percentile):
    functionals_data = (loader.loadMatFiles(input_file_path))['fixed_spectrum_matrix']
    output_spectrum_name = (input_file_path.split(settings.operative_system)[-1].split('.mat')[0]) + text2add
    #functionals_data_modi = np.reshape(functionals_data, [-1])
    # plt.figure()
    # plt.hist(functionals_data_modi, bins = 200)
    #plt.show()
    #nameFig = "".join([outputPath, (input_file_path.split('/')[-1].split('.mat')[0]), '_distribution.png'])
    #plt.savefig(nameFig)

    #zscore
    if (typeOfNormalization == 7):  # total of the whole audio file
        meanValue = functionals_data.mean()
        stdValue = functionals_data.std()
    else:
        print("Error in type of normalization")
        return -1
    zscore_data = getZscoreData(functionals_data, meanValue, stdValue) #no debería importar
    #zscore_data_modi = np.reshape(zscore_data, [-1])
    # plt.figure()
    # plt.hist(zscore_data_modi, bins = 200)
    #plt.show()
    #percentile
    percentileValue = np.percentile(zscore_data, percentile)
    zscore_data[zscore_data < percentileValue] = percentileValue

    # zscore_data_modiP = np.reshape(zscore_data, [-1])
    # plt.figure()
    # plt.hist(zscore_data_modiP, bins=200)
    # plt.show()
    #normalization
    if (typeOfNormalization == 7):  # total of the whole audio file
        minimumValue = zscore_data.min()
        maximumValue = zscore_data.max()
    else:
        print("Error in type of normalization")
        return -1
    new_spectrum = getNormalizedData(zscore_data, minimumValue, maximumValue, newSupValue=newSupValue, newInfValue=newInfValue)
    # img = Image.fromarray(new_spectrum[:, :].astype('uint8'), 'L')
    # imgOutputPath = '/home/cris/PycharmProjects/MasterProject/data/RESULTS/otrasPruebas/zscore.png'
    # img.save(imgOutputPath, "PNG")
    # plt.figure()
    # plt.imshow(new_spectrum)
    # plt.show()

    #new_spectrum_modi = np.reshape(new_spectrum, [-1])
    # plt.figure()
    # plt.hist(new_spectrum_modi, bins = 200)
    #plt.show()
    savePath = "".join([outputPath, output_spectrum_name])
    saver.saveMatrixAsMat(new_spectrum, savePath, 'fixed_spectrum_matrix')
    return meanValue, stdValue, minimumValue, maximumValue

def imageNormalization(dataId,inputImageFolder,outputPath, typeOfNormalization, newInfValue=0, newSupValue=255,
                       obtainTest=False,text2add="_norm0255",zscoreThreshold=None,percentileNormalizationType4=25):
    input_file_path = "".join([inputImageFolder, dataId])
    # create and save normalized data (just normalization per user is possible)
    if (typeOfNormalization == 1 or typeOfNormalization == 2):
        normalizationPerUser(typeOfNormalization=typeOfNormalization, input_file_path=input_file_path,
                             newSupValue=newSupValue, newInfValue=newInfValue, outputPath=outputPath,
                             text2add="".join([text2add, '.mat']))
    elif (typeOfNormalization == 3 or typeOfNormalization == 4):
        zscorePerUser(input_file_path, typeOfNormalization, newSupValue, newInfValue, outputPath,
                      zscoreThreshold=zscoreThreshold, text2add=text2add)
    elif (typeOfNormalization == 5):
        normPercentilePerUser(input_file_path, typeOfNormalization, newSupValue, newInfValue, outputPath,
                              text2add="".join([text2add, '.mat']), percentile=percentileNormalizationType4)
    elif (typeOfNormalization == 7):
        normPercentileZscorePerUser(input_file_path, typeOfNormalization, newSupValue, newInfValue,
                                    outputPath, text2add, percentile=percentileNormalizationType4)
    else:
        print('Incorrect normalization!!')

def dataNormalization(dataId,inputDataFolder,outputPath, typeOfNormalization, newInfValue=0, newSupValue=255,
                       obtainTest=False,text2add="_norm0255",zscoreThreshold=None,percentileNormalizationType4=25):
    input_file_path = "".join([inputDataFolder, dataId])
    # create and save normalized data (just normalization per user is possible)
    if (typeOfNormalization == 1 or typeOfNormalization == 2):
        normalizationPerUser(typeOfNormalization=typeOfNormalization, input_file_path=input_file_path,
                             newSupValue=newSupValue, newInfValue=newInfValue, outputPath=outputPath,
                             text2add="".join([text2add, '.mat']))
    elif (typeOfNormalization == 3 or typeOfNormalization == 4):
        zscorePerUser(input_file_path, typeOfNormalization, newSupValue, newInfValue, outputPath,
                      zscoreThreshold=zscoreThreshold, text2add=text2add)
    elif (typeOfNormalization == 5):
        normPercentilePerUser(input_file_path, typeOfNormalization, newSupValue, newInfValue, outputPath,
                              text2add="".join([text2add, '.mat']), percentile=percentileNormalizationType4)
    elif (typeOfNormalization == 7):
        normPercentileZscorePerUser(input_file_path, typeOfNormalization, newSupValue, newInfValue,
                                    outputPath, text2add, percentile=percentileNormalizationType4)
    else:
        print('Incorrect normalization!!')
    return 1

def spectrumNormalization(input_path_FFTspectrums, outputPath,listTrainingFilesNames,listValidationFilesNames,typeOfNormalization= 1,
                          perUserOrAllUsers="perUser", newInfValue=0, newSupValue=255, obtainTest=False, text2add="_norm0255",
                          saveMinMaxPickle=False, zscoreThreshold=None, percentileNormalizationType4=25):
    listSpectrumFiles = os.listdir(input_path_FFTspectrums)
    trainingData = []
    trainingNames = []
    validationData = []
    validationNames = []
    testData = []
    testNames = []
    minimumValue = [] #mean or min
    maximumValue = []# std or max
    meanValue = []
    stdValue = []
    for spectrum_file in listSpectrumFiles:
        nameSpectrum = spectrum_file.split(".")
        output_spectrum_name = "".join([nameSpectrum[0],text2add,'.mat'])
        if(".mat" in spectrum_file):
            input_file_path = "".join([input_path_FFTspectrums, spectrum_file])
            #normalization
            if(perUserOrAllUsers=="perUser"): # if normalization is per user, we obtain the normalized value of all the users(train,dev&test)
                #create and save normalized data
                if(typeOfNormalization==1 or typeOfNormalization==2):
                    normalizationPerUser(typeOfNormalization=typeOfNormalization, input_file_path=input_file_path, newSupValue=newSupValue,newInfValue= newInfValue, outputPath=outputPath,
                                         text2add="".join([text2add,'.mat']))
                elif(typeOfNormalization==3 or typeOfNormalization==4):
                    zscorePerUser(input_file_path, typeOfNormalization, newSupValue, newInfValue, outputPath,
                                  zscoreThreshold=zscoreThreshold, text2add=text2add)
                elif(typeOfNormalization==5):
                    normPercentilePerUser(input_file_path, typeOfNormalization, newSupValue, newInfValue, outputPath,
                                          text2add="".join([text2add, '.mat']), percentile=percentileNormalizationType4)
                elif(typeOfNormalization==7):
                    normPercentileZscorePerUser(input_file_path, typeOfNormalization, newSupValue, newInfValue,
                                                outputPath, text2add, percentile=percentileNormalizationType4)

            elif(perUserOrAllUsers=="allUsers"):
                #create array with all the normalized data but not save it until the end
                functionals_data = (loader.loadMatFiles(input_file_path))['fixed_spectrum_matrix']
                sizeSpectrum = np.shape(functionals_data)
                recordName = spectrum_file.split("_spectrum.mat")
                if(recordName[0] in listTrainingFilesNames):
                    trainingData,trainingNames = addData2Lists(functionals_data, trainingData, output_spectrum_name, trainingNames)
                elif(recordName[0] in listValidationFilesNames):
                    validationData,validationNames = addData2Lists(functionals_data, validationData, output_spectrum_name, validationNames)
                elif(("test" in output_spectrum_name)):
                    if(obtainTest):
                        testData,testNames = addData2Lists(functionals_data, testData, output_spectrum_name, testNames)
                else:
                    print("file not in training or validation: ",recordName[0])
            else:
                print('Not valid option for normalization, choose: perUser or allUsers')
                return -1
    if (perUserOrAllUsers=="allUsers"): #FALTA IMPLEMENTAR NORMALIZATION 5 Y 6 PARA TODOS LOS USUARIO, POR AHORA SÓLO PER USER
        #SCALING
        if(typeOfNormalization==1 or typeOfNormalization==2):
            if(typeOfNormalization==1):
                minimumValue = trainingData.min()
                maximumValue = trainingData.max()
            elif(typeOfNormalization==2):
                minimumValue = trainingData.min(1)
                maximumValue = trainingData.max(1)
                minimumValue = np.reshape(minimumValue, [-1, 1])
                maximumValue = np.reshape(maximumValue, [-1, 1])
            trainingData = np.array(trainingData)
            validationData = np.array(validationData)
            trainingDataNorm = getNormalizedData(trainingData, minimumValue, maximumValue, newSupValue, newInfValue)
            validationDataNorm = getNormalizedData(validationData, minimumValue, maximumValue, newSupValue, newInfValue)
            if (obtainTest):
                testData = np.array(testData)
                testDataNorm = getNormalizedData(testData, minimumValue, maximumValue, newSupValue, newInfValue)
        #ZSCORE + SCALING
        elif(typeOfNormalization==3 or typeOfNormalization==4):
            if (typeOfNormalization == 3):  # total of the whole audio file
                meanValue = trainingData.mean()
                stdValue = trainingData.std()
            elif (typeOfNormalization == 4):  # min/max per frequency (224 or 672 depending on number of FFT points)
                meanValue = trainingData.mean(1)
                stdValue = trainingData.std(1)
                meanValue = np.reshape(meanValue, [-1, 1])
                stdValue = np.reshape(stdValue, [-1, 1])
            trainingData = np.array(trainingData)
            validationData = np.array(validationData)
            trainingDataNorm,minimumValue,maximumValue = getZscoreNormData(trainingData, meanValue, stdValue, minimumValue, maximumValue, newSupValue, newInfValue, zscoreThreshold=zscoreThreshold)
            validationDataNorm,_,_ = getZscoreNormData(validationData, meanValue, stdValue, minimumValue, maximumValue, newSupValue, newInfValue, zscoreThreshold=zscoreThreshold)
            if (obtainTest):
                testData = np.array(testData)
                testDataNorm,_,_ = getZscoreNormData(testData, meanValue, stdValue, minimumValue, maximumValue, newSupValue, newInfValue, zscoreThreshold=zscoreThreshold)

        else:
            print("Error in type of normalization")
            return -1
        if(not trainingNames == []):
            saveNormalizedFiles4AllUsers(trainingNames, sizeSpectrum, trainingDataNorm, outputPath)
        if(not validationNames == []):
            saveNormalizedFiles4AllUsers(validationNames, sizeSpectrum, validationDataNorm, outputPath)
        if (obtainTest):
            saveNormalizedFiles4AllUsers(testNames, sizeSpectrum, testDataNorm, outputPath)

        if(saveMinMaxPickle):
            if(typeOfNormalization==1 or typeOfNormalization==2):
                matrix = np.array([minimumValue,maximumValue])
            elif(typeOfNormalization==3 or typeOfNormalization==4):
                matrix = np.array([meanValue, stdValue,minimumValue,maximumValue])
            saver.savePicklefromMatrix(outputPath, matrix, ('matrixTrainingAllUsersCase'+str(typeOfNormalization)))
    return meanValue, stdValue,minimumValue,maximumValue




# def functionalNormalization(trainingData, validationData, outputPath,listTrainingFilesNames,listValidationFilesNames,typeOfNormalization= 1,
#                           perUserOrAllUsers="perUser", newInfValue=0, newSupValue=255,withLabels = True,
#                           saveMinMaxPickle=True, zscoreThreshold=None, includeNormAsImg=True, text2add='norm0255', matInternalName='new_functionals'):
#     #listFunctionalFiles = os.listdir(input_path_functionals)
#     minimumValue = []  # mean or min
#     maximumValue = []  # std or max
#     meanValue = []
#     stdValue = []
#
#     # normalization
#     if (perUserOrAllUsers == "perUser"):  # if normalization is per user, we obtain the normalized value of all the users(train,dev&test)
#         # create and save normalized data
#         #zscore + scaling
#         if (typeOfNormalization == 2 or typeOfNormalization==1):
#             trainingDataNorm = stand_feat_perUserArray(trainingData, len(listTrainingFilesNames), withLabels=withLabels,
#                                                  zscoreThreshold=zscoreThreshold, typeOfNormalization=typeOfNormalization,
#                                                  includeNormAsImg=includeNormAsImg, normSupValue=newSupValue,normInfValue=newInfValue)
#             validationDataNorm = stand_feat_perUserArray(validationData, len(listValidationFilesNames), withLabels=withLabels,
#                                                                       zscoreThreshold=zscoreThreshold,
#                                                                       typeOfNormalization=typeOfNormalization,
#                                                                       includeNormAsImg=includeNormAsImg,
#                                                                       normSupValue=newSupValue,
#                                                                       normInfValue=newInfValue)
#         #scaling
#         elif (typeOfNormalization == 3):
#             trainingDataNorm = norm_feat_perUserArray(trainingData, len(listTrainingFilesNames), newSuperiorValue=newSupValue, newInferiorValue=newInfValue,
#                                                                    withLabels=withLabels)
#             validationDataNorm = norm_feat_perUserArray(validationData, len(listValidationFilesNames), newSuperiorValue=newSupValue, newInferiorValue=newInfValue,
#                                                                      withLabels=withLabels)
#         else:
#             print("Error in type of normalization")
#             return -1
#     elif (perUserOrAllUsers == "allUsers"):  # FALTA IMPLEMENTAR NORMALIZATION 5 Y 6 PARA TODOS LOS USUARIO, POR AHORA SÓLO PER USER
#         # ZSCORE + SCALING
#         if (typeOfNormalization == 2 or typeOfNormalization == 1):
#             trainingDataNorm,validationDataNorm = stand_feat_allUsers(trainingData, validationData,withLabels=withLabels,
#                                              zscoreThreshold=zscoreThreshold,
#                                              typeOfNormalization=typeOfNormalization,
#                                              includeNormAsImg=includeNormAsImg,
#                                              normSupValue=newSupValue,
#                                              normInfValue=newInfValue, saveData4Test=saveMinMaxPickle,pathData4Test=outputPath)
#
#         # scaling
#         elif (typeOfNormalization == 3):
#             trainingDataNorm,validationDataNorm, minimumValue, maximumValue = norm_feat_allUsersArray(trainingData, validationData,
#                                                                                                 newSuperiorValue=newSupValue, newInferiorValue=newInfValue,
#                                                                                                 withLabels=withLabels, saveData4Test=saveMinMaxPickle, pathData4Test=outputPath)
#     else:
#         print("Error in type of normalization")
#         return -1
#     if (not listTrainingFilesNames == []):
#         sizeFile = int(np.shape(trainingDataNorm)[0]/len(listTrainingFilesNames))
#         saveNormalizedFunctionals4allUsers(listTrainingFilesNames, sizeFile, trainingDataNorm, outputPath, text2add=text2add, matInternalName=matInternalName)
#     if (not listValidationFilesNames == []):
#         sizeFile = int(np.shape(validationDataNorm)[0]/len(listValidationFilesNames))
#         saveNormalizedFunctionals4allUsers(listValidationFilesNames, sizeFile, validationDataNorm, outputPath, text2add=text2add, matInternalName=matInternalName)
#
#     return meanValue, stdValue, minimumValue, maximumValue


#SAME VERSION THAT WE HAVE IN MATLAB BUT WITH PYTHON
#stand_app = 1 (online=allUsers); stand_app = 2(per user)
# def stand_feat_allUsers(concatenatedFiles_train, concatenatedFiles_test,typeOfNormalization=1, zscoreThreshold=None, withLabels=True,
#                         includeNormAsImg=True, normSupValue = 255, normInfValue = 0,saveData4Test=True, pathData4Test='', extraInfo=""): # other version: read from folder: inputFileNameTrain ="conc_train", inputFileNameTest ="conc_test",files2std=[]
#     '''
#         Standarization of features using zscore with info of all users in the training set.
#         :param concatenatedFiles_train:
#         :param concatenatedFiles_test:
#         :rtype: return standarized training and test sets.
#             '''
#     #concatenatedFiles_train = saver.importMatrixfromPICKLE(feat_path, inputFileNameTrain)
#     if(withLabels):
#         #train
#         labels_train = concatenatedFiles_train[:,-1]
#         labels_train = np.reshape(labels_train, [len(labels_train), 1])
#         data_train = concatenatedFiles_train[:,:-1]
#         #test
#         # concatenatedFiles_test = saver.importMatrixfromPICKLE(feat_path, inputFileNameTest)
#         labels_test = concatenatedFiles_test[:, -1]
#         labels_test = np.reshape(labels_test, [len(labels_test), 1])
#         data_test = concatenatedFiles_test[:, :-1]
#     else:
#         #train
#         data_train = concatenatedFiles_train
#         #test
#         data_test = concatenatedFiles_test
#     #Calculation of z-score
#     meanTrain = np.nanmean(data_train, axis=0)
#     stdTrain = np.nanstd(data_train, axis=0)
#     data_train_std = (data_train-meanTrain)/stdTrain
#     data_test_std = (data_test - meanTrain) / stdTrain
#     #remove outliers (RO)
#     #(RO) train
#     if(typeOfNormalization==1):
#         if (not zscoreThreshold == None):
#             #train
#             lowerBound = np.percentile(data_train_std, zscoreThreshold, axis=0) #with axis 0 we obtain percentile per columns(features)
#             upperBound = np.percentile(data_train_std, (100-zscoreThreshold), axis=0)
#             zscore_data = data_train_std.transpose()
#             zscore_data = filesCreator.getZscoreWithouthPercentile(zscore_data, lowerBound, upperBound)
#             data_train_std = zscore_data.transpose()
#
#     elif(typeOfNormalization==2): #metodo Fernando
#         if (zscoreThreshold == None):
#             zscoreThreshold = 25 #just for doing the normalization of the next if
#         minorPercentile = np.percentile(data_train_std, zscoreThreshold, axis=0)
#         mayorPercentile = np.percentile(data_train_std, 100-zscoreThreshold, axis=0)
#         outliersDistance = (mayorPercentile - minorPercentile) * 1.5
#         medianValue = np.percentile(data_train_std, 50)
#         lowerBound = medianValue - outliersDistance
#         upperBound = medianValue + outliersDistance
#         zscore_data = data_train_std.transpose()
#         zscore_data = filesCreator.getZscoreWithouthPercentile(zscore_data, lowerBound, upperBound)
#         data_train_std = zscore_data.transpose()
#     #(RO) test
#     if (not zscoreThreshold == None):  # for both normalizations it is the same way of application for test
#         zscore_data = data_test_std.transpose()
#         zscore_data = filesCreator.getZscoreWithouthPercentile(zscore_data, lowerBound, upperBound)
#         data_test_std = zscore_data.transpose()
#
#
#     #NORMALIZATION TO SQUEZENET'S IMAGE SCALE (0-255)
#     if (includeNormAsImg):
#         data_train_std,data_test_std, minTrain, maxTrain = norm_feat_allUsersArray(data_train_std, data_test_std,
#                                                  newSuperiorValue=normSupValue,
#                                                  newInferiorValue=normInfValue, withLabels=False)
#
#     if(withLabels):
#         data_train_std = np.hstack([data_train_std, labels_train])
#         data_test_std = np.hstack([data_test_std, labels_test])
#     if(saveData4Test): #first row: standarization data; second row: normalzation data
#         if(not zscoreThreshold == None):
#             data4Test = np.array([[meanTrain,stdTrain,lowerBound,upperBound],[minTrain, maxTrain]])
#         else:
#             if(not includeNormAsImg):
#                 data4Test = np.array([[meanTrain, stdTrain]])
#             else:
#                 data4Test = np.array([[meanTrain, stdTrain], [minTrain, maxTrain]])
#         saver.savePicklefromMatrix(pathData4Test, data4Test, 'Data4TestZscore'+str(typeOfNormalization)+extraInfo)
#     return data_train_std,data_test_std



# def stand_feat_perUserArray(concatenatedFiles, nFiles, withLabels=False,zscoreThreshold=None, typeOfNormalization=1,
#                             includeNormAsImg=True, normSupValue=255, normInfValue=0):
#     shapeConcatenatedFiles = np.shape(concatenatedFiles)
#     dataSizePerFile = int(shapeConcatenatedFiles[0]/nFiles)
#     if(withLabels):
#         dataWithoutLabels = concatenatedFiles[:,0:-1]
#         labels = concatenatedFiles[:, -1]
#         dataStandarized = np.zeros(shapeConcatenatedFiles)
#     else:
#         dataWithoutLabels = concatenatedFiles
#         dataStandarized = np.zeros(shapeConcatenatedFiles)
#         columnLabels = np.zeros(shapeConcatenatedFiles[0])
#         columnLabels = columnLabels.reshape(shapeConcatenatedFiles[0], 1)
#         dataStandarized = np.hstack((dataStandarized, columnLabels))
#
#     for fileIndex in range(nFiles):
#         indexIni = fileIndex * dataSizePerFile
#         indexEnd = (fileIndex + 1) * dataSizePerFile
#         dataFilei = dataWithoutLabels[indexIni:indexEnd,:]
#         meanVal = np.nanmean(dataFilei, axis=0)
#         stdVal = np.nanstd(dataFilei, axis=0)
#         dataStandarized[indexIni:indexEnd,0:-1]= (dataFilei-meanVal)/stdVal
#         if (typeOfNormalization == 1):
#             if (not zscoreThreshold == None):
#                 lowerBound = np.percentile(dataStandarized[indexIni:indexEnd,0:-1], zscoreThreshold,axis=0)  # with axis 0 we obtain percentile per columns(features)
#                 upperBound = np.percentile(dataStandarized[indexIni:indexEnd,0:-1], (100 - zscoreThreshold), axis=0)
#                 zscore_data = dataStandarized[indexIni:indexEnd,0:-1].transpose()
#                 zscore_data = filesCreator.getZscoreWithouthPercentile(zscore_data, lowerBound, upperBound)
#                 dataStandarized[indexIni:indexEnd,0:-1] = zscore_data.transpose()
#         elif (typeOfNormalization == 2):  # metodo Fernando
#             if (zscoreThreshold == None):
#                 zscoreThreshold = 25  # just for doing the normalization of the next if
#             minorPercentile = np.percentile(dataStandarized[indexIni:indexEnd,0:-1], zscoreThreshold,axis=0)
#             mayorPercentile = np.percentile(dataStandarized[indexIni:indexEnd,0:-1], (100 - zscoreThreshold), axis=0)
#             outliersDistance = (mayorPercentile - minorPercentile) * 1.5
#             medianValue = np.percentile(dataStandarized[indexIni:indexEnd,0:-1], 50)
#             lowerBound = medianValue - outliersDistance
#             upperBound = medianValue + outliersDistance
#             zscore_data = dataStandarized[indexIni:indexEnd, 0:-1].transpose()
#             zscore_data = filesCreator.getZscoreWithouthPercentile(zscore_data, lowerBound, upperBound)
#             dataStandarized[indexIni:indexEnd, 0:-1] = zscore_data.transpose()
#         # NORMALIZATION TO SQUEZENET'S IMAGE SCALE (0-255)
#         if(includeNormAsImg):
#             dataNorm = norm_feat_perUserArray(dataStandarized[indexIni:indexEnd, 0:-1], 1,
#                                                                     newSuperiorValue=normSupValue,
#                                                                     newInferiorValue=normInfValue, withLabels=False)
#             dataStandarized[indexIni:indexEnd, 0:-1] = dataNorm
#     if(withLabels):
#         dataStandarized[:,-1]=labels
#     else:
#         dataStandarized=np.delete(dataStandarized, -1, -1)
#     return dataStandarized



def norm_feat_perUserArray(concatenatedFiles, nFiles, newSuperiorValue=1, newInferiorValue=0, withLabels=False):
    shapeConcatenatedFiles = np.shape(concatenatedFiles)
    dataSizePerFile = int(shapeConcatenatedFiles[0]/nFiles)
    dataNormalized = np.zeros(shapeConcatenatedFiles)
    if(withLabels):
        dataWithoutLabels = concatenatedFiles[:,0:-1]
        labels = concatenatedFiles[:,-1]
    else:
        dataWithoutLabels = concatenatedFiles
        labels = []

    for fileIndex in range(nFiles):
        indexIni = fileIndex * dataSizePerFile
        indexEnd = (fileIndex + 1) * dataSizePerFile
        dataFilei = dataWithoutLabels[indexIni:indexEnd,:]
        minVal = np.nanmin(dataFilei, axis=0)
        maxVal = np.nanmax(dataFilei, axis=0)
        dataNormFilei = getNormalizedData(dataFilei, minVal, maxVal, newSupValue=newSuperiorValue, newInfValue=newInferiorValue)
        dataNormalized[indexIni:indexEnd, :] = dataNormFilei
        if(withLabels):
            dataNormalized[indexIni:indexEnd, 0:-1] = dataNormFilei
            dataNormalized[indexIni:indexEnd,-1]=labels[indexIni:indexEnd]
    return dataNormalized


def norm_feat_allUsersArray(concatenatedFiles_train, concatenatedFiles_test, newSuperiorValue=1, newInferiorValue=0,
                            withLabels=False, saveData4Test=True, pathData4Test=''):
    '''
           Standarization of features using zscore with info of all users in the training set.
           :param concatenatedFiles_train:
           :param concatenatedFiles_test:
           :rtype: return standarized training and test sets.
               '''
    # concatenatedFiles_train = saver.importMatrixfromPICKLE(feat_path, inputFileNameTrain)
    if (withLabels):
        labels_train = concatenatedFiles_train[:, -1]
        labels_train = np.reshape(labels_train, [len(labels_train), 1])
        data_train = concatenatedFiles_train[:, :-1]
    else:
        data_train = concatenatedFiles_train
    minTrain = np.nanmin(data_train, axis=0)
    maxTrain = np.nanmax(data_train, axis=0)
    data_train_norm = getNormalizedData(data_train, minTrain, maxTrain, newSupValue=newSuperiorValue, newInfValue=newInferiorValue)
    if (withLabels):
        data_train_norm = np.hstack([data_train_norm, labels_train])
        # concatenatedFiles_test = saver.importMatrixfromPICKLE(feat_path, inputFileNameTest)
        labels_test = concatenatedFiles_test[:, -1]
        labels_test = np.reshape(labels_test, [len(labels_test), 1])
        data_test = concatenatedFiles_test[:, :-1]
    else:
        data_test = concatenatedFiles_test

    data_test_norm = getNormalizedData(data_test, minTrain, maxTrain, newSupValue=newSuperiorValue, newInfValue=newInferiorValue)

    if (withLabels):
        data_test_norm = np.hstack([data_test_norm, labels_test])
    if (saveData4Test):  # first row: normalzation data
        data4Test = np.array([minTrain, maxTrain])
        saver.savePicklefromMatrix(pathData4Test, data4Test, 'Data4TestNorm')
    return data_train_norm, data_test_norm, minTrain, maxTrain


def stand_feat_perUser(feat_path, files2std=[],columns2Delete=[]):
    '''
       Standarization of features using zscore per user
       :param feat_path: path in which we have the gold standard and the training sets
       :param files2std: list with the names of the files to standarize that are in feat path folder.
       :rtype: return standarized training and test sets.
               '''
    stdFiles = []
    headers =[]
    for file in files2std:
        if "gs_" in file:
            path_file = "".join([feat_path, file])
            loadedData = loader.loadArffAsArray(path_file, columns2Delete)
            data =loadedData[:,:-1]
            labels = loadedData[:,-1]
            labels = np.reshape(labels, [len(labels), 1])

            meanVal = np.nanmean(data,axis=0)
            stdVal = np.nanstd(data,axis=0)

            data_std = (data-meanVal)/stdVal
            data_std = np.hstack([data_std, labels])
            if (stdFiles == []):
                stdFiles = data_std
                headers = loader.loadArffHeadersAsArray(path_file, columns2Delete)
            else:
                stdFiles = np.concatenate([stdFiles, data_std])
    return stdFiles, headers

#TEST NORMALIZATION:
# input_file_path = '/home/cris/PycharmProjects/MasterProject/data/SPECTRUMS/RECOLA/spectrums_16kHz_672points/frameSize100ms/test_9_spectrum.mat'
# typeOfNormalization = 7
# newSupValue = 255
# newInfValue = 0
# outputPath= '/home/cris/PycharmProjects/MasterProject/data/IMAGES/RECOLA/DISTRIBUTIONFFTs/'
# text2add = "pruebas7"
# percentileNormalizationType4= 25
#
#
# normPercentileZscorePerUser(input_file_path, 7, newSupValue, newInfValue,
#                                                 outputPath, text2add, percentile=percentileNormalizationType4)
#
#
# normPercentilePerUser(input_file_path, 5, newSupValue, newInfValue,
#                                                 outputPath, text2add, percentile=percentileNormalizationType4)
#
#












# def detectOutliers_method2(dataVector,isTrain):
#     auxVector = dataVector
#     list_index_outliers = []
#     minorPercentile = np.percentile(dataVector,25)
#     mayorPercentile = np.percentile(dataVector,75)
#     outliersDistance = (mayorPercentile - minorPercentile) * 1.5
#     medianValue = np.percentile(dataVector,50)
#     lowerBound = medianValue - outliersDistance
#     upperBound = medianValue + outliersDistance
#     list_index_outliers = []
#     for i in range(len(dataVector)):
#         if((dataVector[i]>upperBound) | (dataVector[i]<lowerBound)):
#             list_index_outliers.append(i)
#     dataVector = np.delete(dataVector.transpose(),list_index_outliers)
#     mean = np.mean(dataVector)
#     standDev = np.std(dataVector)
#     [normalizationVector,max, min]=zscoreAndNormalization(auxVector, dataVector, list_index_outliers, mean, standDev, max, min, isTrain)
#     return normalizationVector, mean,standDev,max, min, lowerBound, upperBound







# CREATE DEV MATRIX FROM MIN & MAX SAVED IN THE NORMALIZATION OF TRAINING PICKLE
# PATH = os.getcwd()
# filepath_pickle = '../data/dataset_Recola/delay0/arousal/features/'
# arff_file_train1 = '../data/dataset_Recola/delay0/arousal/features/stand_train_withoutFrameTime.arff'
# arff_file_conc = '../data/dataset_Recola/delay0/arousal/features/conc_train.arff'
#
# arff_matrix = reader.obtainArffAsArray(arff_file_train1,[])
# arff_matrix_train = reader.obtainArffAsArray(arff_file_train1,[])
# arff_matrix_conc = reader.obtainArffAsArray(arff_file_conc,[])

#
# data = zscore(arff_matrix_conc[:,2])
# print('hi')

#
# # #SAVE DATA INTO PICKLE
#dataset_path= filepath_pickle+'conc_train.arff'
# name_pickle = 'devNorm0255'
# name_pickleMaxminMatrix= 'trainNorm0255_MaxMin'
# minMaxMatrix = (saver.importMatrixfromPICKLE(filepath_pickle, name_pickleMaxminMatrix)).transpose()
#data = reader.obtainArffAsArray(dataset_path,[])
#store.exportPicklefromMatrix(filepath_pickle, data, 'conc_dev')
#b = [i[0] for i in data['attributes']]
#df = pd.DataFrame(data=data['data'], columns= b)
#pd.DataFrame.to_pickle(path=filepath_pickle+'conc_trainDF.p', self=df)


#TRAINING METHOD2-------------------
# data = store.importMatrixfromPICKLE(filepath_pickle, 'conc_train')
# labels_data_arff = '../data/dataset_Recola/delay_2con8sec/stand_train_withoutFrameTime.arff'
# lables_data = reader.obtainArffAsArray(labels_data_arff,[])
# filepath_pickle = '../data/dataset_Recola/delay_2con8sec/arousal/features/'
#
# matrix_train_data = np.zeros(shape= (6, data.shape[1]-1))
# train_zscoreNorm = np.zeros(shape = (data.shape[0], data.shape[1]-1))
# for j in range(0,(data.shape[1]-1)): #quitar labels
#     [newVector, mean, standDev, max, min, lowerBound, upperBound] = detectOutliers_method2(data[:, j],True)
#     matrix_train_data[:,j-1]=[mean, standDev, max, min, lowerBound, upperBound]
#     train_zscoreNorm[:,j-1]=newVector
# train_zscoreNorm[:,-1] = lables_data[:,-1]#labels
#
# store.exportPicklefromMatrix(filepath_pickle, train_zscoreNorm, 'zscoreNorm_train_METHOD2')
# store.exportPicklefromMatrix(filepath_pickle, matrix_train_data, 'zscoreNorm_MeanDevMaxMinBounds_METHOD2')
# #DEV PART:
# data = store.importMatrixfromPICKLE(filepath_pickle, 'conc_dev')
# labels_data_arff = '../data/dataset_Recola/delay_2con8sec/stand_dev_withoutFrameTime.arff'
# lables_data = reader.obtainArffAsArray(labels_data_arff,[])
# filepath_pickle = '../data/dataset_Recola/delay_2con8sec/arousal/features/'
# matrix_test_data = store.importMatrixfromPICKLE(filepath_pickle, 'zscoreNorm_MeanDevMaxMinBounds_METHOD2')
# test_zscoreNorm = np.zeros(shape = (data.shape[0], data.shape[1]-1))
#
# for j in range(1,(data.shape[1]-1)):
#     [mean, standDev, max, min, lowerBound, upperBound] = matrix_test_data[:,j-1]
#     test_zscoreNorm[:, j - 1] = ZscoreAndNormalizationTest(data[:, j],mean, standDev, max, min, lowerBound, upperBound)
# test_zscoreNorm[:,-1] = lables_data[:,-1]
# store.exportPicklefromMatrix(filepath_pickle, test_zscoreNorm, 'zscoreNorm_dev_METHOD2')
#














#CREATION OF TRIANING ZCORE-NORM:----------------
#
# data = store.importMatrixfromPICKLE(filepath_pickle, 'conc_train')
# labels_data_arff = '../data/dataset_Recola/delay_2con8sec/stand_train_withoutFrameTime.arff'
# lables_data = reader.obtainArffAsArray(labels_data_arff,[])
# filepath_pickle = '../data/dataset_Recola/delay_2con8sec/arousal/features/'
#
# matrix_train_data = np.zeros(shape= (6, data.shape[1]-1))
# train_zscoreNorm = np.zeros(shape = (data.shape[0], data.shape[1]-1))
# minPer = 2
# maxPer = 98
# for j in range(0,(data.shape[1]-1)): #quitar labels
#     [newVector, mean, standDev, max, min, lowerBound, upperBound] = ZscoreAndNormalizationTrain(data[:, j],minPer,maxPer)
#     matrix_train_data[:,j-1]=[mean, standDev, max, min, lowerBound, upperBound]
#     train_zscoreNorm[:,j-1]=newVector
# train_zscoreNorm[:,-1] = lables_data[:,-1]#labels
#
# store.exportPicklefromMatrix(filepath_pickle, train_zscoreNorm, 'zscoreNorm_train_2-98')
# store.exportPicklefromMatrix(filepath_pickle, matrix_train_data, 'zscoreNorm_MeanDevMaxMinBounds2-98')


# #CREATION OF DEV ZCORE-NORM:---------------
#
# data = store.importMatrixfromPICKLE(filepath_pickle, 'conc_dev')
# labels_data_arff = '../data/dataset_Recola/delay_2con8sec/stand_dev_withoutFrameTime.arff'
# lables_data = reader.obtainArffAsArray(labels_data_arff,[])
# filepath_pickle = '../data/dataset_Recola/delay_2con8sec/arousal/features/'
# matrix_test_data = store.importMatrixfromPICKLE(filepath_pickle, 'zscoreNorm_MeanDevMaxMinBounds2-98')
# test_zscoreNorm = np.zeros(shape = (data.shape[0], data.shape[1]-1))
#
# for j in range(1,(data.shape[1]-1)):
#     [mean, standDev, max, min, lowerBound, upperBound] = matrix_test_data[:,j-1]
#     test_zscoreNorm[:, j - 1] = ZscoreAndNormalizationTest(data[:, j],mean, standDev, max, min, lowerBound, upperBound)
# test_zscoreNorm[:,-1] = lables_data[:,-1]
# store.exportPicklefromMatrix(filepath_pickle, test_zscoreNorm, 'zscoreNorm_dev_2-98')
# #END ZSCORE-NORM FILES CREATION----------------
# print('hi')



# normData= normalizationTest(0,255,data, minMaxMatrix)
#


#saver.exportPicklefromMatrix(filepath_pickle, normData.transpose(), name_pickle)
#saver.exportPicklefromMatrix(filepath_pickle, maxMinMatrix.transpose(), name_pickleMaxminMatrix)

#
#
#LOAD DATA FROM PICKLE
#dataset_path= filepath_pickle+'conc_train.arff'
# name_pickle = 'devNorm0255'
# new_name_pickle = 'devNorm0255subset3audios'
# training_pickle = filepath_pickle+name_pickle
# training_data = saver.importMatrixfromPICKLE(filepath_pickle, name_pickle)
# #dev_data = saver.importMatrixfromPICKLE(filepath_pickle, dev_pickle)
# saver.exportPicklefromMatrix(filepath_pickle, training_data[:,0:22503], new_name_pickle)

# window = 8960
# shift = 40
# batch_size = 224
# n_frames = math.trunc(window/shift) #frames per image

#ZM+NORM AND SAVE DATA AS CSV
# dataset_path= filepath_pickle+'conc_train.arff'
# train_pick_name = 'trainNorm0255'
# output_file = filepath_pickle+'testCsv_ZMPlusNorm.csv'
#
# data = reader.obtainArffAsArray(dataset_path,[])
#
#
# [ZMmatrix,meanMatrx] = ZMsubstractionTrain(data)
#
#
# [normData,minMaxmatrix]= normalizationTrain(0,255,ZMmatrix)
#
# saver.exportMatrixAsCSV(normData,output_file)
#
# arff_file_train = '../data/dataset_Recola/delay_2con8sec/stand_train_withoutFrameTime.arff'
# arff_file_test = '../data/dataset_Recola/delay_2con8sec/stand_dev_withoutFrameTime.arff'
#
# arff_file_train_original = filepath_pickle+'conc_train.arff'
#
# train_pickle_name = 'trainNorm0255'#'zscoreNorm_train_5-95'
# dev_pickle_name = 'devNorm0255'#'zscoreNorm_dev_5-95'
#
# new_pickle_path = '../data/dataset_Recola/delay_2con8sec/arousal/features/'
#
#
# # arff_matrix_train = reader.obtainArffAsArray(arff_file_train,[])
# # #arff_matrix_test = reader.obtainArffAsArray(arff_file_test,[])
# # arff_file_train_original= reader.obtainArffAsArray(arff_file_train_original,[])
# training_pickle = saver.importMatrixfromPICKLE(filepath_pickle, train_pickle_name)
#
# [data,matrix] = ZscoreAndNormalizationTrain(arff_file_train_original[:,1],0,100)
# print(data)
# #
# training_pickle = saver.importMatrixfromPICKLE(filepath_pickle, train_pickle_name)
# dev_pickle = saver.importMatrixfromPICKLE(filepath_pickle, dev_pickle_name)
#
# #training_pickle[:,-1]=arff_matrix_train[:,-1]
# #dev_pickle[:,-1]=arff_matrix_test[:,-1]
# training_pickle[-1,:]=arff_matrix_train[:,-1]
# dev_pickle[-1,:]=arff_matrix_test[:,-1]
#
#
# saver.exportPicklefromMatrix(new_pickle_path, training_pickle, train_pickle_name)
# saver.exportPicklefromMatrix(new_pickle_path, dev_pickle, dev_pickle_name)
#
