import numpy as np
import matplotlib as mlp


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
    normalized_data = (((data2Norm - minimumValue) / (maximumValue - minimumValue)) * (newSupValue - newInfValue)) + newInfValue
    # check if some number surphase the maximum and minimum (newSupValue & newInfValue)
    normalized_data[normalized_data > newSupValue] = newSupValue
    normalized_data[normalized_data < newInfValue] = newInfValue
    return normalized_data

def getZscoreData(data2Zscore, meanValue, stdValue,zscoreThreshold=None, withLabels=False):
    epsilon = 1e-20
    if(withLabels):
        labels = data2Zscore[:,-1]
        data2Zscore = data2Zscore[:,:-1]
    zscore_data = (data2Zscore - meanValue) / (stdValue + epsilon)
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

