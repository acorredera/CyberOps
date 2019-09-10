#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from sklearn.externals import joblib
import pickle
import numpy as np
import scipy,os
from PIL import Image

"""
Función que guarda un modelo hecho en tensorflow
    session: sesión del modelo tf, debe estar abierta
    filename: path en el que se desea guardar
    global_n: id del modelo (o del estado del mismo si procede)
return:
    save_path: path del modelo guardado (con id)
"""
def saveModel(session, filename, global_n):
    saver = tf.train.Saver()
    save_path = saver.save(session, filename, global_step=global_n)
    print("Model saved to %s" % save_path)
    return save_path


"""
A comentar
"""
def save_metrics_to_file(filePath,header,description, metrics):
    if (os.path.isfile(filePath)):
        openingMode = 'a+'
        header = ""
    else:
        header = header
        openingMode = 'w'
    with open(filePath, openingMode) as output:
        output.write(header)
        output.write(description+';')
        for metric in metrics:
            output.write(str("{:.6f}".format(metric))+';')
        output.write('\n')
        output.close()



"""
A comentar
"""
def save_logs(filePath, description1, id):
    output = open(filePath, 'a')
    output.write(str(id)+';')
    for log in description1:
        output.write(str("{:.3f}".format(log)) + ',')
    output.write('\n')
    output.close()


def save_to_file(filePath, x):
    output = open(filePath, 'a')
    for value in x:
        for valueI in value:
            output.write(str("{:.6f}".format(valueI))+';')
    output.write('\n')
    output.close()

def saveAsArff(path2save, data, headers, relation, class_labels = " {1.0, -1.0}"):
    #arff.dump(path2save, data, relation=relation, names=headers)
    with open(path2save, "w") as file:
        file.write("".join(["@relation ",relation]))
        file.write("\n")
        file.write("\n")
        for attribute in headers:
            if(isinstance(headers[0], tuple)):
                attribute = attribute[0]
            if(not type(attribute) is str):
                attribute = str(attribute)
            if(attribute == "class"):
                file.write("".join(["@attribute ", attribute, class_labels]))
            else:
                file.write("".join(["@attribute ",attribute, " numeric"]))
            file.write("\n")
        file.write("\n")
        file.write("\n")
        file.write("@data\n")
        file.write("\n")
        for rowData in data:
            for irow in rowData[0:-1]:
                file.write("".join([str(irow),","]))
            file.write(str(rowData[-1]))
            file.write("\n")


def save_model_sklearn(model, path2saveModel):
    #extension .sav
    pickle.dump(model, open(path2saveModel, 'wb'))


#Export np matrix to pickle file
def savePicklefromMatrix(filePath, matrix, name):
    filePath=filePath + str(name) + ".p"
    with open(filePath, 'wb') as f:
        pickle.dump(matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def saveMatrixAsCSV(matrix, csvPath):
    np.savetxt(csvPath, matrix, delimiter=',', fmt='%1.8f')

def save_img(img, imgOutputPath):
    #image received must have the same format that: img = Image.fromarray(new_image.astype('uint8'), 'RGB')
    img.save(imgOutputPath, "PNG")

def saveMatrixAsCSVPredictions(matrix, csvPath, arousalOrValence, fileName):
    with open(csvPath, 'w') as file:
        file.write("@relation GOLDSTANDARD_"+arousalOrValence)
        file.write("\n")
        file.write("\n")
        file.write("@attribute Instance_name string\n")
        file.write("@attribute frameTime numeric\n")
        file.write("@attribute GoldStandard numeric\n")
        if ((matrix.shape[1]) >= 2):
            file.write("@attribute label numeric\n")
        file.write("\n")
        file.write("\n")
        file.write("@data\n")
        file.write("\n")
        time = 0.00
        if((matrix.shape[1])>=2):
            for line in matrix:
                file.write(fileName+','+"{:.{}f}".format(time, 2)+','+ "{:.{}f}".format(line[0], 6)+','+ "{:.{}f}".format(line[1], 6))
                file.write("\n")
                time+=0.04
            file.write("\n")
        else:
            for line in matrix:
                file.write(fileName+','+"{:.{}f}".format(time, 2)+','+ "{:.{}f}".format(line[0], 6))
                file.write("\n")
                time+=0.04
            file.write("\n")



    #np.savetxt(csvPath, matrix, delimiter=",",fmt='%1.8f')

def saveListAsCSV(list, csvPath,nameFile, mode='w'):
    with open((csvPath+nameFile), mode) as file:
        for item in list:
            file.write(item)
            file.write('\n')

def saveDataAscsv(numpy_array, csvPath_out, headers=[]):
    num_of_cols= np.shape(numpy_array)[-1]
    with open(csvPath_out, 'w') as file:
        line_with_headers = "0"
        if(headers==[]):
            headers = range(1,num_of_cols)
        for i in headers:
            line_with_headers += ("," +str(i))
        file.write((line_with_headers+"\n"))
        saveMatrixAsCSV(numpy_array, file)




def saveMatrixAsMat(matrix, matPath, nameMatrixMatlab='matrix'):
    scipy.io.savemat(matPath, mdict={nameMatrixMatlab: matrix})

def saveModelKeras(model, json_path, weights_path):
    # serialize model to JSON
    model_json = model.to_json()
    json_path = json_path+'.json'
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_path+ '.h5')
    print("Saved model in disk")


