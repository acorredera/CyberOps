import codecs
import arff, os
#from keras.models import model_from_json
import numpy as np
#import tensorflow as tf
import scipy.io as sio
import scipy
import pickle
import pandas as pd
from PIL import Image
from scipy.io import arff as arff_v2
import tensorflow as tf
from keras.models import model_from_json

def loadArffAsArray(arff_file, columns2delete=[]):
    file_ = codecs.open(arff_file, 'rb', 'utf-8')
    arff_file = arff.load(file_)
    arff_data = arff_file['data']
    columns2delete.sort(reverse=True)
    if(not columns2delete==[]):
        for row in arff_data:
            for col in columns2delete:
                del row[col]
    arff_data_array = np.asarray(arff_data,dtype=float)
    return arff_data_array

def loadArffHeadersAsArray(arff_file,columns2delete = []):
    file_ = codecs.open(arff_file, 'rb', 'utf-8')
    arff_file = arff.load(file_)
    arff_header = arff_file['attributes']
    #arff_header_array = np.asarray(arff_header)
    headerNames = [i[0] for i in arff_header]
    for col in columns2delete:
        del headerNames[col]
    return headerNames

def loadCompleteArff(arff_file, columns2delete=[], stringAttr = False):
    file_ = open(arff_file, 'r')#codecs.open(arff_file, 'rb', 'utf-8')
    arff_file = arff.load(file_)
    columns2delete.sort(reverse=True)
    arff_data = arff_file['data']
    arff_header = arff_file['attributes']
    arff_relation = arff_file['relation']
    # data
    if (not stringAttr):
        arff_data_array = np.asarray(arff_data, dtype=float)
        for col in columns2delete:
            arff_data_array = np.delete(arff_data_array, col, axis=1)
    else:
        for col in columns2delete:
            del arff_data[0][col]
        arff_data_array = arff_data #arff_data_array[0][0:5] #first index has to be unique
        if(not isinstance(arff_data_array[0][0],str)):
            arff_data_array = np.asarray(arff_data_array, dtype=float)
    # headers
    for col in columns2delete:
        del arff_header[col]
    return arff_data_array, arff_header, arff_relation, stringAttr

def change_string_columns_by_numbers(arff_data_array, column_number):
    n=0
    for arff_row in range(0,len(arff_data_array)):
        arff_data_array[arff_row][column_number]=n
        n+=1
    arff_data_np = np.asarray(arff_data_array, dtype=float)
    return arff_data_np

def loadCompleteArff_v2(arff_file, columns2delete=[]):
    columns2delete.sort(reverse=True)
    data, meta = arff_v2.loadarff(arff_file)
    print("arff_loaded")
    arff_header = meta._attrnames
    arff_relation = meta.name
    if(len(data)==0):
        return data,"",""
    else:
        arff_data_array = np.asarray(data.tolist(), dtype=np.float32)
        arff_data_array = np.delete(arff_data_array, columns2delete, axis=1)
        for col in columns2delete:
            del arff_header[col]
        return arff_data_array, arff_header, arff_relation


def check_data_not_empty(arff_file):
    if(arff_file):
        print("tenemos file")

def loadArffAsDataset(arff_file, columns2delete):
    file_ = codecs.open(arff_file, 'rb', 'utf-8')
    arff_file = arff.load(file_)
    return arff_file

def loadImageAsArray(path_img):
    img = load_image(path_img)
    imgAsArray = np.asarray(img)
    return imgAsArray

def load_image(path_img):
    return Image.open(path_img)

def loadMatFiles(path_mat):
    dataMat = sio.loadmat(path_mat)
    return dataMat

def loadCsv(filePath, delim, force=False, header = -1):
    #header = 0 to infer headers from the first row
    dataset = pd.read_csv(filePath, sep=delim, header=header, low_memory=True, encoding = "ISO-8859-1")
    return dataset

# Import np matrix from pickle file
def loadMatrixfromPICKLE(filePath, name):
    filePath = filePath + str(name) + ".p"
    with open(filePath, 'rb') as f:
        matrix = pickle.load(f)
    f.close()
    return matrix

def load_model_sklearn(pathOfModel):
    model = pickle.load(open(pathOfModel, 'rb'))
    return model

def loadMatrixFromMat(matPath):
    return scipy.io.loadmat(matPath)

def loadtxt_as_list(txtPath):
    f = open(txtPath, 'r')
    data_list = f.read().splitlines()
    f.close()
    return data_list

def load_npy(npy_path):
    return np.load(npy_path)

def load_summaries_tensorboard(path_summary):
    for event in tf.train.summary_iterator(path_summary):
        for value in event.summary.value:
            print(value.tag)
            if value.HasField('simple_value'):
                print(str(value.simple_value))

def analyse_results_tensorboard_xval(initial_path, tag_to_save, num_folds=10):
    # initial_path = "/home/cristinalunaj/Downloads/20190311-173353/"
    # tag_to_save = "accuracy_1"  # loss_function or accuracy_1
    # num_folds = 10
    output_file = initial_path + tag_to_save + ".csv"
    # Auxiliar variables
    max_train = np.zeros(num_folds)
    max_val = np.zeros(num_folds)
    max_test = np.zeros(num_folds)
    last_train = np.zeros(num_folds)
    last_val = np.zeros(num_folds)
    last_test = np.zeros(num_folds)
    for part in ["train", "val", "test"]:
        for fold in range(num_folds):
            path_file = "".join([initial_path, part, "/fold", str(fold), "/"])
            if(not os.path.exists(path_file)):
                continue
            files = os.listdir(path_file)
            for file in files:
                if ("events" in file):
                    path_to_events_file = path_file + file
                    try:
                        for e in tf.train.summary_iterator(path_to_events_file):
                            for v in e.summary.value:
                                if v.tag == tag_to_save:
                                    print(v.simple_value)
                                    new_value = v.simple_value
                                    if (part == "train"):
                                        if (max_train[fold] < new_value):
                                            max_train[fold] = new_value
                                        last_train[fold] = new_value
                                    elif (part == "val"):
                                        if (max_val[fold] < new_value):
                                            max_val[fold] = new_value
                                        last_val[fold] = new_value
                                    elif (part == "test"):
                                        if (max_test[fold] < new_value):
                                            max_test[fold] = new_value
                                        last_test[fold] = new_value
                    except:
                        continue
    # save average results in same folder
    with open(output_file, "w") as f:
        # headers
        f.write(
            "max_" + tag_to_save + "train,max_" + tag_to_save + "val,max_" + tag_to_save + "test" + "last_" + tag_to_save + "train,last_" + tag_to_save + "val,last_" + tag_to_save + "test\n")
        # values
        f.write(str(np.mean(max_train)) + "," + str(np.mean(max_val)) + "," + str(
            np.mean(max_test)) + "," + str(
            np.mean(last_train)) + "," + str(np.mean(last_val)) + "," + str(np.mean(last_test)) + "\n")
    print("Written summary data into " + output_file)
    return max_val


def loadModelKeras(weights_path,json_path):  # For weights saved with model.save_weights(filepath): saves the weights of the model as a HDF5 file.
    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)  # example weigths.h5
    print("Loaded model from disk")
    return loaded_model

# initial_path = "/home/cris/PycharmProjects/InterSpeech19/data/results/LSTM_with_embeddings/lstm_variable_logs/split_xval_embeddings/"
# listFolders = os.listdir(initial_path)
# tag_to_save = "accuracy_1"
# for fold in listFolders:
#     path = os.path.join(initial_path, fold)
#     if('20190318' in fold):
#         analyse_results_tensorboard_xval(path+"/", tag_to_save, num_folds=10)

# path_summaries = "data/results/LSTM_with_embeddings/lstm_variable_logs/split_xval_embeddings/20190311-173353/val/fold0/events.out.tfevents.1552322090.cris-X550VXK"
# load_summaries_tensorboard(path_summaries)
# #


# """
# Función que carga un modelo tensorflow en el programa
#     filename: path del modelo a cargar (con id)
# return:
#     session: sesión de del modelo tf
#     tf_test_dataset: variable input para test del modelo
#     test_prediction: variable output para test del modelo
# """
# def loadModel(filename):
#     session = tf.Session()
#     filenamemeta = filename + '.meta'
#     new_saver = tf.train.import_meta_graph(filenamemeta)
#     new_saver.restore(session, filename)
#     tf_train_dataset = session.graph.get_tensor_by_name("trainds_input:0")
#     tf_train_labels = session.graph.get_tensor_by_name("trainlb_input:0")
#     # tf_valid_dataset = session.graph.get_tensor_by_name("valid_input:0")
#     tf_test_dataset = session.graph.get_tensor_by_name("test_input:0")
#     train_prediction = session.graph.get_tensor_by_name("train_output:0")
#     # valid_prediction = session.graph.get_tensor_by_name("valid_output:0")
#     test_prediction = session.graph.get_tensor_by_name("test_output:0")
#     weights_SGD_1 = session.graph.get_tensor_by_name("weights_SGD_1:0")
#     weights_SGD_2 = session.graph.get_tensor_by_name("weights_SGD_2:0")
#     biases_SGD_1 = session.graph.get_tensor_by_name("biases_SGD_1:0")
#     biases_SGD_2 = session.graph.get_tensor_by_name("biases_SGD_2:0")
#
#     return session, tf_train_dataset, tf_train_labels, tf_test_dataset, train_prediction, test_prediction, weights_SGD_1, weights_SGD_2, biases_SGD_1, biases_SGD_2

        #
# dataMat = loadMatFiles('/home/cris/Documentos/becas-DIE/AUDIO/datasets/RECOLA_dataset/avec_2016/ratings_gold_standard/gs_delayed_0seg.mat')
# dataMat['gold_standard'][0,0]
# print('Hola')
# print(dataMat)