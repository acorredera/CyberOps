from math import sqrt
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM
import metrics.lossFunctions as lossFunct
import attributeGenerator.setCreator as filesCreator
import numpy as np
import keras
import metrics.metrics_regression as metrics
from keras import optimizers
import os, time
import utils.dataSaver as saver
import math
import utils.dataLoader as loader
import attributeSelector.attributeSelector as attributeSelector
import attributeSelector.datasetCleaner as cleaner
import settings.Constraints as constraints
import utils.dataDisplayer as displayer



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1): #(terms t-i) CREATIN OF THE SUB-DATASETS WITH SAMPLES SHIFTED i TIMESTEPS
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out): #terms (t ot (t+i))
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1) #concatenate in columns for having: t-i....t....t+i
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepareData(data, timesteps, featuresIni, returnLabelsAndData = False, withLabels = True):
    #cleanData...
    reframed = series_to_supervised(data, timesteps, 1)
    data_y = []
    if(withLabels):
        reframed.drop(reframed.columns[[featuresIni]], axis=1, inplace=True) #eliminate colums that we are not gonna use for prediction
        column_remove = featuresIni

        for i in range(timesteps-1):
            column_remove = column_remove+featuresIni
            reframed.drop(reframed.columns[[column_remove]], axis=1, inplace=True)
        #print(reframed.head())

    data_values = reframed.values
    data_X = data_values
    if(returnLabelsAndData==True):
        data_X, data_y = data_values[:, :-1], data_values[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    data_X = data_X.reshape((data_X.shape[0], timesteps+1, featuresIni))
    return data_X, data_y



def getLSTMpredictions(jsonModel_path,weightsModel_path, testData,labels=[]):
    model = loader.loadModelKeras(weightsModel_path, jsonModel_path)
    predictions = model.predict(testData)
    if(labels==[]):
        print("no labls")
    else:
        print("si labels")
        reShapedLabels = labels.reshape((len(predictions), 1))
        # calculate RMSE
        rmse = sqrt(mean_squared_error(reShapedLabels, predictions))
        print('Test RMSE: %.3f' % rmse)
        pred = predictions.reshape(-1)
        true = reShapedLabels.reshape(-1)
        pearson_coeff = np.corrcoef(pred, true)[0, 1]
        mae = metrics.mae(true, pred)
        print('Test MAE :', mae)
        print('Test CC:', pearson_coeff)
        # P_cc = scipy.stats.pearsonr(pred, true)
        # print('Test CC (scipy):', pearson_coeff)
        CCCoeff = metrics.CCC(true, pred)
        print('Test CCC :', CCCoeff)
    return predictions




# name_pickle = 'stand_train'
# dev_name_pickle = 'stand_dev'

# #SAVE DATA INTO PICKLE
#data = reader.obtainArffAsArray(dataset_path, [0]) #Remove the 0 column that contains the time
#saver.exportPicklefromMatrix(filepath_pickle, data, name_pickle)

# data = reader.obtainArffAsArray(dataset_path_train,[])
# dev_data = reader.obtainArffAsArray(dataset_path_dev,[])
#
# jsonModel_path = "/home/cris/PycharmProjects/CompactedModels/data/results/models_trained/Recola_features_audio__LSTM_funcionals_optimizer_sgd__neurons80_timeSteps_10_lr_0.0001_lossf_mse_batch_75.json"
# weightsModel_path = "/home/cris/PycharmProjects/CompactedModels/data/results/models_trained/Recola_features_audio__LSTM_funcionals_optimizer_sgd__neurons80_timeSteps_10_lr_0.0001_lossf_mse_batch_75.h5"
# timesteps = 10
# featuresIni = 88
# data_X, data_y = prepareData(dev_data, timesteps, featuresIni, returnLabelsAndData = True)
# getLSTMpredictions(jsonModel_path,weightsModel_path, data_X,labels=data_y)
#

def getOptName(optimizer, learning_rate):
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)  # decay=1e-6,
    adam = optimizers.Adam(lr=learning_rate,beta_1=0.9, beta_2=0.1)  # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adamMax = optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.1, epsilon=1e-08, decay=0.0)
    rms = optimizers.RMSprop(lr=learning_rate, rho=0.9)

    if (optimizer == adam):
        opt_name = "".join(['adam_lr_', str(learning_rate)])
    elif(optimizer==sgd):
        opt_name = "".join(['sgd_lr_', str(learning_rate)])
    elif (optimizer == adamMax):
        opt_name = "".join(['adamMax_lr_', str(learning_rate)])
    elif (optimizer == rms):
        opt_name = "".join(['rms_lr_', str(learning_rate)])
    else:
        opt_name = "UNKNOWN"
    return opt_name



def defineGRU_modelStatefulFalse(num_neurons,trainingDataXShape,regularization,batch_size,lossFunction='mse',
                 optimizer = 'sgd',learning_rate = 0.0001, momentum = 0.9,stateful=False, decay = 1e-6, oldWeigths = []):
    # OPTIMIZER AND LOSS FUNCTION
    if (optimizer == 'sgd'):
        # nesterov: boolean. Whether to apply Nesterov momentum.
        optim = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    elif (optimizer == 'rms'):
        optim = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9)
    elif (optimizer == 'adam'):
        optim = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.1, decay=decay)
    elif (optimizer == 'adamMax'):
        optim = keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.1, epsilon=1e-08, decay=decay)
    else:
        optim = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)

    if (lossFunction == 'rmse'):  # RMSE is just the square root of variance, which is actually Standard Deviation.
        lossFunc = lossFunct.RMSE_v2
    elif (lossFunction == 'mse'):  # MSE incorporates both the variance and the bias of the predictor.
        lossFunc = keras.losses.mean_squared_error
    elif (lossFunction == 'mae'):
        lossFunc = keras.losses.mean_absolute_error
    elif (lossFunction == 'CCC'):
        lossFunc = lossFunct.lossFunctCCC4Keras
    else:
        lossFunc = keras.losses.mean_squared_error

    model = Sequential()  # training_data_X.shape

    model.add(GRU(num_neurons,input_shape=(trainingDataXShape[1], trainingDataXShape[2]),
                   stateful=False,
                   kernel_regularizer=keras.regularizers.l2(regularization)))  # Test stateful: default: False

    # If the model is stateless, the cell states are reset at each sequence. With the stateful mode, all the states are propagated to the next batch.
    #  It means that the state of the sample located at index i, Xi will be used in the computation of the sample Xi+bs in the next batch,
    # where bs is the batch size (no shuffling).
    model.add(Dense(1, activation='tanh', kernel_regularizer=keras.regularizers.l2(regularization)))
    if (not oldWeigths == []):
        model.set_weights(oldWeigths)
    model.compile(loss=lossFunc, optimizer=optim,
                  metrics=['mse', 'mae', metrics.RMSE_v2, metrics.PearsonCorrealtion4Keras,
                           metrics.CCC4Keras])
    return model

def deine_LSTMmodelstatefulFalse(num_neurons,trainingDataXShape,regularization,batch_size,lossFunction='mse',
                 optimizer = 'sgd',learning_rate = 0.0001, momentum = 0.9,stateful=False, decay = 1e-6, oldWeigths = []):
    # OPTIMIZER AND LOSS FUNCTION
    if (optimizer == 'sgd'):
        # nesterov: boolean. Whether to apply Nesterov momentum.
        optim = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    elif (optimizer == 'rms'):
        optim = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9)
    elif (optimizer == 'adam'):
        optim = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.1, decay=decay)
    elif (optimizer == 'adamMax'):
        optim = keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.1, epsilon=1e-08, decay=decay)
    else:
        optim = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)

    if (lossFunction == 'rmse'):  # RMSE is just the square root of variance, which is actually Standard Deviation.
        lossFunc = lossFunct.RMSE_v2
    elif (lossFunction == 'mse'):  # MSE incorporates both the variance and the bias of the predictor.
        lossFunc = keras.losses.mean_squared_error
    elif (lossFunction == 'mae'):
        lossFunc = keras.losses.mean_absolute_error
    elif (lossFunction == 'CCC'):
        lossFunc = lossFunct.lossFunctCCC4Keras
    else:
        lossFunc = keras.losses.mean_squared_error

    model = Sequential()  # training_data_X.shape  #CuDNNLSTM
    model.add(LSTM(num_neurons, input_shape=(trainingDataXShape[1], trainingDataXShape[2]),
                   stateful=False,
                   kernel_regularizer=keras.regularizers.l2(regularization)))  # Test stateful: default: False

    # If the model is stateless, the cell states are reset at each sequence. With the stateful mode, all the states are propagated to the next batch.
    #  It means that the state of the sample located at index i, Xi will be used in the computation of the sample Xi+bs in the next batch,
    # where bs is the batch size (no shuffling).
    model.add(Dense(1, activation='tanh', kernel_regularizer=keras.regularizers.l2(regularization)))
    if (not oldWeigths == []):
        model.set_weights(oldWeigths)
    model.compile(loss=lossFunc, optimizer=optim,
                  metrics=['mse', 'mae', metrics.RMSE_v2, metrics.PearsonCorrealtion4Keras,
                           metrics.CCC4Keras])
    return model

def deine_LSTMmodelstatefulFalse2layers(num_neurons,trainingDataXShape,regularization,batch_size,lossFunction='mse',
                 optimizer = 'sgd',learning_rate = 0.0001, momentum = 0.9,stateful=False, decay = 1e-6, oldWeigths = []):
    # OPTIMIZER AND LOSS FUNCTION
    if (optimizer == 'sgd'):
        # nesterov: boolean. Whether to apply Nesterov momentum.
        optim = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    elif (optimizer == 'rms'):
        optim = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9)
    elif (optimizer == 'adam'):
        optim = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.1, decay=decay)
    elif (optimizer == 'adamMax'):
        optim = keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.1, epsilon=1e-08, decay=decay)
    else:
        optim = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)

    if (lossFunction == 'rmse'):  # RMSE is just the square root of variance, which is actually Standard Deviation.
        lossFunc = lossFunct.RMSE_v2
    elif (lossFunction == 'mse'):  # MSE incorporates both the variance and the bias of the predictor.
        lossFunc = keras.losses.mean_squared_error
    elif (lossFunction == 'mae'):
        lossFunc = keras.losses.mean_absolute_error
    elif (lossFunction == 'CCC'):
        lossFunc = lossFunct.lossFunctCCC4Keras
    else:
        lossFunc = keras.losses.mean_squared_error

    model = Sequential()  # training_data_X.shape  #CuDNNLSTM
    model.add(LSTM(num_neurons[0], input_shape=(trainingDataXShape[1], trainingDataXShape[2]),
                   stateful=False,
                   kernel_regularizer=keras.regularizers.l2(regularization),return_sequences=True))  # Test stateful: default: False
    model.add(LSTM(num_neurons[1],stateful=False,kernel_regularizer=keras.regularizers.l2(regularization)))
    #model.add(LSTM(num_neurons[2], stateful=False, kernel_regularizer=keras.regularizers.l2(regularization)))
    # If the model is stateless, the cell states are reset at each sequence. With the stateful mode, all the states are propagated to the next batch.
    #  It means that the state of the sample located at index i, Xi will be used in the computation of the sample Xi+bs in the next batch,
    # where bs is the batch size (no shuffling).
    model.add(Dense(1, activation='tanh', kernel_regularizer=keras.regularizers.l2(regularization)))
    if (not oldWeigths == []):
        model.set_weights(oldWeigths)
    model.compile(loss=lossFunc, optimizer=optim,
                  metrics=['mse', 'mae', metrics.RMSE_v2, metrics.PearsonCorrealtion4Keras,
                           metrics.CCC4Keras])
    return model











def define_model(num_neurons,trainingDataXShape,regularization,batch_size,lossFunction='mse',
                 optimizer = 'sgd',learning_rate = 0.0001, momentum = 0.9,stateful=False, decay = 1e-6, oldWeigths = []):
    # OPTIMIZER AND LOSS FUNCTION
    if (optimizer == 'sgd'):
        # nesterov: boolean. Whether to apply Nesterov momentum.
        optim = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    elif (optimizer == 'rms'):
        optim = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9)
    elif (optimizer == 'adam'):
        optim = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.1, decay=decay)
    elif (optimizer == 'adamMax'):
        optim = keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.1, epsilon=1e-08, decay=decay)
    else:
        optim = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)

    if (lossFunction == 'rmse'):  # RMSE is just the square root of variance, which is actually Standard Deviation.
        lossFunc = lossFunct.RMSE_v2
    elif (lossFunction == 'mse'):  # MSE incorporates both the variance and the bias of the predictor.
        lossFunc = keras.losses.mean_squared_error
    elif (lossFunction == 'mae'):
        lossFunc = keras.losses.mean_absolute_error
    elif (lossFunction == 'CCC'):
        lossFunc = lossFunct.lossFunctCCC4Keras
    else:
        lossFunc = keras.losses.mean_squared_error


    model = Sequential() #training_data_X.shape

    model.add(LSTM(num_neurons, batch_input_shape=(batch_size,trainingDataXShape[1], trainingDataXShape[2]), stateful=stateful,
                       kernel_regularizer=keras.regularizers.l2(regularization)))  # Test stateful: default: False

    # If the model is stateless, the cell states are reset at each sequence. With the stateful mode, all the states are propagated to the next batch.
    #  It means that the state of the sample located at index i, Xi will be used in the computation of the sample Xi+bs in the next batch,
    # where bs is the batch size (no shuffling).
    model.add(Dense(1, activation='tanh', kernel_regularizer=keras.regularizers.l2(regularization)))
    if(not oldWeigths==[]):
        model.set_weights(oldWeigths)
    model.compile(loss=lossFunc, optimizer=optim, metrics=['mse', 'mae', metrics.RMSE_v2, metrics.PearsonCorrealtion4Keras,
                                metrics.CCC4Keras])
    return model



def LSTM_training(training_data, dev_data,test_data,dimension, timesteps, featuresIni,
                  epochs,batch_size,stateful,early_stoppingParameters,buildModelParameters,
                  results_path,nFiles, shuffle ,id_lstm = 1,resetOnAudioEnd=True, returnLabelsAndData=True): #Suponemos que vienen sin el tiempo ya pero con la label

    #OBTAIN DATA PER AUDIO
    samplesPerAudio = int((np.shape(training_data))[0]/nFiles)
    training_data_X = []
    training_data_y = []
    training_data=(cleaner.detectMissedValues(training_data)).values
    for i in range(nFiles):
        trainingDataX_i,trainingDataY_i = prepareData(training_data[samplesPerAudio*i:samplesPerAudio*(i+1),:], timesteps, featuresIni, returnLabelsAndData=returnLabelsAndData)
        if(training_data_X==[]):
            training_data_X = trainingDataX_i
            training_data_y = trainingDataY_i
        else:
            training_data_X = np.concatenate((training_data_X,trainingDataX_i),axis=0)
            training_data_y = np.concatenate((training_data_y, trainingDataY_i), axis=0)
    #training_data_X, training_data_y = prepareData(training_data, timesteps, featuresIni, returnLabelsAndData=True)
    training_data_XShape = np.shape(training_data_X)
    model = deine_LSTMmodelstatefulFalse(**buildModelParameters, stateful=stateful,trainingDataXShape=training_data_XShape,batch_size=batch_size)

    if(dev_data==[]):
        print('no dev')
    else:
        dev_data_X = []
        dev_data_y = []
        dev_data = (cleaner.detectMissedValues(dev_data)).values
        for i in range(nFiles):
            devDataX_i, devDataY_i = prepareData(dev_data[samplesPerAudio * i:samplesPerAudio * (i + 1), :], timesteps, featuresIni,returnLabelsAndData=True)
            if (dev_data_X == []):
                dev_data_X = devDataX_i
                dev_data_y = devDataY_i
            else:
                dev_data_X = np.concatenate((dev_data_X, devDataX_i), axis=0)
                dev_data_y = np.concatenate((dev_data_y, devDataY_i), axis=0)

    if(test_data==[]):
        print("no test set")
    else:
        test_data_X = []
        test_data_y = []
        test_data = (cleaner.detectMissedValues(test_data)).values
        for i in range(nFiles):
            testDataX_i, testDataY_i = prepareData(test_data[samplesPerAudio * i:samplesPerAudio * (i + 1), :], timesteps, featuresIni,
                returnLabelsAndData=True)
            if (test_data_X == []):
                test_data_X = testDataX_i
                test_data_y = testDataY_i
            else:
                test_data_X = np.concatenate((test_data_X, testDataX_i), axis=0)
                test_data_y = np.concatenate((test_data_y, testDataY_i), axis=0)

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=early_stoppingParameters[0],
                                                  patience=early_stoppingParameters[1], verbose=1, mode='auto')
    # START TRAINING TIME
    start_time = time.time()
    # fit network
    # for checking some properties we could do a for loop and writting the epochs into model.fil equal to 1
    if(dev_data==[]):
        if(stateful):
            shuffle = False
            if(resetOnAudioEnd):
                samplesPerAudio = int(training_data_XShape[0]/nFiles)
                for i in range(epochs):
                    print("Epoch: ", i)
                    for k in range(nFiles):
                        trainingX = training_data_X[k*samplesPerAudio:(k+1)*samplesPerAudio]
                        trainingY = training_data_y[k*samplesPerAudio:(k+1)*samplesPerAudio]
                        history = model.fit(trainingX, trainingY, epochs=1, batch_size=batch_size,verbose=1,shuffle=shuffle)
                        model.reset_states() #reset per audio & epoch
            else:#reset on epoch end
                for i in range(epochs):
                    print("Epoch: ", i)
                    history = model.fit(training_data_X, training_data_y, epochs=1, batch_size=batch_size, verbose=1,
                                        shuffle=shuffle)
                    model.reset_states()#reset per epoch
        else:
            history = model.fit(training_data_X, training_data_y, epochs=epochs, batch_size=batch_size,
                                verbose=1,shuffle=shuffle, validation_data=(training_data_X, training_data_y))
    else:
        if(stateful):
            shuffle=False
            if (resetOnAudioEnd):
                samplesPerAudio = int(training_data_XShape[0] / nFiles)
                for i in range(epochs):
                    print("Epoch: ", i)
                    for k in range(nFiles):
                        trainingX = training_data_X[k*samplesPerAudio:(k+1)*samplesPerAudio]
                        trainingY = training_data_y[k*samplesPerAudio:(k+1)*samplesPerAudio]
                        history = model.fit(trainingX, trainingY, epochs=1, batch_size=batch_size,
                                                verbose=1,validation_data=(dev_data_X, dev_data_y),
                                                shuffle=shuffle, callbacks=[earlyStopping])
                        model.reset_states()  # reset per audio & epoch
            else:  # reset on epoch end
                for i in range(epochs):
                    print("Epoch: ", i)
                    history = model.fit(training_data_X, training_data_y, epochs=1, batch_size=batch_size, verbose=1,
                                        shuffle=shuffle)
                    model.reset_states()  # reset per epoch

        else:
            history = model.fit(training_data_X, training_data_y, epochs=epochs, batch_size=batch_size, verbose=1,
                                validation_data=(dev_data_X, dev_data_y),
                                shuffle=shuffle, callbacks=[earlyStopping])

    #RELOAD NETWORK FOR USING batch_size=1 FOR DOING PREDICTIONS:
    #old_weights = model.get_weights()
    #new_model = defineGRU_modelStatefulFalse(**buildModelParameters, stateful=stateful,trainingDataXShape=training_data_XShape,batch_size=1, oldWeigths=old_weights)
    new_model = model
    # TRAINING PREDICTION:
    #for predict single values
    metricsDev = []
    metricsTest = []
    print('------------START TRAIN METRICS---------------')
    ypred_traininig = new_model.predict(training_data_X, batch_size=batch_size)
    #COPY WEIGHTS
    # old_weights = model.get_weights()
    # new_model.set_weights(old_weights)
    labels_training = training_data_y.reshape((len(training_data_y), 1))
    rmse_train, mae_train, PC_train, CCCoeff_train = metrics.getMetrics(ypred_traininig, labels_training, "TRAIN ")
    metricsTraining = [math.sqrt(rmse_train), math.sqrt(rmse_train), mae_train, rmse_train,  PC_train, CCCoeff_train]
    print('------------END TRAIN METRICS---------------')
    print('------------START DEV METRICS---------------')
    if (dev_data == []):
        print("no metrics for dev")
    else:
        ypred_dev = new_model.predict(dev_data_X, batch_size=batch_size)
        labels_dev = dev_data_y.reshape((len(dev_data_y), 1))
        rmse_dev, mae_dev, PC_dev, CCCoeff_dev=metrics.getMetrics(ypred_dev, labels_dev, "DEV ")
        metricsDev = [math.sqrt(rmse_dev), math.sqrt(rmse_dev), mae_dev, rmse_dev, PC_dev, CCCoeff_dev]
    print('------------END DEV METRICS---------------')
    print('------------START TEST METRICS---------------')
    if(test_data==[]):
        print("no metrics for test")
    else:
        ypred_test = new_model.predict(test_data_X, batch_size=batch_size)
        labels_test= test_data_y.reshape((len(test_data_y), 1))
        rmse_test, mae_test, PC_test, CCCoeff_test=metrics.getMetrics(ypred_test, labels_test, "TEST ")
        metricsTest = [math.sqrt(rmse_test),math.sqrt(rmse_test),mae_test,rmse_test,PC_test,CCCoeff_test]
    print('------------END TEST METRICS---------------')

    stopEpoch = len(history.epoch)
    final_time = (time.time() - start_time)
    print("--- %s TRAINING TIME IN seconds ---" % (final_time))
    print("----END EPOCH: ----", stopEpoch)

    return new_model,history, metricsTraining, metricsDev,metricsTest, final_time, epochs


#modality = 'AUDIO' or 'HRHRV'
def nFoldxValidation(features_path, traininigNameList, testNameList, lstmTrainingParameters,
                     stdPerUserOrAllUsers, col2delete, delay, nFolds=7,
                     reducedFeaturePathTrain=[], modality='AUDIO',  epochsFinalModel=0):
    #[nRows, nCols] = data.shape #rows = samples; cols = features
    metricsArray = np.zeros([nFolds + 1, 10])  # 10 metrics: MAE,RMSE,PC,CCC(x2-train&val/test), time, epochs
    # ------START XVALIDATION----------
    # CREATION OF SETS:
    # FEATURE SELECTION ADAPTATION
    if(not reducedFeaturePathTrain==[]):
        headers = loader.loadArffHeadersAsArray(reducedFeaturePathTrain)
        print("features Selected: ",headers)
        dictionary = {k: v for k, v in enumerate(headers)}
        if(modality == 'AUDIO'):
            traininig_attribute_index_list = attributeSelector.obtainAttributesIndexToDeleteAudio(dictionary)
        elif(modality=='HRHRV'):
            traininig_attribute_index_list = attributeSelector.obtainAttributesIndexToDeletHRHRV(dictionary)
        else:
            print('error in modality')
            return -1
        traininig_attribute_index_list=list(map(int, traininig_attribute_index_list.split(',')[0:-1]))
        traininig_attribute_index_list.append(89)#label
        print(traininig_attribute_index_list)  # index of training attributes
        setTotalAttributes = set(range(0,90))
        newCols2Delete = setTotalAttributes.difference(set(traininig_attribute_index_list))
        col2delete = list(newCols2Delete)
        lstmTrainingParameters['trainLSTM']['featuresIni']=len(headers)-1

    # 1. CONCATENATION OF FILES
    training_files, headers, relation = filesCreator.conc_files(features_path, traininigNameList, columns2Delete=col2delete,
                                                                saveFileAsPickle=True)
    print(headers)
    test_files, _, _ = filesCreator.conc_files(features_path, testNameList, columns2Delete=col2delete)
    if (stdPerUserOrAllUsers == 'allUsers'):
        # 2. STANDARIZATION OF FILES
        data_train_std, data_test_std = filesCreator.stand_feat_allUsers(training_files, test_files)
    else:  # per user standarization
        # 2. STANDARIZATION OF FILES
        data_train_std = filesCreator.stand_feat_perUserArray(training_files, len(traininigNameList))
        data_test_std = filesCreator.stand_feat_perUserArray(test_files, len(testNameList))
        relation = "".join(["std_per_User_delay", str(delay), "sec"])

    #---VALIDATION:
    if (not nFolds == 0):
        indexValidation = 0
        epochsFinalModel = 0
        nFiles = len(traininigNameList)
        stepValidation = int(nFiles / nFolds)
        for fold in range(nFolds):
            print("FOLD: ", fold)
            training_data = []
            training_filesNames = []
            validation_data = []
            validation_filesNames = []

            for index_train in range(0, nFiles):
                if ((index_train >= indexValidation) and (index_train < (indexValidation + stepValidation))):
                    # case of all users standarization
                    if (validation_data == []):
                        validation_data = training_files[(7501 * index_train):((index_train + 1) * 7501),:]  # TRAINING FILE CONCATENATED PREVIOUSLY
                    else:
                        validation_data = np.vstack(
                            [validation_data, training_files[(7501 * index_train):((index_train + 1) * 7501), :]])
                    validation_filesNames.append(traininigNameList[index_train])
                else:
                    if (training_data == []):
                        training_data = training_files[(7501 * index_train):((index_train + 1) * 7501),:]  # TRAINING FILE CONCATENATED PREVIOUSLY
                    else:
                        training_data = np.vstack(
                            [training_data, training_files[(7501 * index_train):((index_train + 1) * 7501), :]])
                    training_filesNames.append(traininigNameList[index_train])
            print("validation files: ", validation_filesNames)
            print("traininig files: ", training_filesNames)
            print("concatenated files created!")
            if(stdPerUserOrAllUsers == 'allUsers'):
                # 2. STANDARIZATION OF FILES
                data_trainxval_std, data_validation_std = filesCreator.stand_feat_allUsers(training_data, validation_data)
            else:  # per user standarization
                # 2. STANDARIZATION OF FILES
                data_trainxval_std = filesCreator.stand_feat_perUserArray(training_data, nFiles=len(training_filesNames))
                data_validation_std = filesCreator.stand_feat_perUserArray(validation_data, nFiles=len(validation_filesNames))
            print("standarized files created!")

            model, history, metricsTraining, metricsDev, metricsTest , time,epochs= \
                LSTM_training(training_data=data_trainxval_std, test_data=[], dev_data=data_validation_std,
                             buildModelParameters=lstmTrainingParameters['buildModel'],
                             **lstmTrainingParameters['trainLSTM'], id_lstm=fold, nFiles=len(training_filesNames))
            if (len(history.epoch) > epochsFinalModel):
                epochsFinalModel = len(history.epoch)

            if (len(history.epoch) <= 1):
                num_epochs = epochs
            else:
                num_epochs = len(history.epoch)
            metricsArray[fold+1, 0:8] = np.concatenate([metricsTraining[2::], metricsDev[2::]])
            metricsArray[fold + 1, 8] = num_epochs
            metricsArray[fold + 1, 9] = time
            indexValidation = indexValidation + stepValidation


    #TRAIN MODEL (ALL DATA)
    print("validation files: ", testNameList)
    print("traininig files: ", traininigNameList)
    if (epochsFinalModel <= 0):
        epochsFinalModel = lstmTrainingParameters['trainLSTM']['epochs']
    else:
        epochsFinalModel += 2

    lstmTrainingParameters['trainLSTM']['epochs'] = epochsFinalModel
    if(nFolds==0):
        modelTest, historyTest, metricsTraining, metricsDev, metricsTest, time, epochs=\
        LSTM_training(training_data=data_train_std, test_data=[], dev_data=[],buildModelParameters = lstmTrainingParameters['buildModel'],
                      **lstmTrainingParameters['trainLSTM'], nFiles=len(traininigNameList))
    else: #FINAL MODEL
        modelTest, historyTest, metricsTraining, metricsDev, metricsTest, time, epochs = \
            LSTM_training(training_data=data_train_std, test_data=data_test_std, dev_data=[],
                          buildModelParameters=lstmTrainingParameters['buildModel'],
                          **lstmTrainingParameters['trainLSTM'], nFiles=len(traininigNameList))
    if(len(historyTest.epoch)<=1):
        num_epochs = epochs
    else:
        num_epochs = len(historyTest.epoch)

    if (nFolds == 0):
        metricsArray[0, 0:4] = metricsTraining[2::]
        metricsArray[0, 8] = num_epochs
        metricsArray[0, 9] = time
    else:
        metricsArray[0, 0:8] = np.concatenate([metricsTraining[2::], metricsTest[2::]])
        metricsArray[0, 8] = num_epochs
        metricsArray[0, 9] = time
    #-----END TEST STEP

    # SAVE TEST MODEL THAT HAS BEEN TRAINED WITH THE FULL TRAINING SET AND TESTED ON TEST SET
    model_filename = 'Recola_audio_LSTM_' + str(lstmTrainingParameters['trainLSTM']['dimension'])+\
                     '_numNeur_'+str(lstmTrainingParameters['buildModel']['num_neurons'])+ \
                     '_numFeatures_' + str(lstmTrainingParameters['trainLSTM']['featuresIni']) + \
                     '_timSteps_'+str(lstmTrainingParameters['trainLSTM']['timesteps'])+ \
                     '_stateFul_' + str(lstmTrainingParameters['trainLSTM']['stateful']) + \
                     '_shuffle_'+ str(lstmTrainingParameters['trainLSTM']['shuffle']) + \
                     '_resetOnAudio' + str(lstmTrainingParameters['trainLSTM']['resetOnAudioEnd']) + \
                     '_features_' + str(lstmTrainingParameters['trainLSTM']['featuresIni']) + \
                     '_optimizer_' + lstmTrainingParameters['buildModel']['optimizer'] + \
                     '_lr_' + str(lstmTrainingParameters['buildModel']['learning_rate']) + \
                     '_mom_' + str(lstmTrainingParameters['buildModel']['momentum']) + \
                     '_reg_' + str(lstmTrainingParameters['buildModel']['regularization']) + \
                     '_lossf_' + lstmTrainingParameters['buildModel']['lossFunction'] + \
                     '_batch_' + str(lstmTrainingParameters['trainLSTM']['batch_size']) + \
                     '_realStopEpoch_' + str(len(historyTest.epoch))

    folderSaveModels = lstmTrainingParameters['trainLSTM']['results_path']
    saver.saveModelKeras(modelTest, (folderSaveModels + model_filename), (folderSaveModels + model_filename))

    # SAVE METRICS:
    header = 'Name; MAE_train; RMSE_train; PC_train; CCC_train;' \
             'MAE_valOrTest; RMSE_valOrTest; PC_valOrTest; CCC_valOrTest; Epochs; Time(sec); \n'
    nameModel = model_filename
    filePath =  folderSaveModels + modality+'_resultsXval.csv'
    for metricfoldI in metricsArray:
        saver.save_metrics_to_file(filePath, header, description=nameModel, metrics=metricfoldI)
    avgEpochs = np.mean(metricsArray[:,8])
    return modelTest, model_filename, int(avgEpochs)



def predictNewSampleDelayed(model, testNameList, traininigNameList,
                            features_path, withLabels, saveLabels, labels_path, savePredictions,
                            stdPerUserOrAllUsers, col2delete, delayLabels,
                            outputFolderPredictions, filePredictionsName,
                            saveImagesOfPredictions, delayPred, modality, audioOrHR,
                            stateful, buildModelParameters, timesteps, featuresIni, GRUorLSTM, reducedFeaturePathTrain=[]):
    labels_test = []
    if (not reducedFeaturePathTrain == []):
        headers = loader.loadArffHeadersAsArray(reducedFeaturePathTrain)
        print("features Selected: ", headers)
        dictionary = {k: v for k, v in enumerate(headers)}
        if (audioOrHR == 'AUDIO'):
            traininig_attribute_index_list = attributeSelector.obtainAttributesIndexToDeleteAudio(dictionary)
        elif (audioOrHR == 'HRHRV'):
            traininig_attribute_index_list = attributeSelector.obtainAttributesIndexToDeletHRHRV(dictionary)
        else:
            print('error in modality')
            return -1
        traininig_attribute_index_list = list(map(int, traininig_attribute_index_list.split(',')[0:-1]))
        #if(withLabels):
        traininig_attribute_index_list.append(89)  # label
        print(traininig_attribute_index_list)  # index of training attributes
        setTotalAttributes = set(range(0, 90))
        newCols2Delete = setTotalAttributes.difference(set(traininig_attribute_index_list))
        col2delete = list(newCols2Delete)
        featuresIni= len(headers) - 1


    # 1. CONCATENATION OF FILES
    test_files, _, _ = filesCreator.conc_files(features_path, testNameList, columns2Delete=col2delete)
    if ('gs' in testNameList[0]):
        lab = True
    else:
        lab = False
    if (stdPerUserOrAllUsers == 'allUsers'):
        # 2. STANDARIZATION OF FILES
        training_files, headers, relation = filesCreator.conc_files(features_path, traininigNameList,
                                                                    columns2Delete=col2delete,
                                                                    saveFileAsPickle=True)
        print(headers)
        data_train_std, data_test_std = filesCreator.stand_feat_allUsers(training_files, test_files, withLabels=lab)
    else:  # per user standarization
        # 2. STANDARIZATION OF FILES

        data_test_std = filesCreator.stand_feat_perUserArray(test_files, len(testNameList), withLabels=lab)
        relation = "".join(["std_per_User_delay", str(delayLabels), "sec"])

    nFiles = len(testNameList)
    samplesPerAudio = int((np.shape(data_test_std))[0] / nFiles)
    if (data_test_std == []):
        print("no test set")
    else:
        test_data_X = []
        test_data_y = []
        test_data = (cleaner.detectMissedValues(data_test_std)).values


        for i in range(nFiles):
            testDataX_i, testDataY_i = prepareData(test_data[samplesPerAudio * i:samplesPerAudio * (i + 1), :],
                                                   timesteps, featuresIni,
                                                   returnLabelsAndData=lab, withLabels=lab)
            if (test_data_X == []):
                test_data_X = testDataX_i
                test_data_y = testDataY_i
            else:
                test_data_X = np.concatenate((test_data_X, testDataX_i), axis=0)
                test_data_y = np.concatenate((test_data_y, testDataY_i), axis=0)

    old_weights = model.get_weights()
    test_data_XShape = np.shape(test_data_X)
    if(GRUorLSTM == 'GRU'):
        new_model = defineGRU_modelStatefulFalse(**buildModelParameters, stateful=stateful,trainingDataXShape=test_data_XShape,batch_size=1, oldWeigths=old_weights)
    else:
        new_model = deine_LSTMmodelstatefulFalse(**buildModelParameters, stateful=stateful,
                                                 trainingDataXShape=test_data_XShape, batch_size=1,
                                                 oldWeigths=old_weights)
    print('Starting predictions')
    ypred_test = new_model.predict(test_data_X, batch_size=1)

    pred = ypred_test.reshape(-1)
    print('LABELS PREDICTED: ', str(len(pred)))

    predNew = np.zeros(shape=(np.shape(pred)[0] + timesteps))
    index = 0
    if(delayLabels== '0'):
        for i in range(0, (np.shape(pred)[0]+timesteps)):
            if (i > int((delayPred * 1000) / 40)+timesteps):#+timesteps???
                index += 1
            predNew[i] = pred[index]
    else:
        for i in range(0, (np.shape(pred)[0]+timesteps)):
            if (i > timesteps):
                index += 1
            predNew[i] = pred[index]

    labels_test = []
    isTest = True
    if('dev' in testNameList[0] or 'train' in testNameList[0] and saveLabels):
        print('------------START TEST METRICS---------------')
        isTest = False
        #labels_testv1 = loader.loadMatrixFromMat(labels_path+'gs_delayed_'+delay+'seg.mat')['gold_standard'][0][3][0]
        labels_test = filesCreator.getLabels(labels_path +'gs_delayed_' + delayLabels + 'seg.mat', modality, testNameList[0])
        #labels_test = test_data_y.reshape((len(test_data_y), 1))#data_test_std[:,-1]
        print('file: ',testNameList[0] )
        rmse_test, mae_test, PC_test, CCCoeff_test = metrics.getMetrics(predNew, labels_test, "TEST ")
        metricsTest = [math.sqrt(rmse_test), math.sqrt(rmse_test), mae_test, rmse_test, PC_test, CCCoeff_test]
        print('------------END TEST METRICS---------------')
    if (savePredictions):
        print('SAVING PREDICTIONS...')
        if (saveLabels):
            predNew = np.asarray(predNew)
            true = np.asarray(labels_test)
            predNew = predNew.reshape((-1, 1))
            true = true.reshape((-1, 1))
            x = np.hstack((predNew, true))
        else:
            x = predNew.reshape((-1, 1))

        if (len(testNameList) <= 1):
            path2savePredictions = outputFolderPredictions + testNameList[0]
            saver.saveMatrixAsCSVPredictions(x, path2savePredictions, modality, testNameList[0])
            if (saveImagesOfPredictions and constraints.display):
                path2saveImg = outputFolderPredictions + 'imgs/'
                displayer.paintAndSaveFinalPredictions(predNew, labels_test, path2saveImg, n_graphs=1, info=testNameList[0] + delayLabels, isTest=isTest)
        else:
            print('DEMASIADOS ARCHIVOS')
            # numAudios = len(testNameList)
            # filesPerAudio = int(len(pred) / numAudios)
            # for audioIndex in range(numAudios):
            #     dataAudioi = x[(audioIndex * filesPerAudio):(filesPerAudio * (audioIndex + 1)), :]
            #     nameAudioi = (testNameList[audioIndex * filesPerAudio]).split('_')[1] + \
            #                  (testNameList[audioIndex * filesPerAudio]).split('_')[2]
            #     path2savePredictions = outputFolderPredictions + filePredictionsName + '_' + nameAudioi
            #     saver.saveMatrixAsCSVPredictions(dataAudioi, path2savePredictions , modality, nameAudioi)
            #     if (saveImagesOfPredictions and constraints.display):
            #         path2saveImg = outputFolderPredictions + 'imgs/'
            #         displayer.paintAndSaveFinalPredictions(dataAudioi[:,0], dataAudioi[:,1], path2saveImg, n_graphs=1,
            #                                                info=nameAudioi+delay, isTest=isTest)
    return predNew, labels_test



#
# ---------------FOR PREDICTIONS--------------
# PATH = os.getcwd()
# dimension = "arousal" # or "valence"
# if(dimension=="arousal"):
#     dataset_path_train ='../data/dataset_Recola/delay_2con8sec/stand_train_withoutFrameTime.arff'
#     dataset_path_dev = '../data/dataset_Recola/delay_2con8sec/stand_dev_withoutFrameTime.arff'
# elif(dimension=="valence"):
#     dataset_path_train = '../data/dataset_Recola/delay_3con6sec/valence/stand_train_withoutFrameTime.arff'
#     dataset_path_dev = '../data/dataset_Recola/delay_3con6sec/valence/stand_dev_withoutFrameTime.arff'
#
# results_path = PATH+'/data/results/LSTM-GRU_audioFeatures/'
# filepath_pred_training = PATH+'/data/results/LSTM-GRU_audioFeatures/LSTM_predictions_training.csv'
# filepath_pred_dev = PATH+'/data/results/LSTM-GRU_audioFeatures/LSTM_predictions_dev.csv'
# -----------END PREDICTIONS PART-------------
#
#
# PATH = os.getcwd()
# dimension = "arousal" # or "valence"
# if(dimension=="arousal"):
#     dataset_path_train =PATH+'/data/dataset_Recola/delay_2con8sec/stand_train_withoutFrameTime.arff'
#     dataset_path_dev = PATH+'/data/dataset_Recola/delay_2con8sec/stand_dev_withoutFrameTime.arff'
# elif(dimension=="valence"):
#     dataset_path_train = PATH+'/data/dataset_Recola/delay_3con6sec/valence/stand_train_withoutFrameTime.arff'
#     dataset_path_dev = PATH+'/data/dataset_Recola/delay_3con6sec/valence/stand_dev_withoutFrameTime.arff'
#
# results_path = PATH+'/data/results/LSTM-GRU_audioFeatures/'
#
# # lr = [0.0001, 0.00001]
# # batches = [10,32,75]#batches = [75 ,128, ]
# # neurons = [50,80,100,120]#, 80, 100, 120
# # timeStep = [10,20,30]
# # n_epochs = [700]#500
# featuresIni = 88
# earlyStopping_param1 = [0.000001]
# l2_reg = [0.01]
# lr = [0.0001]
# neurons = [50,80,100]
# timeStep = [5,10,20]
# n_epochs = [700]
# batches = [10,32,75]
#
# # #LOAD DATA FROM PICKLE:
# # data = saver.importMatrixfromPICKLE(filepath_pickle, name_pickle)
# # dev_data = saver.importMatrixfromPICKLE(filepath_pickle, dev_name_pickle)
# #
# training_data = reader.obtainArffAsArray(dataset_path_train,[])
# dev_data = reader.obtainArffAsArray(dataset_path_dev,[])
#
#
#
# for learning_rate in lr:
#     sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)  # decay=1e-6,
#     adam = optimizers.Adam(lr=learning_rate,beta_1=0.9, beta_2=0.1)  # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     adamMax = optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.1, epsilon=1e-08, decay=0.0)
#     rms = optimizers.RMSprop(lr=learning_rate, rho=0.9)
#
#     opt = [sgd]
#     #opt = [sgd, adam, rms]
#     for optimizer in opt:
#         if (optimizer == adam):
#             opt_name = "".join(['adam_lr_', str(learning_rate)])
#         elif(optimizer==sgd):
#             opt_name = "".join(['sgd_lr_', str(learning_rate)])
#         elif (optimizer == adamMax):
#             opt_name = "".join(['adamMax_lr_', str(learning_rate)])
#         elif (optimizer == rms):
#             opt_name = "".join(['rms_lr_', str(learning_rate)])
#         else:
#             opt_name = "UNKNOWN"
#
#         for num_neurons in neurons:
#             for batch_size in batches:
#                 for timesteps in timeStep:
#                     for epochs in n_epochs:
#                         for param1 in earlyStopping_param1:
#                             for reg in l2_reg:
#                                 LSTM_training(training_data, dev_data, dimension, timesteps, featuresIni, num_neurons,
#                                               learning_rate, reg, optimizer, epochs, batch_size, param1, results_path,opt_name,
#                                               id_lstm=1, test_data=[])

#                                 # callbacks
#                                 earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=param1,
#                                                                               patience=2, verbose=1, mode='auto')
#                                 print('LSTM',str(num_neurons), '_timesteps_', str(timesteps), '_optimizer_', opt_name,'_batch_', str(batch_size), '_epochs_', str(epochs))
#                                 reframed = series_to_supervised(training_data, timesteps, 1)
#                                 dev_reframed = series_to_supervised(dev_data, timesteps, 1)
#
#                                 # # drop columns we don't want to predict
#                                 reframed.drop(reframed.columns[[featuresIni]], axis=1, inplace=True)
#                                 dev_reframed.drop(dev_reframed.columns[[featuresIni]], axis=1, inplace=True)
#                                 column_remove = featuresIni
#                                 for i in range(timesteps-1):
#                                     column_remove = column_remove+featuresIni
#                                     reframed.drop(reframed.columns[[column_remove]], axis=1, inplace=True)
#                                     dev_reframed.drop(dev_reframed.columns[[column_remove]], axis=1, inplace=True)
#                                 #print(reframed.head())
#
#                                 train_values = reframed.values
#                                 dev_values = dev_reframed.values
#                                 train_X, train_y = train_values[:, :-1], train_values[:, -1]
#                                 dev_X, dev_y = dev_values[:, :-1], dev_values[:, -1]
#                                 # reshape input to be 3D [samples, timesteps, features]
#                                 train_X = train_X.reshape((train_X.shape[0], timesteps+1, featuresIni))
#                                 dev_X = dev_X.reshape((dev_X.shape[0], timesteps+1, featuresIni))
#                                 #If we use statefil=True, the number of train and test examples have to be the same to fix with the number of epochs.
#                                 #for this reason, we use dev_X_modi and dev_y_modi
#                                 dev_X_modi = dev_X[0:train_X.shape[0], :, :]
#                                 dev_y_modi = dev_y[0:train_X.shape[0]]
#                                 #print(train_X.shape, train_y.shape)
#
#
#                                 # design network
#                                 model = Sequential()
#                                 model.add(LSTM(num_neurons, input_shape=(train_X.shape[1], train_X.shape[2]),stateful=False, kernel_regularizer=keras.regularizers.l2(reg)))#Test stateful: default: False
#                                 #If the model is stateless, the cell states are reset at each sequence. With the stateful model, all the states are propagated to the next batch.
#                                 #  It means that the state of the sample located at index i, Xi will be used in the computation of the sample Xi+bs in the next batch,
#                                 # where bs is the batch size (no shuffling).
#                                 model.add(Dense(1,activation='tanh',kernel_regularizer=keras.regularizers.l2(reg)))
#                                 model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])
#                                 # fit network
#                                 #for checking some properties we could do a for loop and writting the epochs into model.fil equal to 1
#                                 history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=1,validation_data=(dev_X_modi, dev_y_modi),
#                                                     shuffle=False,callbacks=[earlyStopping]) #, callbacks=[earlyStopping]
#
#                                 print('STOP AT EPOCH: ',len(history.epoch))
#                                 stopEpoch = len(history.epoch)
#                                 # plot history
#                                 # pyplot.plot(history.history['loss'], label='train')
#                                 # #pyplot.plot(history.history['val_loss'], label='test')
#                                 # pyplot.legend()
#                                 # pyplot.show()
#
#                                 #TRAINING PREDICTION:
#                                 ypred_traininig = model.predict(train_X, batch_size=batch_size)
#                                 labels_training = train_y.reshape((len(train_y), 1))
#
#                                 rmse_training = sqrt(mean_squared_error(labels_training, ypred_traininig))
#                                 print('Train RMSE: %.3f' % rmse_training)
#                                 pred = ypred_traininig.reshape(-1)
#                                 true = labels_training.reshape(-1)
#                                 pearson_coeff_training = np.corrcoef(pred, true)[0, 1]
#                                 mae_training = metrics.mae(true, pred)
#                                 print('Train MAE :', mae_training)
#                                 print('Train CC:', pearson_coeff_training)
#                                 # P_cc = scipy.stats.pearsonr(pred, true)
#                                 # print('Test CC (scipy):', pearson_coeff)
#                                 CCCoeff_training = CCC(true, pred)
#                                 print('Train CCC :', CCCoeff_training)
#
#                                 with open(filepath_pred_training, "a") as predFile:
#                                     predFile.write('predictions, complete_labels')
#                                     predFile.write('\n')
#                                     for i in range(pred.shape[0]):
#                                         values = "".join([str(pred[i]), ';',str(true[i])])
#                                         predFile.write(values)
#                                         predFile.write('\n')
#                                     predFile.close()
#
#
#
#                                 #TEST
#                                 # make a prediction
#                                 yhat = model.predict(dev_X_modi, batch_size=batch_size)
#                                 # invert scaling for forecast
#
#                                 # invert scaling for actual
#                                 test_y = dev_y_modi.reshape((len(dev_y_modi), 1))
#
#
#                                 # calculate RMSE
#                                 rmse = sqrt(mean_squared_error(test_y, yhat))
#                                 print('Test RMSE: %.3f' % rmse)
#                                 pred = yhat.reshape(-1)
#                                 true = test_y.reshape(-1)
#                                 pearson_coeff = np.corrcoef(pred, true)[0,1]
#                                 mae = metrics.mae(true,pred)
#                                 print('Test MAE :', mae)
#                                 print('Test CC:', pearson_coeff)
#                                 #P_cc = scipy.stats.pearsonr(pred, true)
#                                 #print('Test CC (scipy):', pearson_coeff)
#                                 CCCoeff = CCC(true, pred)
#                                 print('Test CCC :', CCCoeff)
#
#                                 with open(filepath_pred_dev, "a") as predFile:
#                                     predFile.write('predictions, complete_labels')
#                                     predFile.write('\n')
#                                     for i in range(pred.shape[0]):
#                                         values = "".join([str(pred[i]), ';',str(true[i])])
#                                         predFile.write(values)
#                                         predFile.write('\n')
#                                     predFile.close()
#
#
#
#                                 results_file = "".join([results_path, 'results.txt'])
#                                 F = open(results_file ,'a')
#                                 metrics_results = "".join(['LSTM_',dimension,'_',str(num_neurons),'_timesteps_',str(timesteps),'_optimizer_',opt_name, '_batch_',str(batch_size), '_epochs_',str(epochs),'_realStopEpoch_',str(stopEpoch),'_earlyStopParam1_',str(param1),'_L2_',str(reg),\
#                                                   ', TRAINING:',str(rmse_training),', ',str(pearson_coeff_training), ', ',str(mae_training), ', ',str(CCCoeff_training),',DEV:',str(rmse), ', ',str(pearson_coeff), ', ',str(mae), ', ',str(CCCoeff)])
#
#                                 F.write(metrics_results)
#                                 F.write('\n')
#                                 F.close()
#
#                                 model_filename = PATH + '/data/results/models_trained/Recola_features_audio__LSTM_funcionals_optimizer_sgd__neurons' + str(num_neurons) +'_timeSteps_'+str(timesteps)+ '_lr_' + str(
#                                     learning_rate) + '_lossf_' + 'mse' + '_batch_' + str(batch_size)+dimension
#
#
#                                 json_filename = model_filename + '.json'
#                                 # serialize model to JSON
#                                 model_json = model.to_json()
#                                 with open(json_filename, "w") as json_file:
#                                     json_file.write(model_json)
#                                 # serialize weights to HDF5
#                                 model.save_weights(model_filename + '.h5')
#
#
#
