import sklearn, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.utils import shuffle
import numpy as np
import UTILS.normalization as normalize
import UTILS.metrics_classification as metrics
import pickle
from datetime import datetime
from sklearn.externals import joblib


def SVM_normal(path_data_train_with_labels, dateparse, shuffle_data=True,ten_fold_xval=True, svm_parameters={}, normal_class = 0,anomaly_class = 1, path_data_test_with_labels="", normalization=True):
    df_train = pd.read_csv(path_data_train_with_labels, parse_dates=['timestamp'], header=0, index_col=0, squeeze=True)
    #check if we have test set, if not, we use training set as test set
    if (not path_data_test_with_labels == ""):
        df_test = pd.read_csv(path_data_train_with_labels, parse_dates=['timestamp'], header=0, index_col=0, squeeze=True)
    else:
        df_test = df_train
        print("TEST DATA IS EQUAL TO TRAINING DATA")

    #convert anomaly & normal labels to: -1(anomaly), 1(normal)
    if(not anomaly_class == -1):
        df_train.loc[df_train["label"] == anomaly_class, "label"] = -1
        df_test.loc[df_test["label"] == anomaly_class, "label"] = -1
    if(not normal_class == 1):
        df_train.loc[df_train["label"] == normal_class, "label"] = 1
        df_test.loc[df_test["label"] == normal_class, "label"] = 1

    #randomize data
    if(shuffle_data):
        df_train = shuffle(df_train)

    #split normal and anomalies (training set)
    only_normal_data = df_train.loc[df_train["label"]==1]
    X_train_normal = (only_normal_data["value"].values).reshape(-1, 1)

    #only_anomalies = df_train.loc[df_train["label"]==-1]
    #X_anomaly = (only_anomalies["value"].as_matrix()).reshape(-1,1)

    X_train_complete = (df_train["value"].values).reshape(-1,1)
    X_train_labels_complete = (df_train["label"].values).reshape(-1,1)

    # test set
    X_test_complete = (df_test["value"].values).reshape(-1, 1)
    X_test_labels_complete = (df_test["label"].values).reshape(-1, 1)

    #creation and training of model
    if(svm_parameters=={}):
        svm_parameters = {
            'nu': 0.1,
            'kernel': 'rbf',
            'gamma': 0.1,
            'coef0': 0.0,
            'tol': 0.001,
            'shrinking': True,
            'max_iter': -1,
        }
    model = svm.OneClassSVM(nu=svm_parameters['nu'], kernel=svm_parameters['kernel'],
                            gamma=svm_parameters['gamma'], coef0=svm_parameters['coef0'],
                            tol=svm_parameters['tol'], shrinking=svm_parameters['shrinking'],
                            max_iter=svm_parameters['max_iter'])
    # xvalidation
    avg = []
    std = []
    if (ten_fold_xval):
        nFolds = 10
        metrics_array = np.zeros(shape=(nFolds,10))#8 metrics to save
        dict_index_fold = xval_index(X_train_normal, nFolds)
        for fold in range(nFolds):
            validation_ini_index = dict_index_fold[fold]["validation_ini_index"]
            validation_end_index = dict_index_fold[fold]["validation_end_index"]
            training_data = np.vstack([X_train_normal[0:validation_ini_index, :], X_train_normal[validation_end_index:, :]])
            validation_data = X_train_normal[validation_ini_index:validation_end_index, :]
            #NORMALIZATION
            if(normalization):
                avg_training, std_training, min_training, max_training = normalize.get_statistics_from_data(training_data, axis_attributes=0)
                zscore_data_training = normalize.getZscoreData(training_data, avg_training, std_training,zscoreThreshold=None, withLabels=False)
                zscore_data_validation = normalize.getZscoreData(validation_data, avg_training, std_training,zscoreThreshold=None, withLabels=False)
            else:
                zscore_data_training = training_data
                zscore_data_validation = validation_data
            # training
            model.fit(zscore_data_training)
            #predictions and testing model without outliers
            y_pred_data_train = (model.predict(zscore_data_training)).reshape(-1,1)# normal prediction correct -> 1; anomaly prediction correct -> -1
            y_pred_data_val = (model.predict(zscore_data_validation)).reshape(-1,1)  # normal prediction correct -> 1; anomaly prediction correct -> -1
            train_labels = np.ones(shape=np.shape(y_pred_data_train), dtype=np.float32)
            val_labels = np.ones(shape=np.shape(y_pred_data_val), dtype=np.float32)
            print("FOLD: "+str(fold)+"---------------------------------------")
            precission_train, recall_train, accuracy_train, F1_train, _,\
            precission_test, recall_test, accuracy_test, F1_test, _ , NPV_train, NPV_test=\
            metrics.get_summary_metrics(y_pred_data_train, train_labels, y_pred_data_val, val_labels, classes=[1, -1],saveData=False, path2save="")
            #add metrics into metrics matrix
            metrics_array[fold,:] = [precission_train, recall_train, accuracy_train, F1_train, NPV_train, precission_test, recall_test, accuracy_test, F1_test, NPV_test]
        avg = np.average(metrics_array, axis=0)
        std = np.std(metrics_array, axis=0)
        print("-------------------FINAL RESULTS XVAL:---------------------------------")
        print("precission_train, recall_train, accuracy_train, F1_train,NPV_train, precission_test, recall_test, accuracy_test, F1_test, NPV_test")
        print("Avg: "+str(avg))
        print("Standard deviation: "+str(std))
        #save metrics:


    #NORMALIZATION/standarization
    # final training (with all the normal data -> more precisse)
    if(normalization):
        avg_training, std_training, min_training, max_training = normalize.get_statistics_from_data(X_train_normal,axis_attributes=0)
        zscore_data_training_normal = normalize.getZscoreData(X_train_normal, avg_training, std_training, zscoreThreshold=None,withLabels=False)
        zscore_training_complete = normalize.getZscoreData(X_train_complete, avg_training, std_training,
                                                               zscoreThreshold=None, withLabels=False)
        zscore_test_complete = normalize.getZscoreData(X_test_complete, avg_training, std_training,
                                                           zscoreThreshold=None, withLabels=False)
    else:
        zscore_data_training_normal = X_train_normal
        zscore_training_complete = X_train_complete
        zscore_test_complete = X_test_complete
    #normalization of all the others (normal & anomalies) -> we use the statistics of normal data because we want to increase the difference with outliers
    # predictions and testing model (with normal and anomalies)
    y_pred_data_train = (model.predict(zscore_training_complete)).reshape(-1,1)  # normal prediction correct -> 1; anomaly prediction correct -> -1
    y_pred_data_test = (model.predict(zscore_test_complete)).reshape(-1,1)  # normal prediction correct -> 1; anomaly prediction correct -> -1
    print("FINAL MODEL---------------------------------------")
    metrics.get_summary_metrics(y_pred_data_train, X_train_labels_complete, y_pred_data_test,
                                X_test_labels_complete, classes=[1, -1], saveData=False, path2save="")
    return model, avg, std


def xval_index(training_data_dataframe, n_folds):
    shape_training_data = training_data_dataframe.shape
    #index_samples = shape_training_data.index(max(shape_training_data))
    num_samples = max(shape_training_data)
    num_samples_per_fold = int(num_samples / n_folds)
    rest = num_samples % n_folds
    if (rest > 0):
        validation_end_index = num_samples_per_fold + 1
    else:
        validation_end_index = num_samples_per_fold

    dict_index_fold = {0:{
                            "validation_ini_index":0,
                            "validation_end_index":validation_end_index
                            }
    }
    for fold in range(n_folds-1):
        if (rest > fold + 1):
            validation_ini_index = validation_end_index
            validation_end_index = validation_end_index + num_samples_per_fold + 1
        else:
            validation_ini_index = validation_end_index
            validation_end_index = validation_end_index + num_samples_per_fold
        dict_index_fold[fold+1] = {"validation_ini_index":validation_ini_index,"validation_end_index":validation_end_index}
    return dict_index_fold


if __name__ == "__main__":
    PATH = os.getcwd().rsplit("/",1)[0]
    folder_save_models = os.path.join(PATH, "our_models/trained_models/SVM/")
    if (not os.path.exists(folder_save_models)):
        os.makedirs(folder_save_models)
    folder_DS = os.path.join(PATH, "Labeled_data/")
    # listDS_names = os.listdir(folder_DS)
    # for dataset_name in listDS_names:
    #     dataset_name = dataset_name.split(".csv")[0]
    dataset_name = "ambient_temperature_system_failure_labeled"
    path_DS = os.path.join(folder_DS, dataset_name+".csv")

    #parameters model- CON QUE VAYAS VARIANDO LAS LISTAS Y LOS PARÁMETROS ANTES DEL FOR YA SE IRÍAN ENTRENANDO LOS MODELOS (ARRIBA PUEDES IR ESCRIBIENDO EN 'dataset_name' EL NOMBRE DEL DATASET O DESCOMENTAR EL FOR PARA QUE ITERE Y VAYA PROBANDO CON TODOS LOS MISMOS PARÁMETROS
    # TO DO: DEPUÉS DE PROBAR VARIAS COMBIANCIONES YO DIRÍA QUE HABRÍA QUE QUEDRSE CON EL QUE SALE MEJOR PARA EL NPV (EL MÁS ALTO) Y LUEGO VER CÓMO DAN EN ACCURACY Y PRECISSION
    # PARA EL TO-DO DE ARRIBA ESTARÍA BIEN CREARSE UN FOR QUE ENTRE EN LOS FICHEROS DE MÉTRICAS (EN LA CARPETA /our models/trained_models/*.csv una vez corramos experimentos) Y QUE LEA LOS CSVs Y VAYA BUSCANDO EL MÁXIMO

    kernel_list = ["rbf", "poly", "linear", "sigmoid"]
    gamma_list = ['auto', 'scale', 0.1, 1, 10, 100]
    nu_list = [0.1, 0.25, 0.5, 0.75, 1]
    coef0_list = [-1, 0, 1]
    tol = 0.001
    shrinking = True
    normalization = True # check if False better
    max_iter = -1
    for kernel in kernel_list:
        for gamma in gamma_list:
            for coef0 in coef0_list:
                for nu in nu_list:
                    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
                    svm_parameters = {
                            'nu': nu,
                            'kernel': kernel,
                            'gamma': gamma,
                            'coef0': coef0,
                            'tol': tol,
                            'shrinking': shrinking,
                            'max_iter': max_iter,
                        }

                    shuffle_data=True
                    model, avg, std = SVM_normal(path_DS,0, shuffle_data, svm_parameters=svm_parameters, ten_fold_xval=True, normalization=normalization)

                    # save the model
                    current_time = datetime.now()
                    path_model = os.path.join(folder_save_models,dataset_name+"_"+current_time.strftime("%Y%m%d-%H%M%S")+".sav")
                    pickle.dump(model, open(path_model, 'wb'))
                    # save metrics
                    path_metrics = os.path.join(folder_save_models, "AVGmetrics_"+dataset_name + "_" + current_time.strftime("%Y%m%d-%H%M%S") + ".csv")
                    with open(path_metrics, "w") as f:
                        f.write('nu_'+str(svm_parameters["nu"])+'_'+
                                           'kernel_'+svm_parameters["kernel"]+'_'+
                                           'gamma_'+str(svm_parameters["gamma"])+'_'+
                                           'coef0_'+str(svm_parameters["coef0"])+'_'+
                                           'tol_'+str(svm_parameters["tol"])+'_'+
                                           'shrinking_'+str(svm_parameters["shrinking"])+'_'+
                                           'max-iter_'+str(svm_parameters["max_iter"])+ '_'+
                                            'normalization_'+str(normalization)+', '+
                                'precission_train, recall_train, accuracy_train, F1_train, NPV_train, precission_test, recall_test, accuracy_test, F1_test, NPV_test \n')
                        f.write("-1, ")
                        list_avg = list(avg)
                        for idx in range(len(list_avg)-1):
                            f.write(str(list_avg[idx])+",")
                        f.write(str(list_avg[-1])+"\n")


                    #np.savetxt(X =avg, fname=path_metrics)
                    # load the model
                    # loaded_model = joblib.load(path_model)