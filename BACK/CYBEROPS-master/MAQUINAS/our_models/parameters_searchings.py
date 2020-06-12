import os
import glob
import pandas as pd
import csv

def get_best_model(root_path_models, metric_2_eval="NPV"):

    parameter = []
    NPV = []
    file = []

    models = glob.glob(root_path_models + "*.csv")
    flag = 0
    for model in models:
        # print(model)
        df_values = pd.read_csv(model, sep='\s*,\s*',header=0, encoding='ascii', engine='python')
        df_values.fillna(-1, inplace=True)
        value_metric = df_values[(metric_2_eval+"_test")].values[0]
        model_parameters = df_values.columns[0]


#         NPV_value = df[df.columns[10]]
#         print(parameters_value)
#         print(NPV_value)
#         if NPV_value[0] == "":
#             NPV_value[0] = -1
#
        if flag==0:
            parameter.append(model_parameters)
            NPV.append(value_metric)
            file.append(model)
            flag=1
        else:
            if float(value_metric)>float(NPV[0]):
                parameter.pop()
                NPV.pop()
                file.pop()
                parameter.append(model_parameters)
                NPV.append(value_metric)
                file.append(model)

    print(parameter, NPV, model)

    list = parameter[0].split("_")

    key, value = [], []

    for x in list:
        index = list.index(x)
        if index%2 == 1:
            value.append(x)
        else:
            key.append(x)

    dictionary = dict(zip(key, value))
    print(dictionary)

PATH = os.getcwd().rsplit("/", 1)[0]
root_path_models = PATH + "/our_models/trained_models/SVM/"
get_best_model(root_path_models, metric_2_eval="NPV")