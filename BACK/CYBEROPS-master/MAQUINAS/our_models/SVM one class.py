import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from pandas import read_csv
from sklearn.model_selection import train_test_split

# transform series into train and test sets for supervised learning
def prepare_data(series):
    train = series[series['label'] == 0]
    test = series[series['label'] == 1]
    # train, test = train_test_split(series, test_size=0.15)
    return train, test

series = read_csv('/home/marta/PycharmProjects/CYBEROPS/MAQUINAS/Labeled_data/ambient_temperature_system_failure_labeled.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)

train, test = prepare_data(series)
X_train= (train["value"]).values
y_train= (train["label"]).values
X_test= (test["value"]).values
y_test= (test["label"]).values
X_train = np.reshape(X_train, (-1, 1))
X_test = np.reshape(X_test, (-1, 1))
# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# X_train.reshape(1, -1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size


