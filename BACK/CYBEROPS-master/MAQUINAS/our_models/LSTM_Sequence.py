from  numpy import array
from  numpy import cumsum
from  keras.models import Sequential
from  keras.layers import LSTM
from  keras.layers import Dense
from  keras.layers import TimeDistributed
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
import random


# convert time series into supervised learning problem
def series_to_supervised(raw_values, raw_label, n_in, n_out, dropnan=True):
	n_vars = 1 if type(raw_values) is list else raw_values.shape[1]
	df = DataFrame(raw_values)
	df_label= DataFrame(raw_label)
	cols, names, labels = list(), list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):  # (terms t-i) CREATING OF THE SUB-DATASETS WITH SAMPLES SHIFTED i TIMESTEPS
		cols.append(df.shift(i))
		# if i==n:
		labels.append(df_label.shift(i))
		names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):  # terms (t ot (t+i))
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
			labels.append(df_label.shift(-i))
		else:
			names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
			labels.append(df_label.shift(-i))
	# put it all together
	# names= names.append("label")
	agg = concat(cols, axis=1)  # concatenate in columns for having: t-i....t....t+i
	agg2 = concat(labels, axis=1)
	agg.columns = names
	agg2.columns = ["lab(t-3)", "lab(t-2)", "lab(t-1)", "lab(t)"]

	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
		agg2.dropna(inplace=True)
	#merge the 2 Dataframes
	# agg2=agg2["var1(t)"]
	# agg2=agg2.to_frame()
	# agg2.columns = ['label_t']
	merge= concat([agg, agg2], axis=1)
	return merge



# transform series into train and test sets for supervised learning
def prepare_data(series, n_lag, n_seq):
	# extract raw values
	raw_values = series["value"].values
	raw_values = raw_values.reshape(len(raw_values), 1)
	# transform into supervised learning problem X, y
	raw_label=series["label"].values
	raw_label = raw_label.reshape(len(raw_label), 1)
	supervised = series_to_supervised(raw_values, raw_label, n_in, n_out)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = train_test_split(supervised, test_size=0.15)
	# X= supervised[["var1(t-3)", "var1(t-2)", "var1(t-1)", "var1(t)"]]
	# y=supervised["label_t"]
	# sss = ShuffleSplit(y, random_state=0, train_size=0.9)
    #
	# for train_index, test_index in sss.split(X, y):
	# 	print("TRAIN:", train_index, "TEST:", test_index)
	# 	X_train, X_test = X[train_index], X[test_index]
	# 	y_train, y_test = y[train_index], y[test_index]
	return train, test

series = read_csv(
    '/home/marta/PycharmProjects/CYBEROPS/MAQUINAS/Labeled_data/ambient_temperature_system_failure_labeled.csv',
    header=0, parse_dates=[0], index_col=0, squeeze=True)

n_in = 3
n_out = 1
n_timesteps = n_in+n_out

train, test = prepare_data(series, n_in, n_out)
# train = (train.values).reshape(1, 4, 1)

X_train= (train[["var1(t-3)", "var1(t-2)", "var1(t-1)", "var1(t)"]]).values
y_train= (train[["lab(t-3)", "lab(t-2)", "lab(t-1)", "lab(t)"]]).values
X_test= (test[["var1(t-3)", "var1(t-2)", "var1(t-1)", "var1(t)"]]).values
y_test= (test[["lab(t-3)", "lab(t-2)", "lab(t-1)", "lab(t)"]]).values

X_train_reshape=[]
X_test_reshape=[]
y_train_reshape=[]
y_test_reshape=[]

for row in X_train:
    row_reshape= row.reshape(1, n_timesteps,1)
    X_train_reshape.append(row_reshape)

for row in X_test:
    row_reshape= row.reshape(1, n_timesteps,1)
    X_test_reshape.append(row_reshape)

for row in y_train:
    row_reshape= row.reshape(1, n_timesteps,1)
    y_train_reshape.append(row_reshape)

for row in y_test:
    row_reshape= row.reshape(1, n_timesteps,1)
    y_test_reshape.append(row_reshape)

# X = X_train.reshape(1, n_timesteps, 1)
# y = y_train.reshape(1, n_timesteps, 1)

# define LSTM
model = Sequential()
model.add(LSTM(8, input_shape=(n_timesteps, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

for epoch in range (1000):
# fit model for one epoch on this sequence
    model.fit(X_train_reshape[0], y_train_reshape[0], epochs = 1, batch_size = 1, verbose = 2)

# evaluate LSTM
yhat = model.predict_classes(X_test_reshape[1000], verbose = 0)
for i in range(n_timesteps):
    print('Expected:', y_test[0, i], 'Predicted', yhat[0, i])

# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/