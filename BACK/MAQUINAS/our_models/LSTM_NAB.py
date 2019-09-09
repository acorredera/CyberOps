from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers.embeddings import Embedding
import math
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# convert time series into supervised learning problem
def series_to_supervised(data, raw_label, n_in, n_out, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
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
	agg2.columns = names

	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
		agg2.dropna(inplace=True)
	#merge the 2 Dataframes
	agg2=agg2["var1(t)"]
	agg2=agg2.to_frame()
	agg2.columns = ['label_t']
	merge= concat([agg, agg2], axis=1)
	return merge


# transform series into train and test sets for supervised learning
def prepare_data(series, n_in, n_out):
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

# # make a persistence forecast
# def persistence(last_ob, n_seq):
# 	return [last_ob for i in range(n_seq)]
#
#
# # evaluate the persistence model
# def make_forecasts(train, test, n_lag, n_seq):
# 	forecasts = list()
# 	for i in range(len(test)):
# 		X, y = test[i, 0:n_lag], test[i, n_lag:]
# 		# make forecast
# 		forecast = persistence(X[-1], n_seq)
# 		# store the forecast
# 		forecasts.append(forecast)
# 	return forecasts
#
#
# # evaluate the RMSE for each forecast time step
# def evaluate_forecasts(test, forecasts, n_lag, n_seq):
# 	for i in range(n_seq):
# 		actual = test[:, (n_lag + i)]
# 		predicted = [forecast[i] for forecast in forecasts]
# 		rmse = sqrt(mean_squared_error(actual, predicted))
# 		print('t+%d RMSE: %f' % ((i + 1), rmse))
#
#
# # plot the forecasts in the context of the original dataset
# def plot_forecasts(series, forecasts, n_test):
# 	# plot the entire dataset in blue
# 	pyplot.plot(series.values)
# 	# plot the forecasts in red
# 	for i in range(len(forecasts)):
# 		off_s = len(series) - n_test + i - 1
# 		off_e = off_s + len(forecasts[i]) + 1
# 		xaxis = [x for x in range(off_s, off_e)]
# 		yaxis = [series.values[off_s]] + forecasts[i]
# 		pyplot.plot(xaxis, yaxis, color='red')
# 	# show the plot
# 	pyplot.show()

# load dataset
series = read_csv('/home/marta/PycharmProjects/CYBEROPS/MAQUINAS/Labeled_data/ambient_temperature_system_failure_labeled.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# #summarize first few rows
# print(series.head())
# #line plot
# series.plot()
# pyplot.show()

# configure
n_in = 3
n_out = 1
# n_lag = 4 #n_lag = n_in
# n_seq = 4 #n_seq = n_out
# prepare data
train, test = prepare_data(series, n_in, n_out)
X_train= (train[["var1(t-3)", "var1(t-2)", "var1(t-1)", "var1(t)"]]).values
y_train= (train[["label_t"]]).values
X_test= (test[["var1(t-3)", "var1(t-2)", "var1(t-1)", "var1(t)"]]).values
y_test= (test[["label_t"]]).values
# print(test)
# print('Train: %s, Test: %s' % (train.shape, test.shape))
# make forecasts
# forecasts = make_forecasts(train, test, n_lag, n_seq)
# # evaluate forecasts
# evaluate_forecasts(test, forecasts, n_lag, n_seq)
# # plot forecasts
# plot_forecasts(series, forecasts, len(test)+2)


# # define LSTM
# model = Sequential()
# model.add(LSTM(8, input_shape=(4, 1), return_sequences=True))
# model.add(TimeDistributed(Dense(1, activation='sigmoid')))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# create the model
# embedding_vecor_length = 32
# values size
values_size=math.ceil((series.max())["value"])
# Simple LSTM
simple_model = Sequential()
simple_model.add(Embedding(values_size, 32))
simple_model.add(LSTM(100))
simple_model.add(Dense(1, activation='sigmoid'))
simple_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
simple_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
simple_model_scores = simple_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy of model %s %s: %.2f%%\n" % (simple_model, "simple_model", (simple_model_scores[1]*100)))

# LSTM with Dropout
dropout_model = Sequential()
dropout_model.add(Embedding(values_size, 32))
# ------- This piece... --------
dropout_model.add(Dropout(0.2))
dropout_model.add(LSTM(100))
dropout_model.add(Dropout(0.2))
# ------- ...is the same as:-------
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
dropout_model.add(Dense(1, activation='sigmoid'))
dropout_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
dropout_model.fit(X_train, y_train, epochs=3, batch_size=64)
dropout_model_scores = dropout_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy of model %s %s: %.2f%%\n" % (dropout_model, "dropout_model", (dropout_model_scores[1]*100)))

# LSTM and Convolutional Neural Network
CNN_model = Sequential()
CNN_model.add(Embedding(values_size, 32))
CNN_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
CNN_model.add(MaxPooling1D(pool_size=2))
CNN_model.add(LSTM(100))
CNN_model.add(Dense(1, activation='sigmoid'))
CNN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
CNN_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
CNN_model_scores = CNN_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy of model %s %s: %.2f%%\n" % (CNN_model, "CNN_model", (CNN_model_scores[1]*100)))

for model in [simple_model,dropout_model,CNN_model]:
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy of model %s: %.2f%%" % (model, (scores[1]*100)))

# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/