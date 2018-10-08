
from __future__ import print_function
import csv
import pandas as pd
import tensorflow as tf
import numpy as np
import math
import tflearn
from datetime import datetime
from tflearn.datasets import titanic
from tflearn.data_utils import load_csv, to_categorical

# The file containing the weather samples (including the column header)
WEATHER_SAMPLE_FILE = 'weather.csv'
data, labels = load_csv(WEATHER_SAMPLE_FILE, target_column=11, columns_to_ignore=[0])

TrainingSetFeatures = data
TrainingSetLabels = labels

def preprocessor(data):
	copyData = np.zeros((len(data), 12))
	for i in range(len(data)):
		sample = data[i]
		#grab the date element
		dayStr = sample[0]
		dayOfYear = datetime.strptime(dayStr, "%m/%d/%Y").timetuple().tm_yday
		hours = int(sample[1])
		hourVectorReal = math.cos(2*math.pi * (hours/24))
		hourVectorImg = math.sin(2*math.pi * (hours/24))		
		dayVectorReal = math.cos(2*math.pi * (dayOfYear/365))
		dayVectorImg = math.sin(2*math.pi * (dayOfYear/365))
		copyData[i][0] = hourVectorReal + 1
		copyData[i][1] = (hourVectorImg + 1) / 2
		copyData[i][2] = (dayVectorReal  + 1) / 2
		copyData[i][3] = (dayVectorImg + 1) / 2
		copyData[i][4] = float(sample[2])
		copyData[i][5] = float(sample[3]) / 10
		copyData[i][6] = float((sample[4])) + 19 / 55
		copyData[i][7] = float((sample[5])) + 25 / 55
		copyData[i][8] = float((sample[6])) + 15 / 120
		copyData[i][9] = float(sample[7]) / 45
		copyData[i][10] = float(sample[8]) / 40
		copyData[i][11] = float(sample[9]) / 35
	return copyData

def categorizeLabels(labels):
	for i in range(len(labels)):
		evSample = float(labels[i])
		if evSample > 4000:
			labels[i] = 4
		elif evSample > 3000:
			labels[i] = 3
		elif evSample > 2000:
			labels[i] = 2
		elif evSample > 1000:
			labels[i] = 1
		else:
			labels[i] = 0

def scaleLabels(labels):
	for i in range(len(labels)):
		labels[i] = [float(labels[i]) / 4750]

TrainingSetFeatures = preprocessor(TrainingSetFeatures)
#categorizeLabels(TrainingSetLabels)
#TrainingSetLabels = to_categorical(TrainingSetLabels, 5)
scaleLabels(TrainingSetLabels)

# init
tflearn.init_graph(seed=1)

#create a test set from the number of samples and traning set
net = tflearn.input_data(shape=[None, 12])
# net = tflearn.fully_connected(net, 64)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 16)
#net = tflearn.fully_connected(net, 5, activation="softmax")
net = tflearn.fully_connected(net, 1, activation="relu")
net = tflearn.regression(net, loss="mean_square")
#net = tflearn.regression(net, optimizer=tflearn.optimizers.AdaGrad(learning_rate=0.01, initial_accumulator_value=0.01), loss='mean_square', learning_rate=0.05)
# categorized the data into bins for and that should be the number of 0.88888

# Define model
model = tflearn.DNN(net, tensorboard_dir='C:/Users/jeffw/AppData/Local/Temp/tflearn_logs', tensorboard_verbose=2)

# Start training (apply gradient descent algorithm)
model.fit(TrainingSetFeatures, TrainingSetLabels, n_epoch=10, batch_size=12, show_metric=True, validation_set=0.1)

model.save('solar_prediction.model')

import matplotlib.pyplot as plt
numDataPoints = 100
pred = model.predict(TrainingSetFeatures[0:numDataPoints])
time = list(map(lambda x: x[1], data[0:numDataPoints]))
scaledPred = list(map(lambda x: x[0] * 4750, pred))
actualOutput = list(map(lambda x: x[0] * 4750, TrainingSetLabels[0:numDataPoints]))
plt.plot(time, actualOutput, 'gs')
plt.plot(time, scaledPred, 'ro')
plt.title("Predicted vs. Actual Solar Power")
plt.ylabel("Solar power")
plt.xlabel("Time")
plt.legend()
plt.show()

# # Let's create some data for DiCaprio and Winslet
# test = [[],[],[]]
# test[0] =  ["2/1/2016",6,0,9.92,0.37,-0.01,89.12,4.72,29.19,29.98,0]
# test[1] = ["7/18/2017",12,0,10,28.71,17.31,47.32,4.44,29.3,30.09,4568.75]
# test[2] = ["2/4/2016",8,0.3,10,-5.49,-8.8,74.57,11.88,29.46,30.26,1750.25]

# test = preprocessor(test)

# pred = model.predict(test)

# # find index
# lowOutputPrediction = pred[0] * 4750
# highOutputPrediction = pred[1] * 4750
# myOutputPrediction = pred[2] * 4750

# print("Estimate:", lowOutputPrediction, "      Actual:", 0)
# print("Estimate:", highOutputPrediction, "      Actual:", 4568.75)
# print("Estimate:", myOutputPrediction, "      Actual:", 1750.25)
