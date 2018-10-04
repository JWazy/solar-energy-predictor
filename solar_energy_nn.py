# Based off code found here http://tflearn.org/tutorials/quickstart.html and here https://github.com/AshitakaLax/WeatherNN/blob/master/weathercategoried.py
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
		copyData[i][0] = hourVectorReal 
		copyData[i][1] = hourVectorImg 
		copyData[i][2] = dayVectorReal 
		copyData[i][3] = dayVectorImg
		copyData[i][4] = sample[2]
		copyData[i][5] = sample[3]
		copyData[i][6] = sample[4]
		copyData[i][7] = sample[5]
		copyData[i][8] = sample[6]
		copyData[i][9] = sample[7]
		copyData[i][10] = sample[8]
		copyData[i][11] = sample[9]
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
		labels[i] = [float(labels[i]) / 10000]

TrainingSetFeatures = preprocessor(TrainingSetFeatures)
#categorizeLabels(TrainingSetLabels)
#TrainingSetLabels = to_categorical(TrainingSetLabels, 5)
scaleLabels(TrainingSetLabels)

#create a test set from the number of samples and traning set
net = tflearn.input_data(shape=[None, 12])
net = tflearn.fully_connected(net, 64)
net = tflearn.fully_connected(net, 64)
#net = tflearn.fully_connected(net, 5, activation="softmax")
net = tflearn.fully_connected(net, 1, activation="relu")
net = tflearn.regression(net, loss="mean_square")
#net = tflearn.regression(net, optimizer=tflearn.optimizers.AdaGrad(learning_rate=0.01, initial_accumulator_value=0.01), loss='mean_square', learning_rate=0.05)
# categorized the data into bins for and that should be the number of 0.88888

# Define model
model = tflearn.DNN(net)

# Start training (apply gradient descent algorithm)
model.fit(TrainingSetFeatures, TrainingSetLabels, n_epoch=2, batch_size=12, show_metric=True, validation_set=0.1)

# Let's create some data for DiCaprio and Winslet
lowOutput =  [0, 0, 0, 0, 0, 9.92, 0.37, -0.01, 89.12, 4.72, 29.19, 29.98]
highOutput = [0, 0, 0, 0, 0, 10, 6.16, 1.26, 68.96, 0, 29.26, 30.05]
myOutput = [-0.8660254,-0.5,0.85207752,0.52341561,0.,10.,8.47,0.63,57.12,3.72,29.24,30.03]

pred = model.predict([lowOutput, highOutput, myOutput])

# find index
lowOutputPrediction = pred[0].argmax()
highOutputPrediction = pred[1].argmax()
myOutputPrediction = pred[2].argmax()

print("lowoutput estimate:", lowOutputPrediction)
print("highOutput estimate:", highOutputPrediction)
print("myOutput estimate:", myOutputPrediction)
