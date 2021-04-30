from random import seed
from random import random
from csv import reader
from math import exp
import matplotlib.pyplot as plt
import numpy as np

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden[0])]
	network.append(hidden_layer)
	for j in range(1,len(n_hidden)):
		hidden_layer=[{'weights':[random() for i in range(n_hidden[j-1] + 1)]} for i in range(n_hidden[j])]
		network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden[-1] + 1)]} for k in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	global n_outputs
	inputs = row
	for i in range(len(network)):
		new_inputs = []
		for neuron in network[i]:
			activation = activate(neuron['weights'], inputs)
			if i < len(network) - n_outputs:
				neuron['output'] = transfer(activation)
			else:
				neuron['output'] = activation
				# print(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		# errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += neuron['weights'][j] * neuron['delta']
				# errors.append(error)
				neuron=layer[j]
				neuron['delta'] = error * transfer_derivative(neuron['output'])
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				# errors.append(expected[j] - neuron['output'])
				neuron['delta'] = expected[j] - neuron['output']
		# for j in range(len(layer)):
		# 	neuron = layer[j]
		# 	neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		max_error=0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [row[-i-1] for i in reversed(range(n_outputs))]
			error=abs(expected[0]-outputs[0])
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			if(error>max_error):
				max_error=error
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		variance=sum_error/(len(train))
		print('>epoch=%d, variance=%.5f, max error=%.5f' % (epoch, variance,max_error))
	# return output
		# for layer in network:
		# 	print(layer)
# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Make a prediction with a network
def predict(network, dataset):
	output=[]
	for row in dataset:
		y=forward_propagate(network, row)
		# print(str(row)+" "+str(y[0]))
		output.append(y[0])
	return output

# Test training backprop algorithm

filename = 'input1.csv'
dataset = load_csv(filename)
# convert string numbers to floats
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# print(dataset)
t=[]
x=[]
y=[]
dataty=[]
data=[]
for row in dataset:
	t.append(row[0])
	x.append(row[1])
	y.append(row[2])
	dataty.append([row[0],row[2]])
	data.append(row[0])
for row in dataset:
	data.append(row[2])
# print(np.array(dataxy).T)
# print(dataset)
# dataset=dataty
seed(1)
n_hidden=[3]
n_outputs=1
interation=1000
n_inputs = len(dataset[0]) - n_outputs
network = initialize_network(n_inputs, n_hidden, n_outputs)
train_network(network, dataset, 1,interation, n_outputs)
for layer in network:
	print(layer)
output=predict(network,dataset)
# draw graphic
plt.scatter(t,x,label='x(t)')
plt.scatter(t,y,label='y(t) expected')
plt.scatter(t,output,label='y(t) network')
plt.legend()
plt.grid()
plt.show()
