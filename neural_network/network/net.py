import sys
import numpy as np


class NeuralNetwork(object):

	def __init__(self, input_layer, hidden_layers, output_layer, learning_rate=0.1, log=True):
		if log:
			print('Setting up network...')
		if len(input_layer) > 1:
			raise ValueError('Invalid input layer: input layer must be a list with one element')
		if len(output_layer) > 1:
			raise ValueError('Invalid output layer: output layer must be a list with one element')
		if output_layer[0] != 1:
			raise ValueError('Invalid output layer: output layer must have only one neuron')
		layers = input_layer + hidden_layers + output_layer
		
		for i in range(len(layers) - 1):
			correct_number = 2 ** layers[i]
			if layers[i + 1] > correct_number:
				raise ValueError('An amount of neurons in hidden layer #{} is {}, {} neuron(s) is redundant'.format(
					i + 2, layers[i + 1], layers[i + 1] - correct_number))

		self.weights = []
		for i in range(len(layers) - 1):
			self.weights.append(np.random.normal(0.0, 2 ** -0.5, (layers[i + 1], layers[i])))
		self.sigmoid_mapper = np.vectorize(self.sigmoid)
		self.learning_rate = np.array([learning_rate])
		
	@staticmethod
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def derivative_of_sigmoid(x):
		return x * (1 - x)

	def predict(self, inputs):
		res = inputs
		for weight in self.weights:
			res = np.dot(weight, res)
			res = self.sigmoid_mapper(res)
		return res
	
	@staticmethod
	def mse(y_1, y_2):
		return np.mean((y_1 - y_2) ** 2)

	def back_prop_train(self, inputs, expected_predict):
		outputs = []
		res = inputs
		outputs.append(inputs)

		for weight in self.weights:
			res = np.dot(weight, res)
			res = self.sigmoid_mapper(res)
			outputs.append(res)

		actual_predict = res
		weights_index = len(self.weights) - 1
		outputs_index = len(outputs) - 2
		error_layer = np.array([actual_predict - expected_predict])
		weights_delta = error_layer * self.derivative_of_sigmoid(actual_predict)
		self.weights[weights_index] -= (np.dot(weights_delta.reshape(weights_delta.shape[1], 1),
												outputs[outputs_index].reshape(1, len(
													outputs[outputs_index])))) * self.learning_rate

		for i, j in zip(range(weights_index, 0, -1), range(outputs_index, 0, -1)):
			error_layer = np.dot(weights_delta, self.weights[i])
			weights_delta = error_layer * self.derivative_of_sigmoid(outputs[j])
			self.weights[i - 1] -= np.dot(outputs[j - 1].reshape(len(outputs[j - 1]), 1),
											weights_delta).T * self.learning_rate
			j -= 1

	def train(self, data, iterations=1000, log=True):
		if log:
			print('Training...')
		train_loss = None
		for e in range(iterations):
			inputs = []
			correct_predictions = []
			for input_stat, correct_predict in data:
				self.back_prop_train(np.array(input_stat), np.array(correct_predict))
				inputs.append(np.array(input_stat))
				correct_predictions.append(np.array(correct_predict))
			if log:
				train_loss = self.mse(self.predict(np.array(inputs).T), np.array(correct_predictions))
				sys.stdout.write(
					"\r Progress: {}%, Training loss: {}".format(str(100 * e / float(iterations))[:4], str(train_loss)[:5]))
		if log:
			sys.stdout.write(
				"\r Progress: 100%, Training loss: {}\nDone.\n\n".format(str(train_loss)[:5]))
