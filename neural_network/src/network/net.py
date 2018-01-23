import numpy as np


class NeuralNetwork(object):

	weights = []

	def __init__(self, layers, learning_rate=0.1):
		for i in range(len(layers) - 1):
			self.weights.append(np.random.normal(0.0, 2 ** -0.5, (layers[i + 1], layers[i])))
		self.sigmoid_mapper = np.vectorize(self.sigmoid)
		self.learning_rate = [learning_rate]

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

	def back_prop_train(self, inputs, expected_predict):
		outputs = []
		res = inputs
		outputs.append(inputs)

		for weight in self.weights:
			res = np.dot(weight, res)
			res = self.sigmoid_mapper(res)
			outputs.append(res)

		actual_predict = res[0]
		weights_index = len(self.weights) - 1
		outputs_index = len(outputs) - 2
		error_layer = np.array([actual_predict - expected_predict])
		weights_delta = error_layer * self.derivative_of_sigmoid(actual_predict)
		self.weights[weights_index] -= (np.dot(weights_delta, outputs[outputs_index].reshape(1, len(outputs[outputs_index])))) * self.learning_rate

		for i, j in zip(range(weights_index, 0, -1), range(outputs_index, 0, -1)):
			error_layer = weights_delta * self.weights[i]
			weights_delta = error_layer * self.derivative_of_sigmoid(outputs[j])
			self.weights[i - 1] -= np.dot(outputs[j - 1].reshape(len(outputs[j - 1]), 1), weights_delta).T * self.learning_rate
			j -= 1
