from builtins import len
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

	def predict(self, inputs):
		res = inputs
		for weight in self.weights:
			res = np.dot(weight, res)
			res = self.sigmoid_mapper(res)
		return res

	def train(self, inputs, expected_predict):
		outputs = []
		res = inputs
		outputs.append(inputs)
		for weight in self.weights:
			res = np.dot(weight, res)
			res = self.sigmoid_mapper(res)
			outputs.append(res)
		actual_predict = res[0]
		index = len(self.weights) - 1
		j = len(outputs) - 2
		error_layer = np.array([actual_predict - expected_predict])
		gradient_layer = actual_predict * (1 - actual_predict)
		weights_delta = error_layer * gradient_layer
		self.weights[index] -= (np.dot(weights_delta, outputs[j].reshape(1, len(outputs[j])))) * self.learning_rate
		for i in range(index, 0, -1):
			error_layer = weights_delta * self.weights[i]
			gradient_layer = outputs[j] * (1 - outputs[j])
			weights_delta = error_layer * gradient_layer
			self.weights[i - 1] -= np.dot(outputs[j - 1].reshape(len(outputs[j - 1]), 1), weights_delta).T * self.learning_rate
			j -= 1
