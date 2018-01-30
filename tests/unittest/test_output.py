import numpy as np
import unittest
from neural_network.network import net
from neural_network.config.config import LEARNING_RATE, LAYERS, EPOCHS
from neural_network.learning.training import training

"""
	training_data = [
		([0, 0, 0, 0], 0),
		([0, 0, 0, 1], 1),
		([0, 0, 1, 0], 0),
		([0, 0, 1, 1], 0),
		([0, 1, 0, 0], 1),
		([0, 1, 0, 1], 1),
		([0, 1, 1, 0], 0),
		([0, 1, 1, 1], 1),
		([1, 0, 0, 0], 1),
		([1, 0, 0, 1], 1),
		([1, 0, 1, 0], 0),
		([1, 0, 1, 1], 1),
		([1, 1, 0, 0], 0),
		([1, 1, 0, 1], 0),
		([1, 1, 1, 0], 0),
		([1, 1, 1, 1], 1)
	]
"""


class TestOutput(unittest.TestCase):

	training_data = [
		([0, 0, 0], 0),
		([0, 0, 1], 1),
		([0, 1, 0], 0),
		([0, 1, 1], 0),
		([1, 0, 0], 1),
		([1, 0, 1], 1),
		([1, 1, 0], 0),
		([1, 1, 1], 1)
	]
	neural_net = net.NeuralNetwork(layers=LAYERS, learning_rate=LEARNING_RATE)
	training(neural_net=neural_net, training_data=training_data, epochs=EPOCHS)

	def make_predict(self, input_data):
		return self.neural_net.predict(np.array(input_data))

	def test_bool_result(self):
		for input_data, correct_predict in self.training_data:
			self.assertEqual(self.make_predict(input_data) >= 0.5, np.array(correct_predict) == 1)

	def test_numbers_result(self):
		pass
		self.assertLess(self.make_predict(self.training_data[0][0]), 0.5)
		self.assertGreaterEqual(self.make_predict(self.training_data[1][0]), 0.5)
		self.assertLess(self.make_predict(self.training_data[2][0]), 0.5)
		self.assertLess(self.make_predict(self.training_data[3][0]), 0.5)
		self.assertGreaterEqual(self.make_predict(self.training_data[4][0]), 0.5)
		self.assertGreaterEqual(self.make_predict(self.training_data[5][0]), 0.5)
		self.assertLess(self.make_predict(self.training_data[6][0]), 0.5)
		self.assertGreaterEqual(self.make_predict(self.training_data[7][0]), 0.5)


def run(suite):
	suite.addTest(TestOutput('test_bool_result'))
	suite.addTest(TestOutput('test_numbers_result'))
