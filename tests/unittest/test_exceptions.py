import unittest
from neural_network.network import net


class TestExceptions(unittest.TestCase):

	def test_last_layer_exception(self):
		with self.assertRaises(ValueError):
			net.NeuralNetwork(layers=[3, 8, 2])
		
	def test_redundant_layers_exception(self):
		with self.assertRaises(ValueError):
			net.NeuralNetwork(layers=[3, 9, 1])


def run(suite):
	suite.addTest(TestExceptions('test_last_layer_exception'))
	suite.addTest(TestExceptions('test_redundant_layers_exception'))
