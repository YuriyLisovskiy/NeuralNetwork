import unittest
from neural_network.network import net
from neural_network.config.config import INPUT_LAYER, HIDDEN_LAYERS, OUTPUT_LAYER


class TestExceptions(unittest.TestCase):

	def test_last_layer_exception(self):
		with self.assertRaises(ValueError):
			params = {
				'input_layer': INPUT_LAYER,
				'hidden_layers': HIDDEN_LAYERS,
				'output_layer': [2]
			}
			net.NeuralNetwork(**params)
		
	def test_redundant_layers_exception(self):
		with self.assertRaises(ValueError):
			params = {
				'input_layer': INPUT_LAYER,
				'hidden_layers': [9],
				'output_layer': OUTPUT_LAYER
			}
			net.NeuralNetwork(**params)
			
	def test_input_layer_exception(self):
		with self.assertRaises(ValueError):
			params = {
				'input_layer': INPUT_LAYER,
				'hidden_layers': HIDDEN_LAYERS,
				'output_layer': [1, 2]
			}
			net.NeuralNetwork(**params)
	
	def test_output_layer_exception(self):
		with self.assertRaises(ValueError):
			params = {
				'input_layer': [1, 1],
				'hidden_layers': HIDDEN_LAYERS,
				'output_layer': [1]
			}
			net.NeuralNetwork(**params)


def run(suite):
	suite.addTest(TestExceptions('test_last_layer_exception'))
	suite.addTest(TestExceptions('test_redundant_layers_exception'))
	suite.addTest(TestExceptions('test_input_layer_exception'))
	suite.addTest(TestExceptions('test_output_layer_exception'))
