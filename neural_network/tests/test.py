from neural_network.src.network import net
from neural_network.config.config import LEARNING_RATE, LAYERS
from .unittest import test_case_1

train = [
	([0, 0, 0], 0),
	([0, 0, 1], 1),
	([0, 1, 0], 0),
	([0, 1, 1], 0),
	([1, 0, 0], 1),
	([1, 0, 1], 1),
	([1, 1, 0], 0),
	([1, 1, 1], 1)
]

'''
train = [
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
'''

network = net.NeuralNetwork(layers=LAYERS, learning_rate=LEARNING_RATE)


def run(neural_net, training_data):
	print("Testing:")
	test_case_1.test_with_bool(neural_net=neural_net, training_data=training_data)
	print("\n")
	test_case_1.test_with_numbers(neural_net=neural_net, training_data=training_data)
	print("\n")
