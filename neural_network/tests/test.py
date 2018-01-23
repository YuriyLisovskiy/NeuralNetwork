import numpy as np
from neural_network.src.network import net
from neural_network.config.config import LEARNING_RATE, LAYERS


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

network = net.NeuralNetwork(layers=LAYERS, learning_rate=LEARNING_RATE)


def test_with_bool(neural_net, training_data):
	for input_stat, correct_predict in training_data:
		print(" For input: {} the prediction is: {}, expected: {}".format(str(input_stat),
			str(neural_net.predict(np.array(input_stat)) > 0.5), str(correct_predict == 1)))
	print("")


def test_with_numbers(neural_net, training_data):
	for input_stat, correct_predict in training_data:
		print(" For input: {} the prediction is: {}, expected: {}".format(str(input_stat),
			str(neural_net.predict(np.array(input_stat))), str(correct_predict == 1)))
	print("")
