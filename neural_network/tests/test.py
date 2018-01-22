import numpy as np
from neural_network.src import network
from neural_network.src.config.config import LEARNING_RATE

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

net = network.NeuralNetwork(layers=[3, 2, 1], learning_rate=LEARNING_RATE)


def test_with_bool(neural_net, training_data):
	for input_stat, correct_predict in training_data:
		print("For input: {} the prediction is: {}, expected: {}".format(str(input_stat),
			str(neural_net.predict(np.array(input_stat)) > 0.5), str(correct_predict == 1)))


def test_with_numbers(neural_net, training_data):
	for input_stat, correct_predict in training_data:
		print("For input: {} the prediction is: {}, expected: {}".format(str(input_stat),
			str(neural_net.predict(np.array(input_stat))), str(correct_predict == 1)))
