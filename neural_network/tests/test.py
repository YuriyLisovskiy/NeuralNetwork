from neural_network.src.config import config
from neural_network.src import network
import numpy as np
import sys


def MSE(y, Y):
	return np.mean((y - Y)**2)

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

net = network.NeuralNetwork(learning_rate=config.LEARNING_RATE)


def training():
	for e in range(config.EPOCHS):
		inputs = []
		correct_predictions = []
		for input_stat, correct_predict in train:
			net.train(np.array(input_stat), correct_predict)
			inputs.append(np.array(input_stat))
			correct_predictions.append(np.array(correct_predict))

		train_loss = MSE(net.predict(np.array(inputs).T), np.array(correct_predictions))
		sys.stdout.write("\rProgress: {}%, Training loss: {}".format(str(100 * e / float(config.EPOCHS))[:4], str(train_loss)[:5]))
	print("")