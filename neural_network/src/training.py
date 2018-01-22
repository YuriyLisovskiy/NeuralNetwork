from neural_network.src.config.config import EPOCHS
import numpy as np
import sys


def MSE(y, Y):
	return np.mean((y - Y)**2)


def training(neural_net, training_data):
	for e in range(EPOCHS):
		inputs = []
		correct_predictions = []
		for input_stat, correct_predict in training_data:
			neural_net.train(np.array(input_stat), correct_predict)
			inputs.append(np.array(input_stat))
			correct_predictions.append(np.array(correct_predict))

		train_loss = MSE(neural_net.predict(np.array(inputs).T), np.array(correct_predictions))
		sys.stdout.write("\rProgress: {}%, Training loss: {}".format(str(100 * e / float(EPOCHS))[:4], str(train_loss)[:5]))
	print("")
