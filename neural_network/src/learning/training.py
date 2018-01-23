import sys
import numpy as np
from neural_network.config.config import EPOCHS


def mse(y_1, y_2):
	return np.mean((y_1 - y_2)**2)


def training(neural_net, training_data):
	train_loss = None
	print("\nTraining:")
	for e in range(EPOCHS):
		inputs = []
		correct_predictions = []
		for input_stat, correct_predict in training_data:
			neural_net.back_prop_train(np.array(input_stat), correct_predict)
			inputs.append(np.array(input_stat))
			correct_predictions.append(np.array(correct_predict))

		train_loss = mse(neural_net.predict(np.array(inputs).T), np.array(correct_predictions))
		sys.stdout.write(
			"\r Progress: {}%, Training loss: {}".format(str(100 * e / float(EPOCHS))[:4], str(train_loss)[:5]))
	sys.stdout.write(
		"\r Progress: 100%, Training loss: {}\nDone.\n\n".format(str(train_loss)[:5]))
