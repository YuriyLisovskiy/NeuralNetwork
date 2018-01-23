import sys
import numpy as np
from config.config import EPOCHS


def MSE(y, Y):
	return np.mean((y - Y)**2)


def training(neural_net, training_data):
	print("\nTraining:")
	for e in range(EPOCHS):
		inputs = []
		correct_predictions = []
		for input_stat, correct_predict in training_data:
			neural_net.train(np.array(input_stat), correct_predict)
			inputs.append(np.array(input_stat))
			correct_predictions.append(np.array(correct_predict))

		train_loss = MSE(neural_net.predict(np.array(inputs).T), np.array(correct_predictions))
		sys.stdout.write("\r Progress: {}%, Training loss: {}".format(str(100 * e / float(EPOCHS))[:4], str(train_loss)[:5]))
	print("\n")
