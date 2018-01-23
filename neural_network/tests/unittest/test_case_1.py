import numpy as np


def test_with_bool(neural_net, training_data):
	for input_stat, correct_predict in training_data:
		print(" For input: {} the prediction is: {}, expected: {}".format(str(input_stat),
			str(neural_net.predict(np.array(input_stat)) > 0.5), str(correct_predict == 1)))


def test_with_numbers(neural_net, training_data):
	for input_stat, correct_predict in training_data:
		print(" For input: {} the prediction is: {}, expected: {}".format(str(input_stat),
			str(neural_net.predict(np.array(input_stat))), str(correct_predict == 1)))