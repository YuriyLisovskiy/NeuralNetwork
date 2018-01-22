import numpy as np
from neural_network.tests.test import training, train, net


def main():
	training()

	print("")

	for input_stat, correct_predict in train:
		print("For input: {} the prediction is: {}, expected: {}".format(str(input_stat), str(net.predict(np.array(input_stat)) > 0.5), str(correct_predict == 1)))

	print("")

	for input_stat, correct_predict in train:
		print("For input: {} the prediction is: {}, expected: {}".format(str(input_stat), str(net.predict(np.array(input_stat))), str(correct_predict == 1)))

if __name__ == '__main__':
	main()
