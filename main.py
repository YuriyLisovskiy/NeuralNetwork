import sys
from neural_network.src.learning.training import training
from neural_network.tests.test import train, network
from neural_network.tests import test


def main():
	if 'train' in sys.argv:
		training(network, train)
	elif 'test' in sys.argv:
		print("\nWARNING!\nNetwork is not trained...\n")
		test.run(neural_net=network, training_data=train)
	else:
		training(network, train)
		test.run(neural_net=network, training_data=train)


if __name__ == '__main__':
	main()
