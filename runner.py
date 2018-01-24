import sys
from neural_network.learning.training import training
from tests.test import train, network
from tests import test


def main():
	if 'test' in sys.argv:
		training(network, train)
		test.run(neural_net=network, training_data=train)


if __name__ == '__main__':
	main()
