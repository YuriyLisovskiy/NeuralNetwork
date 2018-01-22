from neural_network.tests.test import train, net, test_with_bool, test_with_numbers
from neural_network.src.training import training


def main():
	training(net, train)

	test_with_bool(net, train)
	test_with_numbers(net, train)

if __name__ == '__main__':
	main()
