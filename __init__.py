from src.learning.training import training
from tests.test import train, network, test_with_bool, test_with_numbers


def main():
	training(network, train)

	print("Testing:")
	test_with_bool(network, train)
	test_with_numbers(network, train)

if __name__ == '__main__':
	main()
