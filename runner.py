import sys
from tests import all_tests


def main():
	if 'test' in sys.argv:
		all_tests.run()


if __name__ == '__main__':
	main()
