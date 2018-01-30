import unittest

from .unittest import test_output, test_exceptions


def run():
	suite = unittest.TestSuite()
	test_output.run(suite)
	test_exceptions.run(suite)
	unittest.TextTestRunner().run(suite)
