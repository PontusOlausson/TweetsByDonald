import argparse
import sys
from read_data import DataReader


def main():
    """
    Main method. Decodes command-line arguments, and starts the Named Entity Recognition.
    """

    parser = argparse.ArgumentParser(description='Named Entity Recognition', usage='\n* If the -d and -t are both given, the program will train a model, and apply it to the test file. \n* If only -t and -m are given, the program will read the model from the model file, and apply it to the test file.')

    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-t', type=str,  required=True, help='test file (mandatory)')

    group = required_named.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', type=str, help='training file (required if -m is not set)')
    group.add_argument('-m', type=str, help='model file (required if -d is not set)')

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('-s', action='store_true', default=False, help='Use stochastic gradient descent')
    group2.add_argument('-b', action='store_true', default=False, help='Use batch gradient descent')
    group2.add_argument('-mgd', action='store_true', default=False, help='Use mini-batch gradient descent')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    arguments = parser.parse_args()

    DataReader(arguments.d, arguments.t, arguments.m, arguments.s, arguments.mgd)

    input("Press Return to finish the program...")


if __name__ == '__main__':
    main()