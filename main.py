import argparse
import sys
from read_data import DataReader
from nb_classifier import NBClassifier
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--train', '-t', type=str, help='file from which to train sentiment analysis')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the sentiment analysis')
    parser.add_argument('--load', '-l', type=str, help='file from which to load sentiment analysis')

    arguments = parser.parse_args()

    nb_class = NBClassifier()

    if arguments.train:
        data_reader = DataReader(arguments.train)
        tweets, labels = data_reader.tweets, data_reader.labels
        nb_class.train(tweets, labels)
        prediction = nb_class.predict(tweets[0])
        print(prediction)
    if arguments.destination:
        nb_class.write_to_file(arguments.destination)
    if arguments.load:
        nb_class.read_from_file(arguments.load)

        data_reader = DataReader('Data/training_data_small.csv')
        prediction = nb_class.predict(data_reader.tweets[0])
        print(prediction)


if __name__ == '__main__':
    main()
