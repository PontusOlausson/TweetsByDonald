import argparse
import sys
from read_data import DataReader
from nb_classifier import TrainNB
import numpy as np



def main():
    data_reader = DataReader("Data/training_data.csv")
    tweets, labels = data_reader.tweets, data_reader.labels
    nb_class = TrainNB()
    nb_class.train(tweets, labels)

    prediction = nb_class.predict(tweets[140])
    print(prediction)

    nb_class.write_to_file('params_big.p')
    nb_class.read_from_file('params_big.p')

    prediction = nb_class.predict(tweets[140])
    print(prediction)


if __name__ == '__main__':
    main()
