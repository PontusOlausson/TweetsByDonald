import numpy as np
import codecs
import csv


class DataReader:

    def __init__(self, training_file):

        self.training_set = []

        if training_file:
            self.read_and_process_data(training_file)

    def read_and_process_data(self, training_file):
        """
        :param training_file: file to be read and processed.
        :return: void
        """
        with codecs.open(training_file, 'r', 'utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                clean_tweet = self.process_data(row[5])
                self.training_set.append((row[0], clean_tweet))




    def process_data(self, tweet):
        """

        :param tweet: one tweet read from the training document, only contains the body of the tweet.
        :return: clean_tweet, tweet body with dimensionality reduced.
        """
        clean_tweet = ""
        return clean_tweet


def main():
    DataReader("Data/training_data_small.txt")


if __name__ == "__main__":
    main()

