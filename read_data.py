import numpy as np
import codecs
import csv
import re
from nltk.corpus import stopwords


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
                clean_tweet = self.process_tweet(row[5])
                self.training_set.append((row[0], clean_tweet))
        print(self.training_set)

    def process_tweet(self, tweet):
        """
        :param tweet: one tweet read from the training document, only contains the body of the tweet.
        :return: tweet, tweet body with dimensionality reduced.
        """
        split_tweet = tweet.split()
        for index in range(len(split_tweet)):
            split_tweet[index] = self.process_token(split_tweet[index])

        while "" in split_tweet:
            split_tweet.remove("")

        tweet = self.list_to_string(split_tweet)

        return tweet

    def process_token(self, token):
        """
        Method to process a single token in a tweet
        :param token: one word from tweet
        :return: processed token
        """
        if token[0] == "@":
            token = "@user"

        elif token[:5] == "http:" or token[:4] == "www." or token[len(token)-4:] == ".com":
            token = "adress.com"

        else:
            token = re.sub(r'\s[^\s\w]+\s', ' ', token)
            token = re.sub(r'\d+\s?|\n|[^\s\w]', '', token).strip().lower()

            stop_words = set(stopwords.words("english"))
            if token in stop_words:
                token = ""
        return token

    def list_to_string(self, split_tweet):
        """
        :param split_tweet: tweet split into list
        :return: split tweet joined to a string
        """
        return " ".join(split_tweet)


def main():
    DataReader("Data/training_data_small.txt")


if __name__ == "__main__":
    main()

