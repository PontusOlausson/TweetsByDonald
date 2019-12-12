import codecs
import csv
import string
import time

import nltk
from nltk.corpus import stopwords


class DataReader:

    def __init__(self, training_file):

        self.tweets = []
        self.labels = []

        self.unprocessed_tweet = []

        self.negating = False

        if training_file:
            self.read_and_process_data(training_file)

    def read_and_process_data(self, training_file):
        """
        :param training_file: file to be read and processed.
        :return: void
        """
        with codecs.open(training_file, 'r', 'latin-1') as f:
            reader = csv.reader(f, )
            i = 1
            k = 10000
            start_time = time.time()
            for row in reader:
                self.negating = False

                clean_tweet = self.process_tweet(row[5])
                self.tweets.append(clean_tweet)
                self.labels.append(int(row[0]))
                self.unprocessed_tweet.append(row[5])

                if i % k == 0:
                    end_time = time.time()
                    duration = end_time - start_time
                    start_time = time.time()
                    time_left = duration / k * (1600000 - i)
                    print(time_left)
                i += 1

            print('Done!')

    def process_tweet(self, tweet):
        """
        :param tweet: one tweet read from the training document, only contains the body of the tweet.
        :return: tweet, tweet body with dimensionality reduced.
        """
        tokens = nltk.word_tokenize(tweet)
        
        for index in range(len(tokens)):
            tokens[index] = self.process_token(tokens[index])

        while "" in tokens:
            tokens.remove("")

        tokens = set(tokens)

        tweet = self.list_to_string(tokens)

        return tweet

    def process_token(self, token):
        """
        Method to process a single token in a tweet
        :param token: one word from tweet
        :return: processed token
        """
        if token[0] == "@":
            token = "@user"

        elif token[:4] == "http" or token[:4] == "www." or token[len(token)-4:] == ".com":
            token = "adress.com"

        else:
            # token = re.sub(r'\s[^\s\w]+\s', ' ', token)
            # token = re.sub(r'\d+\s?|\n|[^\s\w]', '', token)

            token = token.lower().strip()

            stop_words = set(stopwords.words("english"))

            if token in string.punctuation:
                self.negating = False
                return ""

            if token in stop_words:
                return ""

            if token == "n't":
                self.negating = True
                return ""

            if self.negating:
                token = "NOT_" + token

        return token

    def list_to_string(self, split_tweet):
        """
        :param split_tweet: tweet split into list
        :return: split_tweet joined to a string
        """
        return " ".join(split_tweet)


def main():
    DataReader("Data/training_data_small.csv")


if __name__ == "__main__":
    main()

