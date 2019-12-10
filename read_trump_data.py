import codecs
import csv
import time
import re
from nltk.corpus import stopwords


class TrumpDataReader:

    def __init__(self, training_file, case="-t"):
        self.tweets = []
        self.case = case

        if training_file:
            self.read_and_process_data(training_file, self.case)

    def read_and_process_data(self, training_file, case):
        """
        :param training_file: file to be read and processed.
        :return: void
        """
        with codecs.open(training_file, 'r', 'utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            i = 0
            k = 1000
            start_time = time.time()
            for row in reader:
                tweet = row[0].split("§")[0]
                # print(tweet)
                clean_tweet = self.process_tweet(tweet, case)
                if clean_tweet != "":
                    self.tweets.append(clean_tweet)
                    #print(clean_tweet)

                if i % k == 0:
                    end_time = time.time()
                    duration = end_time - start_time
                    print(duration)
                i += 1
            print('Done!')

    def process_tweet(self, tweet, case):
        """
        :param tweet: one tweet read from the training document, only contains the body of the tweet.
        :return: tweet, tweet body with dimensionality reduced.
        """
        split_tweet = tweet.split()
        if split_tweet[0] == "RT":
            return ""

        for index in range(len(split_tweet)):
            split_tweet[index] = self.process_token(split_tweet[index], case)

        while "" in split_tweet:
            split_tweet.remove("")

        tweet = self.list_to_string(split_tweet)

        return tweet

    def process_token(self, token, case):
        """
        Method to process a single token in a tweet
        :param token: one word from tweet
        :return: processed token
        """
        if token[0] == "@":
            token = "@user"

        elif token[:4] == "http" or token[:4] == "www." or token[len(token)-4:] == ".com":
            token = "adress.com"

        elif 1 == 0:
            pass

        else:
            if case == "-t":
                token = re.sub(r'\s[^\s\w]+\s', ' ', token)
                token = re.sub(r'\d+\s?|\n|[^\s\w]', '', token).lower().strip()
            elif case == "-lm":
                pass

        return token

    def list_to_string(self, split_tweet):
        """
        :param split_tweet: tweet split into list
        :return: split_tweet joined to a string
        """
        return " ".join(split_tweet)


def main():
    TrumpDataReader("Data/tweet_data.txt")


if __name__ == "__main__":
    main()

