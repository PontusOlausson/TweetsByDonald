import argparse
import codecs
import operator
import re

from nltk.corpus import stopwords

from nb_classifier import NBClassifier


class Demo(object):
    def __init__(self, filepath):
        self.tweets = []

        with codecs.open(filepath, 'r', 'latin-1') as f:
            for tweet in f:
                self.negating = False
                clean_tweet = self.process_tweet(tweet)
                self.tweets.append(clean_tweet)

    def process_tweet(self, tweet):
        tokens = tweet.strip().split()
        for index in range(len(tokens)):
            tokens[index] = self.process_token(tokens[index])

        while "" in tokens:
            tokens.remove("")

        tweet = ' '.join(tokens)

        return tweet

    def process_token(self, token):
        if token[0] == "@":
            token = "@user"

        elif token[:4] == "http" or token[:4] == "www." or token[len(token) - 4:] == ".com":
            token = "adress.com"

        elif token == "&amp;":
            token = ""

        else:
            token = token.lower().strip()

            stop_words = set(stopwords.words("english"))  # Går det snabbare om man har self.stopwords?

            if token[len(token) - 3:] in ["n't"] or token in ["not", "no", "never"]:
                self.negating = True

            if token in stop_words:
                return ""

            if self.negating:
                token = "NOT_" + token

            if any(char in token for char in ".,:;()!?"):
                self.negating = False

            if token in stop_words:
                return ""

            token = re.sub(r'\s[^\s\w]+\s', ' ', token)
            token = re.sub(r'\d+\s?|\n|[^\s\w]', '', token)
        return token


def main():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--load', '-l', required=True, type=str, help='file from which to load parameters')
    parser.add_argument('--file', '-f', required=True, type=str, help='file from which to classify tweets')

    arguments = parser.parse_args()

    demo = Demo(arguments.file)
    nb_class = NBClassifier()

    nb_class.read_from_file(arguments.load)

    for tweet in demo.tweets:
        print(tweet)
        stats = nb_class.predict(tweet)
        prediction = max(stats.items(), key=operator.itemgetter(1))[0]
        prediction = 1 if prediction == 4 else 0
        print(prediction)


if __name__ == '__main__':
    main()
