import math
import argparse
import codecs
from collections import defaultdict
import random
import numpy as np

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability 😊 the estimation of the entropy) of the test corpus.
        # Important that it is named self.logProb for the --check flag to work
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))

                for i in range(self.unique_words):
                    id, token, n = f.readline().strip().split(' ')
                    id = int(id)
                    n = int(n)
                    self.word[id] = token
                    self.index[token] = id
                    self.unigram_count[token] = n

                line = f.readline().strip()
                while line != "-1":
                    id1, id2, log_p = line.split(' ')
                    id1 = int(id1)
                    id2 = int(id2)
                    log_p = float(log_p)
                    self.bigram_prob[self.word[id1]][self.word[id2]] = log_p

                    line = f.readline().strip()

                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def generate(self, w):
        """
        Generates and prints n words, starting with the word w, and following the distribution
        of the language model.
        """
        output = ''

        #for i in range(n):
        while True:
            # Can use self.unigram_count.keys() instead
            possible_words = list(self.bigram_prob[w].keys())
            probabilities = []

            if len(possible_words) == 0:
                r = random.randrange(len(self.word))
                next_w = self.word[r]
            else:
                for t in possible_words:
                    if t in self.bigram_prob[w]:
                        Pw2w1 = math.pow(math.e, self.bigram_prob[w][t])
                    else:
                        Pw2w1 = 0
                    Pw2 = self.unigram_count[t] / self.total_words
                    P = Pw2w1 * self.lambda1 + Pw2 * self.lambda2 + self.lambda3
                    probabilities.append(P)
                next_w = np.random.choice(possible_words, 1, probabilities)[0]

            if next_w == "TWEET_END_SIGN":
                break

            output += next_w + ' '
            w = next_w

        print(output)


def main():
    """
    Parse command line arguments
    """
    #parser = argparse.ArgumentParser(description='BigramTester')
    #parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    #parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    #parser.add_argument('--number_of_words', '-n', type=int, default=100)

    #arguments = parser.parse_args()

    generator = Generator()
    generator.read_model("trump_model_2.txt")
    generator.generate("TWEET_START_SIGN")


if __name__ == "__main__":
    main()