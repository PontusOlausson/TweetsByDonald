#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import nltk
from collections import defaultdict
import codecs
from read_trump_data import TrumpDataReader

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """

    def process_files(self, docs):
        """
        Processes the file @code{f}.
        """
        iterations = 0
        for doc in docs:
            self.last_index = -1
            doc = "TWEET_START_SIGN " + doc + " TWEET_END_SIGN"
            if iterations % 1000 == 0:
                print(iterations)
                print(doc)
            iterations += 1
            try:
                self.tokens = doc.split()
                i = 1
                while i < len(self.tokens)-2:
                    # for i in range(len(self.tokens)):
                    if self.tokens[i][0] == "@":
                        self.tokens[i] = "@user"
                    elif self.tokens[i][:4] == "http" or self.tokens[i][:4] == "www." or self.tokens[i][len(self.tokens[i]) - 4:] == ".com":
                        self.tokens[i] = "adress.com"
                    else:
                        tokenized = nltk.word_tokenize(self.tokens[i])
                        if len(tokenized) > 1:
                            self.tokens.remove(self.tokens[i])
                            for j in range(len(tokenized)):
                                self.tokens.insert(i+j, tokenized[j])
                    i += 1

            except LookupError:
                nltk.download('punkt')
                self.tokens = nltk.word_tokenize(doc)

            for token in self.tokens:
                self.process_token(token)
        self.unique_words = len(self.unigram_count)

    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """

        if self.last_index != -1:
            self.bigram_count[self.word[self.last_index]][token] += 1

        # If this is the first occurrence of the current word being processed
        if token not in self.unigram_count:
            self.word[self.unique_words] = token
            self.index[token] = self.unique_words
            self.unique_words += 1

        self.last_index = self.index[token]
        self.unigram_count[token] += 1
        self.total_words += 1

    def stats(self):
        """
        Creates a list of rows to print of the language model.

        """
        rows_to_print = []

        rows_to_print.append("%i %i" % (self.unique_words, self.total_words))

        for t in self.unigram_count:
            rows_to_print.append("%i %s %i" % (self.index[t], t, self.unigram_count[t]))

        for t1 in self.bigram_count:
            for t2 in self.bigram_count[t1]:
                p = math.log(self.bigram_count[t1][t2] / self.unigram_count[t1])
                rows_to_print.append("%i %i %.15f" % (self.index[t1], self.index[t2], p))

        rows_to_print.append("-1")

        return rows_to_print

    def write_to_file(self, rows_to_write, destination):
        if destination is not None:
            with codecs.open(destination, 'w', 'utf-8') as f:
                print("Writing to file" + destination)
                for row in rows_to_write:
                    f.write(row + '\n')

    def __init__(self):
        """
        <p>Constructor. Processes the file <code>f</code> and builds a language model
        from it.</p>

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        self.laplace_smoothing = False


def main():
    """
    Parse command line arguments
    """
    destination = "trump_model_2.txt"
    #destination = None
    bigram_trainer = BigramTrainer()

    trump_data_reader = TrumpDataReader("Data/tweet_data_small.txt")
    tweets = trump_data_reader.tweets_generating

    bigram_trainer.process_files(tweets)
    # stats = bigram_trainer.stats()

    #if destination is not None:
    #    with codecs.open(destination, 'w', 'utf-8') as f:
    #        for row in stats:
    #            f.write(row + '\n')


#    else:
#        for row in stats: print(row)


if __name__ == "__main__":
    main()
