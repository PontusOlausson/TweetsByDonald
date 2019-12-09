import codecs
import nltk
from collections import defaultdict
import numpy as np


"""
This file is part of the project TweetsByDonald for the course DD1418/DD2418 Language engineering at KTH.
Created 2019 by Christoffer Linné and Pontus Olausson.
"""


class TrainNB(object):
    def __init__(self):
        self.Ndoc = 0
        self.Nc = defaultdict(int)
        self.bigdoc = defaultdict(list)
        self.logprior = defaultdict(int)
        self.word_count = {}
        self.loglikelihoods = defaultdict(lambda: defaultdict(int))
        self.V = set()

    def compute_vocabulary(self, documents):
        vocabulary = set()

        for doc in documents:
            for word in doc.split(" "):
                self.V.add(word.lower())

        return vocabulary

    def count_word_in_classes(self):
        counts = {}
        for c in list(self.bigdoc.keys()):
            docs = self.bigdoc[c]
            counts[c] = defaultdict(int)
            for doc in docs:
                for word in doc.split(" "):
                    counts[c][word] += 1

        return counts

    def train(self, documents, labels):
        N_docs = len(documents)
        self.V = self.compute_vocabulary(documents)

        for x, y in zip(documents, labels):
            self.bigdoc[y].append(x)

        all_classes = set(labels)
        self.word_count = self.count_word_in_classes()

        for c in all_classes:
            N_c = labels.count(c)
            self.logprior[c] = np.log(N_c / N_docs)

            total_count = 0
            for word in self.V:
                total_count += self.word_count[c][word]

            for word in self.V:
                count = self.word_count[c][word]
                self.loglikelihoods[c][word] = np.log((count + 1) / (total_count + len(self.V)))

    def predict(self, doc):
        sums = {
            0: 0,
            2: 0,
            4: 0,
        }
        for c in self.bigdoc.keys():
            sums[c] = self.logprior[c]
            words = doc.split(" ")
            for word in words:
                if word in self.V:
                    sums[c] += self.loglikelihoods[c][words]

        return sums