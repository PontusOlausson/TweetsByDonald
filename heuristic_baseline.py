import argparse
import codecs
import operator
import numpy as np

from read_data import DataReader


class HeuristicBaseline(object):
    positive_words = []
    negative_words = []
    tweets = []

    def read_data(self, positive_path, negative_path):
        with codecs.open(positive_path, 'r', 'utf-8') as f:
            for row in f:
                self.positive_words.append(row.strip())

        with codecs.open(negative_path, 'r', 'utf-8') as f:
            for row in f:
                self.negative_words.append(row.strip())

    def classify(self, validation_path):
        data_reader = DataReader(validation_path)
        tweets, labels = data_reader.tweets, data_reader.labels

        confusion = np.zeros((2, 2))
        for i in range(len(tweets)):
            n_negative = 0
            n_positive = 0

            split_tweet = tweets[i].split()
            for token in split_tweet:
                if token in self.negative_words:
                    n_negative += 1
                if token in self.positive_words:
                    n_positive += 1

            prediction = 0
            if n_positive > n_negative:
                prediction = 1

            correct = 1 if labels[i] == 4 else 0
            confusion[prediction][correct] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))

        for i in range(2):
            recall = confusion[i, i] / sum(confusion[:, i])
            precision = confusion[i, i] / sum(confusion[i, :])
            print('Class %i: Recall=%0.6f, Precision=%0.6f' % (i, recall, precision))

        print('Accuracy=%.06f' % ((confusion[0, 0] + confusion[1, 1]) / (np.sum(confusion))))


def main():
    parser = argparse.ArgumentParser(description='heuristic_baseline')
    parser.add_argument('--negative', '-n', required=True, type=str, help='file from which to read negative words')
    parser.add_argument('--positive', '-p', required=True, type=str, help='file from which to read positive words')
    parser.add_argument('--classify', '-c', required=True, type=str, help='file from which to read classify tweets')

    arguments = parser.parse_args()

    hb = HeuristicBaseline()
    hb.read_data(arguments.positive, arguments.negative)
    hb.classify(arguments.classify)


if __name__ == '__main__':
    main()


