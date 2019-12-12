import argparse
import codecs
import sys
from read_data import DataReader
from nb_classifier import NBClassifier
import numpy as np
import operator

from read_trump_data import TrumpDataReader
from bigram_trainer import BigramTrainer
from generator import Generator


def main():

    parser = argparse.ArgumentParser(description='main')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', '-t', type=str, help='file from which to train sentiment analysis')
    group.add_argument('--load', '-l', type=str, help='file from which to load sentiment analysis')

    parser.add_argument('--destination', '-d', type=str, help='file in which to store the sentiment analysis')

    parser.add_argument('--classify', '-c', type=str, help='file from which to classify tweets')
    parser.add_argument('--generate', '-g', type=str, help='file from which to read language model')
    parser.add_argument('--validate', '-v', type=str, help='file from which to validate sentiment analysis')

    arguments = parser.parse_args()

    nb_class = NBClassifier()

    if arguments.train:
        data_reader = DataReader(arguments.train)
        tweets, labels = data_reader.tweets, data_reader.labels
        nb_class.train(tweets, labels)

    if arguments.destination:
        nb_class.write_to_file(arguments.destination)

    if arguments.load:
        nb_class.read_from_file(arguments.load)

    if arguments.validate:
        data_reader = DataReader('Data/training_data_small_v.csv')
        tweets, labels = data_reader.tweets, data_reader.labels

        confusion = np.zeros((2, 2))
        for i in range(len(tweets)):
            stats = nb_class.predict(data_reader.tweets[i])
            prediction = max(stats.items(), key=operator.itemgetter(1))[0]
            prediction = 1 if prediction == 4 else 0
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

    if arguments.classify:
        trump_data_reader = TrumpDataReader(arguments.classify)
        tweets = trump_data_reader.tweets_training

        nb_class.read_from_file('params_big.p')

        pos_index = []
        neg_index = []

        for i in range(len(tweets)):
            stats = nb_class.predict(tweets[i])
            prediction = max(stats.items(), key=operator.itemgetter(1))[0]
            prediction = 1 if prediction == 4 else 0

            if prediction == 1:
                pos_index.append(i)
            elif prediction == 0:
                neg_index.append(i)

        positive_tweets = np.take(trump_data_reader.tweets_generating, pos_index)
        negative_tweets = np.take(trump_data_reader.tweets_generating, neg_index)

        path = arguments.classify[:arguments.classify.find('.')]

        with codecs.open(path + "_positive.txt", 'w', 'UTF-8') as f:
            for row in positive_tweets:
                f.write(row + "\n")

        with codecs.open(path + "_negative.txt", 'w', 'UTF-8') as f:
            for row in negative_tweets:
                f.write(row + "\n")

    if arguments.generate:
        generator = Generator()
        generator.read_model(arguments.generate)
        print("Tweet 1:")
        generator.generate("TWEET_START_SIGN")
        print("Tweet 2:")
        generator.generate("TWEET_START_SIGN")
        print("Tweet 3:")
        generator.generate("TWEET_START_SIGN")


if __name__ == '__main__':
    main()
