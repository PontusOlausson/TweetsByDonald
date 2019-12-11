import argparse
import sys
from read_data import DataReader
from nb_classifier import NBClassifier
import numpy as np
import operator

from read_trump_data import TrumpDataReader


def main():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--train', '-t', type=str, help='file from which to train sentiment analysis')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the sentiment analysis')
    parser.add_argument('--load', '-l', type=str, help='file from which to load sentiment analysis')
    parser.add_argument('--classify', '-c', type=str, help='file from which to classify tweets')

    arguments = parser.parse_args()

    nb_class = NBClassifier()

    if arguments.train:
        data_reader = DataReader(arguments.train)
        tweets, labels = data_reader.tweets, data_reader.labels
        nb_class.train(tweets, labels)
        prediction = nb_class.predict(tweets[200])
        print(prediction)
    if arguments.destination:
        nb_class.write_to_file(arguments.destination)

    if arguments.load:
        nb_class.read_from_file(arguments.load)

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

        nb_class.read_from_file('params.p')

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




if __name__ == '__main__':
    main()
