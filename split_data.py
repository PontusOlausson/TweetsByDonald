import argparse
import codecs
import csv

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='split_data')
    parser.add_argument('--file', '-f', type=str, help='file from which to split data')
    arguments = parser.parse_args()

    ratio = 0.8

    training_data = []
    validation_data = []

    with codecs.open(arguments.file, 'r', 'latin-1') as f:
        reader = csv.reader(f)
        for row in reader:
            rand = np.random.randint(0, 101) / 100
            if rand < ratio:
                training_data.append(row)
            else:
                validation_data.append(row)

    path = arguments.file[:arguments.file.find('.')]

    with codecs.open(path + "_t.csv", 'w', 'latin-1') as f:
        for row in training_data:
            f.write(",".join('"' + item + '"' for item in row) + "\n")

    with codecs.open(path + "_v.csv", 'w', 'latin-1') as f:
        for row in validation_data:
            f.write(",".join('"' + item + '"' for item in row) + "\n")


if __name__ == "__main__":
    main()
