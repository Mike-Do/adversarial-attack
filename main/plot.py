"""
Script for plotting data
"""

import pickle as pk
import sys


def main(input_filepath="../data/result.pk", output_filepath="../fig/default.png"):

    with open(input_filepath, "wb") as fd:
        results = pk.load(fd)  # TODO we can also load MNIST data from sklearn.datasets.load_digits

    # TODO figure out what to plot


if __name__ == "__main__":
    input_filepath, output_filepath = sys.argv[1:3]
    main(input_filepath, output_filepath)
