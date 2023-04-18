"""
Generating adversarial examples using the l2 attack as in Carlini and Walker 2016
"""

import sys
import pickle as pk
import tensorflow as tf
from sklearn.datasets import load_digits


# TODO define class / functions for CNN here

def main(input_filepath="../data/dataset.pk", output_filepath="../data/l2attack.pk"):

    with open(input_filepath, "wb") as fd:
        dataset = pk.load(fd)  # TODO we can also load MNIST data from sklearn.datasets.load_digits

    # TODO preprocess data, generate adversarial examples

    to_store = None  # data to store goes here
    with open(output_filepath, "rb") as fd:
        pk.dump(to_store, fd)


if __name__ == "__main__":
    input_filepath, output_filepath = sys.argv[1:3]
    main(input_filepath, output_filepath)
