"""
CNN defined by Carlini and Walker 2016
"""
import sys
import pickle as pk
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits

class CNN(tf.keras.Model):
    def __init__(self):
        """
        Architecture taken from Carlini and Wagner 2016
        """

        """Hyperparameters"""
        # LearningRate0.1
        self.learning_rate = 0.1
        # Momentum0.9
        self.momentum = 0.9
        # DelayRate-
        # Dropout0.5
        self.dropout = 0.5
        # BatchSize128
        self.batch_size = 128
        # Epochs50
        self.epochs = 50

        # visualize loss over time
        self.loss_list = []
        # momentum-based SGD optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)


        """Architecture"""
        # Convolution+ReLU3×3×32
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        # Convolution+ReLU3×3×32
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        # MaxPooling2×2
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # Convolution+ReLU3×3×64
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        # Convolution+ReLU3×3×64
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        # MaxPooling2×2
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # FullyConnected+ReLU200
        self.fc1 = tf.keras.layers.Dense(units=200, activation='relu')
        # FullyConnected+ReLU200
        self.fc2 = tf.keras.layers.Dense(units=200, activation='relu')
        # Softmax10
        self.softmax = tf.keras.layers.Dense(units=10, activation='softmax')



def main(input_filepath="../data/dataset.pk", output_filepath="../data/result.pk"):

    with open(input_filepath, "wb") as fd:
        dataset = pk.load(fd)  # TODO we can also load MNIST data from sklearn.datasets.load_digits

    # TODO preprocess data, train CNN, and store data in a pickle file

    to_store = None  # store image and its corresponding classification - e.g. list of tuples (image, classification)
    with open(output_filepath, "rb") as fd:
        pk.dump(to_store, fd)


if __name__ == "__main__":
    input_filepath, output_filepath = sys.argv[1:3]
    main(input_filepath, output_filepath)
