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
        super().__init__()
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
        self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate, momentum=self.momentum)


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
        # Dense(10)
        self.fc3 = tf.keras.layers.Dense(10)
        # Softmax10
        self.softmax = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on the network

        :param inputs: input images
        :param is_testing: if True, we do not apply dropout
        :return: output of the network
        """

        # Convolution+ReLU3×3×32
        x = self.conv1(inputs)
        # Convolution+ReLU3×3×32
        x = self.conv2(x)
        # MaxPooling2×2
        x = self.maxpool1(x)
        # Convolution+ReLU3×3×64
        x = self.conv3(x)
        # Convolution+ReLU3×3×64
        x = self.conv4(x)
        # MaxPooling2×2
        x = self.maxpool2(x)
        # Flatten
        x = tf.keras.layers.Flatten()(x)
        # FullyConnected+ReLU200
        x = self.fc1(x)
        # FullyConnected+ReLU200
        x = self.fc2(x)
        # Dense 10
        x = self.fc3(x)

        return x
    
    def loss(self, labels, logits):
        """
        Computes the loss of the network for L2 attack
        The loss is the cross entropy loss of the network
        Loss for L2 attack: f(x)= max(max{Z(x)i:i=t} - Z(x)t,−κ)


        :param logits: output of the network
        :param labels: true labels
        :return: loss
        """

        cce = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(cce)



def main(input_filepath="../data/dataset.pk", output_filepath="../data/result.pk"):

    # with open(input_filepath, "wb") as fd:
    #     dataset = pk.load(fd)  # TODO we can also load MNIST data from sklearn.datasets.load_digits

    # TODO preprocess data, train CNN, and store data in a pickle file
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255) - 0.5
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = (x_test / 255) - 0.5
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    model = CNN()
    model.compile(loss=model.loss, optimizer=model.optimizer, metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(x_train, y_train, batch_size=model.batch_size, epochs=model.epochs, shuffle = True)
    model.save("trained_model")
    model.evaluate(x_test, y_test, batch_size=model.batch_size)

    to_store = None  # store image and its corresponding classification - e.g. list of tuples (image, classification)
    # with open(output_filepath, "rb") as fd:
    #     pk.dump(to_store, fd)


if __name__ == "__main__":
    # input_filepath, output_filepath = sys.argv[1:3]
    # main(input_filepath, output_filepath)
    main()