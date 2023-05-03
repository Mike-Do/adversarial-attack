"""
Script for plotting data
"""

import pickle as pk
import pickle as pk
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn import CNN
import numpy as np
from l2attack import main as l2attack_main

# def main(input_filepath="../data/l2attack.pk", output_filepath="../fig/l2attack.png"):

#     with open(input_filepath, "rb") as fd:
#         results = pk.load(fd)

def original_vs_adversarial():
    # write a function that plots the original image and the adversarial image side by side

    # global plot for the entire script, 10 rows, 2 columns
    fig, ax = plt.subplots(10, 2, figsize=(10, 20))

    for i in range(10):
        kwargs = {
            "index": i,
            # same misclassified label of l+1(mod10)
            "target": (i + 1) % 10,
            "c": 1,
            "learning_rate": 1e-2,
            "num_epochs": 2500,  # If None, attack runs until it reaches the thresholds
            "threshold_dist": 200.0,
            "threshold_f": 0.01,
            "model_filepath": "../models/vanilla",
            "output_filepath": "../data/l2attack.pk"
        }
        l2attack_main(**kwargs)

        # load original image
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = (x_train / 255) - 0.5
        x_train = np.reshape(x_train, (-1, 28, 28, 1))
        x_test = (x_test / 255) - 0.5
        x_test = np.reshape(x_test, (-1, 28, 28, 1))
        original = x_test[i]

        # load adversarial image
        with open("../data/l2attack.pk", "rb") as fd:
            adversarial = pk.load(fd)
        adversarial = tf.reshape(adversarial, [1, 28, 28, 1])

        # plot original image
        ax = fig.add_subplot(10, 2, 2 * i + 1)
        ax.imshow(original, cmap='gray')
        ax.set_title(f"Original Image: {y_test[i]}")

        # plot adversarial image
        ax = fig.add_subplot(10, 2, 2 * i + 2)
        ax.imshow(original, cmap='gray')
        ax.set_title(f"Original Image: {y_test[i]}")

        # plot adversarial image
        ax = fig.add_subplot(10, 2, 2 * i + 2)
        ax.imshow(adversarial[0], cmap='gray')
        ax.set_title(f"Adversarial Image: {(y_test[i] + 1) % 10}")

    plt.tight_layout()
    plt.savefig("../fig/l2attack.png")
    plt.show()


if __name__ == "__main__":
    # main()
    original_vs_adversarial()