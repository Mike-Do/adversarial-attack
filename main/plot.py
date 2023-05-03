"""
Script for plotting data
"""

import pickle as pk
import pickle as pk
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def original_vs_adversarial():
    """
    Function that plots the original image and the adversarial image in a 10x10 grid
    for each row, the original image is the same, but the adversarial image varies based on the column digit

    All data is stored as a pk file in ../data where the format is ApB.pk, where A is the source digit and B is the target digit
    """

    # global plot for the entire script, 10 rows, 10 columns with spaces in between and each row and column is enumerated 0-9
    fig, ax = plt.subplots(10, 10, figsize=(10, 10))
    fig.suptitle("Target Classification (L2)", fontsize=16, fontweight="bold")
    fig.supylabel("Source Classification", fontsize=16, fontweight="bold")

    # give each row a title
    for i in range(10):
        ax[i, 0].set_ylabel(f"{i}", fontsize=16)

    # give each column a title
    for i in range(10):
        ax[0, i].set_title(f"{i}", fontsize=16)

    # reduce spacing between subplots
    fig.tight_layout()

    # loop through each row and column and plot the image by pulling the data from the pk file and placing it in the correct row and column
    for source in range(10):
        for target in range(10):
            input_filepath=f"../data/{source}p{target}.pk"
            #  if the input_filepath does not exist, then skip

            with open(input_filepath, "rb") as fd:
                xp = pk.load(fd)
          
            xp = tf.reshape(xp, [1, 28, 28, 1])
            ax[source, target].imshow(xp[0], cmap='gray')
            # preserve the label on the y-axis
            if target == 0:
                # keep the label on the y-axis
                ax[source, target].set_ylabel(f"{source}", fontsize=16)
                # remove all other axis elements
                ax[source, target].set_xticks([])
                ax[source, target].set_yticks([])
            else:
                ax[source, target].axis("off")

    plt.tight_layout()
    plt.savefig("../fig/l2attack.png")
    plt.show()

if __name__ == "__main__":
    original_vs_adversarial()