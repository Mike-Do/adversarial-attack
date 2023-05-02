"""
Script for plotting data
"""

import pickle as pk


def main(input_filepath="../data/l2attack.pk", output_filepath="../fig/l2attack.png"):

    with open(input_filepath, "rb") as fd:
        results = pk.load(fd)


if __name__ == "__main__":
    main()
