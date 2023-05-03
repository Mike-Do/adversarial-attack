#!/bin/bash

# Given a source (integer), run the attack for each possible target.
# To execute, run `bash attack_source.sh <source>` in the terminal.

for target in 0 1 2 3 4 5 6 7 8 9
do
python l2attack.py $1 $target
done
