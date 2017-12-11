#!/bin/sh

# logistic regression
python3 logistic_reg.py

# random forest
python3 random_forest.py

# cnn
python3 glove_cnn.py -o train
python3 glove_cnn.py -o predict

