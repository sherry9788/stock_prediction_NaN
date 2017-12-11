#!/bin/sh

# preprocess and get word vectors
python3 preprocess.py

# logistic regression
python3 logistic_reg.py

# random forest
python3 random_forest.py

# cnn
python3 glove_cnn.py -o train
python3 glove_cnn.py -o predict

# predict
python3 predict.py
