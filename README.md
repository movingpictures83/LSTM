# LSTM
# Language: Python
# Input: TXT
# Output: PREFIX
# Tested with: PluMA 1.1, Python 3.6

PluMA plugin that runs Long-Short Term Memory (Hochreither and Schmidhuber, 1997).

The plugin expectes as input a tab-delimited file of keyword-value pairs:
csvfile: Input data set
trainpct: Percent of dataset to use for training (rest is test)
slidingwindow: Sliding window size
features: Number of features
inputsize: Total number of inputs
epochs: Number of epochs to run

Results on both the training and test set are output, using the user-specified prefix

