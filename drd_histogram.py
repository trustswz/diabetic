"""
To use this script, first run this to fit your first model:
  python kfkd.py fit
Then train a bunch of specialists that intiliaze their weights from
your first model:
  python kfkd.py fit_specialists net.pickle
Plot their error curves:
  python kfkd.py plot_learning_curves net-specialists.pickle
And finally make predictions to submit to Kaggle:
  python kfkd.py predict net-specialists.pickle
"""

import cPickle as pickle
import os
import sys
import cv2
import math

import numpy as np
import theano
import csv
from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from theano import tensor as T
from scipy.spatial import distance as dist

FTRAIN = 'train1000New.csv'
FTEST = 'train1000New.csv'
KNN = 10

def float32(k):
    return np.cast['float32'](k)

def theano2cv(img):
    """
    Converts image in theano CNN form (c x m x n) back to OpenCV form (m x n x c)
    """
    if img.shape[0] > 1:  # Color
        return np.rollaxis(np.rollaxis(np.squeeze(img), 2), 2)
    else:  # Greyscale
        return np.squeeze(img)

def load():
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname_test = FTEST
    fname_train = FTRAIN

    train_scores = []
    train_hists = []
    test_hists = []

    for df in read_csv(os.path.expanduser(fname_train), chunksize=1000):

        df['g'] = df['g'].apply(lambda im: np.fromstring(im, sep=' '))

        print(df.count())  # prints the number of values for each column
        df = df.dropna()  # drop all rows that have missing values in them

        for ik in xrange(0,len(df['r'])):
            train_hists.append(np.histogram(df['g'][ik],bins=64,range=(0,255))[0])
            train_scores.append(df['level'][ik])

        print(len(train_hists))

    for df in read_csv(os.path.expanduser(fname_test), chunksize=1000):

        df['g'] = df['g'].apply(lambda im: np.fromstring(im, sep=' '))

        print(df.count())  # prints the number of values for each column
        df = df.dropna()  # drop all rows that have missing values in them

        for ik in xrange(0,len(df['r'])):
            test_hists.append(np.histogram(df['g'][ik],bins=64,range=(0,255))[0])

        print(len(test_hists))

    return train_hists, train_scores, test_hists

def calculate(test_hist=None, train_hists=None, train_scores=None):
    scores = []
    for i in xrange(0,len(train_hists)):
        distance = dist.euclidean(train_hists[i],test_hist)
        if distance > 0:
            scores.append((dist.euclidean(train_hists[i],test_hist), train_scores[i]));
    sorted_scores = np.sort(scores, axis=0)
    total = 0
    for i in xrange(0,KNN):
        total += sorted_scores[i][1]
    total /= float32(KNN)
    return math.round(total)

def predict():
    train_hists, train_scores, test_hists = load()

    test_scores = []

    for test_hist in test_hists:
        test_scores.append(calculate(test_hist, train_hists, train_scores))

    b = open(FTESTRESULT, 'w')
    a = csv.writer(b)
    a.writerows(test_scores)
    #TODO this is to be removed!
    a.writerows(train_scores)
    b.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        func = globals()[sys.argv[1]]
        func(*sys.argv[2:])