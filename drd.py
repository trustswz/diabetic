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

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

FTRAIN = 'trainNew.csv'
FTEST = 'train1000New.csv'
FTESTRESULT = 'testResult.csv'

CHANNELS = 3
RANDOM_SEED = 34
IMAGE_SIZE = 256

sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(RANDOM_SEED)

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

def load(test=False):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN

    combinedImages = []
    scores = []

    for df in read_csv(os.path.expanduser(fname), chunksize=1000):
        # The Image column has pixel values separated by space; convert
        # the values to numpy arrays:
        df['r'] = df['r'].apply(lambda im: np.fromstring(im, sep=' '))
        df['g'] = df['g'].apply(lambda im: np.fromstring(im, sep=' '))
        df['b'] = df['b'].apply(lambda im: np.fromstring(im, sep=' '))

        print(df.count())  # prints the number of values for each column
        df = df.dropna()  # drop all rows that have missing values in them

        for ik in xrange(0,len(df['r'])):
            combinedImages.append(np.zeros((CHANNELS,df['r'][ik].shape[0])))
            combinedImages[len(combinedImages) - 1][0] = df['b'][ik]
            combinedImages[len(combinedImages) - 1][1] = df['g'][ik]
            combinedImages[len(combinedImages) - 1][2] = df['r'][ik]
            #combinedImages[len(combinedImages) - 1][0] = df['g'][ik]
            scores.append(df['level'][ik])

        print(len(scores))
        #break;

    X = np.asarray(combinedImages)
    X = X / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = np.asarray(scores).reshape(-1,1)
        y = (y - 2) / 2  # scale target coordinates to [-1, 1]
        y = y.astype(np.float32)
        X, y = shuffle(X, y, random_state=RANDOM_SEED)  # shuffle train data
    else:
        y = np.asarray(scores).reshape(-1,1)

    return X, y


def load2d(test=False):
    X, y = load(test=test)
    X = X.reshape(-1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    #for i in xrange(0,X.shape[0]):
        #cv2.imwrite(repr(i)+'.png',theano2cv(X[i])*255)

    return X, y

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001, initial=0.02, minInterval=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
        self.currentStepSize = initial
        self.best_valid = np.inf
        self.minInterval = minInterval

    def __call__(self, nn, train_history):

        self.ls = (self.currentStepSize - self.stop) / 50
        if math.fabs(self.ls) < math.fabs(self.minInterval):
            self.ls = self.minInterval

        current_valid = train_history[-1]['valid_loss']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.currentStepSize += self.ls;
            if self.ls > 0 and self.currentStepSize > self.start:
                self.currentStepSize = self.start
            if self.ls < 0 and self.currentStepSize < self.start:
                self.currentStepSize = self.start
        else:
            self.currentStepSize = (self.currentStepSize + self.stop)/2;
        print('new '+self.name+' '+repr(self.currentStepSize))
        getattr(nn, self.name).set_value(self.currentStepSize)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

class FlipBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb, yb

net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            #('dropout1', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            #('dropout2', layers.DropoutLayer),
            ('conv4', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            #('dropout3', layers.DropoutLayer),
            ('conv5', layers.Conv2DLayer),
            ('pool5', layers.MaxPool2DLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden1', layers.DenseLayer),
            #('dropout5', layers.DropoutLayer),
            ('hidden2', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],

        input_shape=(None, CHANNELS, IMAGE_SIZE, IMAGE_SIZE),

        conv1_num_filters=16, conv1_filter_size=(7, 7),
        pool1_pool_size=(2, 2), pool1_stride = (2, 2),
        conv2_num_filters=32, conv2_filter_size=(3, 3),
        pool2_pool_size=(2, 2), pool2_stride = (2, 2),
        conv3_num_filters=64, conv3_filter_size=(3, 3),
        pool3_pool_size=(2, 2), pool3_stride = (2, 2),
        conv4_num_filters=128, conv4_filter_size=(3, 3),
        pool4_pool_size=(2, 2), pool4_stride = (2, 2),
        conv5_num_filters=256, conv5_filter_size=(3, 3),
        pool5_pool_size=(2, 2), pool5_stride = (2, 2),

        hidden1_num_units=1024,
        hidden2_num_units=1024,

        #dropout1_p=0.1,
        #dropout2_p=0.2,
        #dropout3_p=0.3,
        dropout4_p=0.5,
        #dropout5_p=0.5,

        output_num_units=1,
        output_nonlinearity=None,

        update_learning_rate=theano.shared(float32(0.01)),
        update_momentum=theano.shared(float32(0.9)),

        regression=True,

        batch_iterator_train=FlipBatchIterator(batch_size=96),
        batch_iterator_test=BatchIterator(batch_size=96),

        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=0.01, stop=0.00001, initial=0.001, minInterval=0.00001),
            AdjustVariable('update_momentum', start=0.90, stop=0.999, initial=0.95, minInterval=-0.001),
            EarlyStopping(patience=100),
        ],
        max_epochs=5000,
        verbose=1,
        eval_size=0.2,
    )

def fit(fname=None):
    X, y = load2d()

    if fname is not None:
        net.load_params_from(fname)

    net.fit(X, y)
    with open('net.pickle', 'wb') as f:
        pickle.dump(net, f, -1)

def predict(fname='net.pickle'):
    with open(fname, 'rb') as f:
        net = pickle.load(f)

    X, y_actual = load2d(test=True)

    y_pred = net.predict(X)

    y_pred = np.round(y_pred * 2 + 2)

    b = open(FTESTRESULT, 'w')
    a = csv.writer(b)
    a.writerows(y_pred)
    a.writerows(y_actual)
    b.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        func = globals()[sys.argv[1]]
        func(*sys.argv[2:])