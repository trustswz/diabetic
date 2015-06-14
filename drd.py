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

import numpy as np
from lasagne import layers
from nolearn.lasagne import NeuralNet
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
import csv

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer


sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(42)

FTRAIN = 'trainSampleNew.csv'
FTEST = 'trainSampleNew.csv'
FTESTRESULT = 'testResult.csv'

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
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['r'] = df['r'].apply(lambda im: np.fromstring(im, sep=' '))
    df['g'] = df['g'].apply(lambda im: np.fromstring(im, sep=' '))
    df['b'] = df['b'].apply(lambda im: np.fromstring(im, sep=' '))

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    combinedImages = []
    for ik in xrange(0,len(df['r'])):
        combinedImages.append(np.zeros((3,df['r'][ik].shape[0])))
        combinedImages[ik][0] = df['b'][ik]
        combinedImages[ik][1] = df['g'][ik]
        combinedImages[ik][2] = df['r'][ik]

    X = np.asarray(combinedImages)
    X = X / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = np.asarray(df['level'].values).reshape(-1,1)
        y = (y - 2) / 2  # scale target coordinates to [-1, 1]
        y = y.astype(np.float32)
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
    else:
        y = None

    return X, y


def load2d(test=False):
    X, y = load(test=test)
    X = X.reshape(-1, 3, 128, 128)

    for i in xrange(0,X.shape[0]):
        cv2.imwrite(repr(i)+'.png',theano2cv(X[i])*255)

    return X, y

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


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


net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv3', Conv2DLayer),
        ('conv4', Conv2DLayer),
        ('pool3', MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('conv5', Conv2DLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout5', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 128, 128),
    conv1_num_filters=96, conv1_filter_size=(7, 7),
    pool1_pool_size=(2, 2),
    conv2_num_filters=256, conv2_filter_size=(3, 3),
    pool2_pool_size=(2, 2),
    dropout1_p=0.1,
    conv3_num_filters=384, conv3_filter_size=(3, 3),
    conv4_num_filters=384, conv4_filter_size=(3, 3),
    pool3_pool_size=(2, 2),
    dropout2_p=0.2,
    conv5_num_filters=256, conv5_filter_size=(3, 3),
    dropout3_p=0.3,
    hidden1_num_units=2048,
    dropout4_p=0.4,
    hidden2_num_units=2048,
    dropout5_p=0.5,
    output_num_units=1, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200),
        ],
    max_epochs=10000,
    verbose=1,
    )


def fit():
    X, y = load2d()
    net.fit(X, y)
    with open('net.pickle', 'wb') as f:
        pickle.dump(net, f, -1)


def predict(fname='net.pickle'):
    with open(fname, 'rb') as f:
        net = pickle.load(f)

    X = load2d(test=True)[0]

    y_pred = net.predict(X)

    y_pred = np.round(y_pred * 2 + 2)

    b = open(FTESTRESULT, 'w')
    a = csv.writer(b)
    a.writerows(y_pred)
    b.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        func = globals()[sys.argv[1]]
        func(*sys.argv[2:])