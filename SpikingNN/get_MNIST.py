#!/usr/bin/env python

import numpy as np
import time
import os.path
#import cPickle as pickle
from struct import unpack
from matplotlib.pyplot import *
from brian2 import *

# specify the location of the MNIST data
MNIST_data_path = '../../data/raw/'


def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile(picklename):
        print('yes')
        data = pickle.load(open(picklename))
        return data

    # Open the images with gzip in read binary mode
    print('no')
    if bTrain:
        images = open(MNIST_data_path + 'train-images-idx3-ubyte','rb')
        labels = open(MNIST_data_path + 'train-labels-idx1-ubyte','rb')
    else:
        images = open(MNIST_data_path + 't10k-images-idx3-ubyte','rb')
        labels = open(MNIST_data_path + 't10k-labels-idx1-ubyte','rb')
    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = unpack('>I', images.read(4))[0]
    rows = unpack('>I', images.read(4))[0]
    cols = unpack('>I', images.read(4))[0]
    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = unpack('>I', labels.read(4))[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')
    # Get the data
    x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
        y[i] = unpack('>B', labels.read(1))[0]

    data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
    # pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

# testing = get_labeled_data(MNIST_data_path + 'testing.pickle', bTrain=False)

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

