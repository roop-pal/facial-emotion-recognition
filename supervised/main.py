import csv
import numpy as np
from matplotlib import pyplot as plt
import alexnet
from time import time
import tensorflow as tf
import pandas as pd

def parser(filename):
    # read in data
    data = pd.read_csv(filename).values
    Y_train = [data[i,0] for i in range(len(data)) if data[i,2] == 'Training']
    X_train = [np.fromstring(data[i,1], dtype=np.uint8, sep=' ') for i in range(len(data)) if data[i,2] == 'Training']
    Y_test = [data[i,0] for i in range(len(data)) if data[i,2] == 'PublicTest' or data[i,2] == 'PrivateTest']
    X_test = [np.fromstring(data[i,1], dtype=np.uint8, sep=' ') for i in range(len(data)) if data[i,2] == 'PublicTest' or data[i,2] == 'PrivateTest']

    # whitening
    # TODO: Whiten test?
    X_train -= np.mean(X_train, axis=0)
    X_train /= np.std(X_train, axis=0)

    # reshaping into 48x48x1 image
    X_train = [i.reshape((48, 48, 1)) for i in X_train]
    X_test = [i.reshape((48, 48, 1)) for i in X_train]
    
    np.save('X_train', X_train)
    np.save('Y_train', Y_train)
    np.save('X_test', X_test)
    np.save('Y_test', Y_test)

if __name__ == '__main__':
    # minibatch by 19
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#     parser('../fer2013.csv')
    
    X_train = np.load('./X_train.npy')
    Y_train = np.load('./Y_train.npy')
    X_test = np.load('./X_test.npy')
    Y_test = np.load('./Y_test.npy')
    
    plt.figure(0)
    plt.imshow(X_train[0,:,:,0], interpolation='none', cmap='gray')
    plt.show()