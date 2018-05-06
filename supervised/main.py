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
    train_y = [data[i,0] for i in range(len(data)) if data[i,2] == 'Training']
    train_x = [np.fromstring(data[i,1], dtype=np.uint8, sep=' ') for i in range(len(data)) if data[i,2] == 'Training']
    test_y = [data[i,0] for i in range(len(data)) if data[i,2] == 'PublicTest' or data[i,2] == 'PrivateTest']
    test_x = [np.fromstring(data[i,1], dtype=np.uint8, sep=' ') for i in range(len(data)) if data[i,2] == 'PublicTest' or data[i,2] == 'PrivateTest']

    # whitening
    # TODO: Whiten test?
    train_x -= np.mean(train_x, axis=0)
    train_x /= np.std(train_x, axis=0)

    # reshaping into 48x48x1 image
    train_x = [i.reshape((48, 48, 1)) for i in train_x]
    test_x = [i.reshape((48, 48, 1)) for i in train_x]
    
    np.save('train_x', train_x)
    np.save('train_y', train_y)
    np.save('test_x', test_x)
    np.save('test_y', test_y)

def input_training_set():
    train_x = np.load('./train_x.npy')
    train_y = np.load('./train_y.npy')
    return {'image':train_x}, train_y    

def input_evaluation_set():
    test_x = np.load('./test_x.npy')
    test_y = np.load('./test_y.npy')
    return {'image':test_x}, test_y    

def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)

def eval_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)

if __name__ == '__main__':
    steps = 1
    batch_size = 19
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#     parser('../fer2013.csv')
    data = pd.read_csv('../fer2013.csv')
    my_feature_columns = [tf.feature_column.numeric_column('image',shape=[48,48,1])]
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        n_classes=7)
    train_x,train_y = input_training_set()
    s = time()
    print("TRAINING")
    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, batch_size),
        steps=steps)
    print("DONE TRAINING:",time()-s)
    
    test_x,test_y = input_evaluation_set()
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x, test_y, batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
#     train_y = [data[i,0] for i in range(len(data)) if data[i,2] == 'Training']
#     train_x = np.load('./train_x.npy')
#     train_y = np.load('./train_y.npy')
#     test_x = np.load('./test_x.npy')
#     test_y = np.load('./test_y.npy')
    
#     plt.figure(0)
#     plt.imshow(train_x[0,:,:,0], interpolation='none', cmap='gray')
#     plt.show()
    