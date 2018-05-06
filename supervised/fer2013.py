import pandas as pd
import numpy as np

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
    test_x = [i.reshape((48, 48, 1)) for i in test_x]
    
    np.save('train_x', train_x)
    np.save('train_y', train_y)
    np.save('test_x', test_x)
    np.save('test_y', test_y)
    
def load_data():
    train_x = np.load('./train_x.npy')
    train_y = np.load('./train_y.npy')
    test_x = np.load('./test_x.npy')
    test_y = np.load('./test_y.npy')
    return train_x, train_y, test_x, test_y

if __name__ == "__main__":
    from time import time
    s = time()
    parser("../fer2013.csv")
    print(time()-s)