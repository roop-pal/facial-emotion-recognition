import csv
import numpy as np
from matplotlib import pyplot as plt
import alexnet

def parser():
    csvr = csv.reader(open('../fer2013.csv'))
    header = next(csvr)
    rows = [row for row in csvr]
    trn = [row[:-1] for row in rows if row[-1] == 'Training']
    return trn
    
if __name__ == "__main__":
    data = parser()
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    im = np.fromstring(data[0][1], dtype=int, sep=" ").reshape((48, 48))
    plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()
    print(data[0])
    print(data[1])
    print(len(data))
