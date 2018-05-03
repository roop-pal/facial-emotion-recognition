import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
# See more at: https://shankarmsy.github.io/posts/pca-sklearn.html#sthash.cgpBn5AH.dpuf
from sklearn.decomposition import PCA

d = []
imgs = []
i = 0
with open("fer2013.csv") as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        d.append(row)

# ignore emotion labeling
# emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ignore emotion assignment
for i in d[1:]:
    # print (emotions[int(i[0])])
    im = np.fromstring(i[1], dtype=int, sep=" ").reshape((48, 48))

    imgs.append(im)
    # plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

imgs = np.array(imgs)
nsamples, nx, ny = imgs.shape
d2_imgs = imgs.reshape(nsamples, nx*ny)

# instantiating PCA with
pca = PCA(48)

imgs_proj = pca.fit_transform(d2_imgs)
