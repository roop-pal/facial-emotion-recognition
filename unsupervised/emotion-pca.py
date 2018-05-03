#!/usr/bin/env python

import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
# See more at: https://shankarmsy.github.io/posts/pca-sklearn.html#sthash.cgpBn5AH.dpuf
from sklearn.decomposition import PCA

csvr = csv.reader(open('fer2013.csv'))
header = next(csvr)
rows = [row for row in csvr]

trn = [row[1:-1] for row in rows if row[-1] == 'Training']

csv.writer(open('test.csv', 'w+')).writerows([header[:-1]] + trn)
# print(len(trn))

tst = [row[1:-1] for row in rows if row[-1] == 'PublicTest']
# csv.writer(open('test.csv', 'w+')).writerows([header[:-1]] + tst)
# print(len(tst))

tst2 = [row[1:-1] for row in rows if row[-1] == 'PrivateTest']
# csv.writer(open('testprivate.csv', 'w+')).writerows([header[:-1]] + tst2)
# print(len(tst2))

#************************** Pre-processing ******************************
# Running PCA and Whitening
imgs = []

# pre-process training data
for i in trn:
    # print (emotions[int(i[0])])
    im = np.fromstring(i[0], dtype=int, sep=" ").reshape((48, 48))

    imgs.append(im)
    # plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

imgs = np.array(imgs)
nsamples, nx, ny = imgs.shape
d2_imgs = imgs.reshape(nsamples, nx*ny)

# instantiating PCA with
pca = PCA(48, whiten= True)
imgs_proj = pca.fit_transform(d2_imgs)
