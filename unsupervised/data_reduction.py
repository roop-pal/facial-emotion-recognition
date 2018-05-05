#!/usr/bin/env python

import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
# Guidelines for PCA can be found
# https://shankarmsy.github.io/posts/pca-sklearn.html#sthash.cgpBn5AH.dpuf
from sklearn.decomposition import PCA

# import csv and separate training data from test data
csvr = csv.reader(open('fer2013.csv'))
header = next(csvr)
rows = [row for row in csvr]

trn = [row[1:-1] for row in rows if row[-1] == 'Training']
tst = [row[1:-1] for row in rows if row[-1] == 'PublicTest']
tst2 = [row[1:-1] for row in rows if row[-1] == 'PrivateTest']

# csv.writer(open('train.csv', 'w+')).writerows([header[:-1]] + trn)
# print(len(trn))

# csv.writer(open('testpublic.csv', 'w+')).writerows([header[:-1]] + tst)
# print(len(tst))

# csv.writer(open('testprivate.csv', 'w+')).writerows([header[:-1]] + tst2)
# print(len(tst2))

#************************** Pre-processing ******************************
# Running PCA and Whitening

class data_reduction():

    # pre-process training data
    def pca_and_whiten(data):
        imgs = []

        for i in data:
            # print (emotions[int(i[0])])
            im = np.fromstring(i[0], dtype=int, sep=" ").reshape((48, 48))

            imgs.append(im)
            # plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
            # plt.xticks([]), plt.yticks([])
            # plt.show()

        imgs = np.array(imgs)
        nsamples, nx, ny = imgs.shape
        d2_imgs = imgs.reshape(nsamples, nx*ny)

        # Randomized PCA with whitening -- retains image features better
        pca = PCA(48, whiten= True, svd_solver='randomized')
        imgs_proj = pca.fit_transform(d2_imgs)

        # retains 0.84115 of the variance
        # print (np.cumsum(pca.explained_variance_ratio_))

        # reconstruct images with reduced dataset
        imgs_inv_proj = pca.inverse_transform(imgs_proj)
        # reshaping as 48x48 dimension images
        imgs_proj_img = np.reshape(imgs_inv_proj,(len(imgs),48,48))

        # show newly reduced images for comparison
        # for i in imgs_proj_img:
        #     plt.imshow(i, cmap = 'gray', interpolation = 'bicubic')
        #     plt.xticks([]), plt.yticks([])
        #     plt.show()

        return imgs_proj_img

    def reduce_data(self):
        trn = pre_process(trn)
        tst = pre_process(tst)
        tst2 = pre_process(tst2)

        return trn, tst, tst2
