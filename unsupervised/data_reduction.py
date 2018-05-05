#!/usr/bin/env python

from itertools import zip_longest
import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
# Guidelines for PCA can be found
# https://shankarmsy.github.io/posts/pca-sklearn.html#sthash.cgpBn5AH.dpuf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


#************************** Pre-processing ******************************
# Running PCA and Whitening

class DataReduction():
    def load_data(self):
        # import csv and separate training data from test data
        csvr = csv.reader(open('fer2013.csv'))
        header = next(csvr)
        rows = [row for row in csvr]

        trn  = [row[1:-1] for row in rows if row[-1] == 'Training']
        tst  = [row[1:-1] for row in rows if row[-1] == 'PublicTest']
        tst2 = [row[1:-1] for row in rows if row[-1] == 'PrivateTest']

        # csv.writer(open('train.csv', 'w+')).writerows([header[:-1]] + trn)
        # print(len(trn))

        # csv.writer(open('testpublic.csv', 'w+')).writerows([header[:-1]] + tst)
        # print(len(tst))

        # csv.writer(open('testprivate.csv', 'w+')).writerows([header[:-1]] + tst2)
        # print(len(tst2))

        return trn, tst, tst2

    # pre-process training data
    def pca_and_whiten(self, data):
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

    def reduce_data(self, data):
        trn, tst, tst2 = data

        trn  = self.pca_and_whiten(trn)
        tst  = self.pca_and_whiten(tst)
        tst2 = self.pca_and_whiten(tst2)

        # translate arrays to 2d
        nsamples, nx, ny = trn.shape
        d2_trn = trn.reshape(nsamples, nx*ny)

        nsamples, nx, ny = tst.shape
        d2_tst = tst.reshape(nsamples, nx*ny)

        nsamples, nx, ny = tst2.shape
        d2_tst2 = tst2.reshape(nsamples, nx*ny)

        return d2_trn, d2_tst, d2_tst2

#************* Run Kmeans on pre-processed data ***************
if __name__ == "__main__":
    d = DataReduction()
    t1, t2, t3 = d.load_data()
    data = [t1, t2, t3]
    t1 = np.array(t1)

    trn, tst, tst2 = d.reduce_data(data)

    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    n = len(emotions)
    kmeans = KMeans(n_clusters=n).fit(trn)
    labels = kmeans.predict(trn)
    centroids = kmeans.cluster_centers_


    csv_output = csv.writer(open('trn_results.csv', 'w+'))
    csv_output.writerow(['pixels', 'emotion'])
    csv_output.writerows(zip_longest(*[t1, labels]))


    # for c, i in  enumerate(trn):
        # print(labels[c])


        # im = trn[c].reshape(48, 48)
        # plt.imshow(im, cmap = 'gray', interpolation = 'bicubic')
        # plt.xticks([]), plt.yticks([])
        # plt.show()


        # og = np.fromstring(t1[c][0], dtype=int, sep=" ").reshape((48, 48))
        # plt.imshow(og, cmap = 'gray', interpolation = 'bicubic')
        # plt.xticks([]), plt.yticks([])
        # plt.show()
        # break
