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

        trn_targets  = np.array([row[0] for row in rows if row[-1] == 'Training'], int)
        tst_targets  = np.array([row[0] for row in rows if row[-1] == 'PublicTest'], int)
        tst2_targets = np.array([row[0] for row in rows if row[-1] == 'PrivateTest'], int)

        return trn, tst, tst2, trn_targets, tst_targets, tst2_targets

    def vector_to_2d_array(self, data):
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

        return d2_imgs

    # pre-process training data
    def pca_and_whiten(self, data):
        # translate arrays to 2d
        d2_imgs = self.vector_to_2d_array(data)

        # Randomized PCA with whitening -- retains image features better
        pca = PCA(48*48, whiten= True, svd_solver='randomized')
        imgs_proj = pca.fit_transform(d2_imgs)

        # retains 0.8427 of the variance
        # print (np.cumsum(pca.explained_variance_ratio_))

        #reconstruct images with reduced dataset
        # better kmeans results without reconstructed image
        #imgs_inv_proj = pca.inverse_transform(imgs_proj)
        #print(imgs_inv_proj.shape)
        #imgs_proj_img = np.reshape(imgs_inv_proj,(len(imgs),48,48))

        return imgs_proj

    def reduce_data(self, data):
        trn, tst, tst2 = data

        trn  = self.pca_and_whiten(trn)
        tst  = self.pca_and_whiten(tst)
        tst2 = self.pca_and_whiten(tst2)

        d2_trn, d2_tst, d2_tst2 = trn, tst, tst2

        return d2_trn, d2_tst, d2_tst2

if __name__ == "__main__":

    emotions = ["Angry", "Disgust", "Fear", "Joy", "Sad", "Surprise", "Neutral"]
    n = len(emotions)

    d = DataReduction()
    t1, t2, t3, trn_targets, tst_targets, tst2_targets = d.load_data()

    # reshape to image dimensions 48 x 48
    d2_imgs    = d.vector_to_2d_array(t1)
    d2_t2_imgs = d.vector_to_2d_array(t2)

    #************* Run Kmeans on pre-processed training data & test ***************
    kmeans = KMeans(n_clusters=n).fit(d2_imgs, trn_targets)
    labels = kmeans.predict(d2_t2_imgs)
    from sklearn.metrics import accuracy_score

    print('Accuracy without PCA: {}'.format(accuracy_score(tst_targets, labels)))

    # centers = kmeans.cluster_centers_
    # fig, ax = plt.subplots(1, 7, figsize=(8, 3))
    # centers = kmeans.cluster_centers_.reshape(n, 48, 48)
    # for axi, center in zip(ax.flat, centers):
    #     axi.set(xticks=[], yticks=[])
    #     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

    # csv_output = csv.writer(open('trn_results.csv', 'w+'))
    # csv_output.writerow(['pixels', 'emotion'])
    # csv_output.writerows(zip_longest(*[t1, labels]))

    data = [t1, t2, t3]
    trn, tst, tst2 = d.reduce_data(data)

    #************* Run Kmeans on pre-processed training data & test ***************
    kmeans = KMeans(n_clusters=n).fit(trn, trn_targets)
    labels = kmeans.predict(d2_t2_imgs)
    from sklearn.metrics import accuracy_score
    print('Accuracy with PCA : {}'.format(accuracy_score(tst_targets, labels)))

    # centers = kmeans.cluster_centers_
    #
    # fig, ax = plt.subplots(1, 7, figsize=(8, 3))
    # centers = kmeans.cluster_centers_.reshape(n, 48, 48)
    # for axi, center in zip(ax.flat, centers):
    #     axi.set(xticks=[], yticks=[])
    #     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

    # Generate some data
    # from sklearn.datasets.samples_generator import make_blobs
    # trn, trn_targets = make_blobs(n_samples=len(t1[0]), centers=7,
    #                        cluster_std=0.60, random_state=0)
    # X = trn[:, ::-1] # flip axes for better plotting
    #
    # # Plot the data with K Means Labels
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(7, random_state=0)
    # labels = kmeans.fit(X).predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
