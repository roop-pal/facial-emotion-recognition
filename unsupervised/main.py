from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import mode
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

#************************** Pre-processing ******************************
# Running PCA and Whitening
def load_data():
    # import csv and separate training data from test data
    csvr = csv.reader(open('fer2013.csv'))
    header = next(csvr)
    rows = [row for row in csvr]

    trn  = [row[1:-1] for row in rows if row[-1] == 'Training']
    tst  = [row[1:-1] for row in rows if row[-1] == 'PublicTest']
    tst2 = [row[1:-1] for row in rows if row[-1] == 'PrivateTest']

    trn_targets  = [row[0] for row in rows if row[-1] == 'Training']
    tst_targets  = [row[0] for row in rows if row[-1] == 'PublicTest']
    tst2_targets = [row[0] for row in rows if row[-1] == 'PrivateTest']

    return trn, tst, tst2, trn_targets, tst_targets, tst2_targets

def vector_to_2d_array(data):
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

    return d2_imgs, imgs

# pre-process training data
def pca_and_whiten(data):
    # translate arrays to 2d
    d2_imgs, imgs = vector_to_2d_array(data)

    # Randomized PCA with whitening -- retains image features better
    pca = PCA(n_components=48, whiten= True, svd_solver='randomized')
    imgs_proj = pca.fit_transform(d2_imgs)

    # retains 0.8427 of the variance
    # print (np.cumsum(pca.explained_variance_ratio_))

    #reconstruct images with reduced dataset
    # better kmeans results without reconstructed image
    #imgs_inv_proj = pca.inverse_transform(imgs_proj)
    #print(imgs_inv_proj.shape)
    #imgs_proj_img = np.reshape(imgs_inv_proj,(len(imgs),48,48))

    return imgs_proj

# emotion indices are the labels in kmeans prediction
emotions = ["Angry", "Disgust", "Fear", "Joy", "Sad", "Surprise", "Neutral"]
n = len(emotions)

t1, t2, t3, trn_targets, tst_targets, tst2_targets = load_data()
test = t2 + t3
test_targets = list(tst_targets) + list(tst2_targets)
# reshape to image dimensions 48 x 48
d2_imgs, imgs       = vector_to_2d_array(t1)
d2_t2_imgs, t2_imgs = vector_to_2d_array(test)

# NO PCA
# 0.2558431153993521
# WITH PCA
# 0.25141941551429864

# change integers to strings for MLP labels
trn_targets = np.array(trn_targets, str)
test_targets= np.array(test_targets, str)

print("starting")
# assign training set and test sets
X_train, X_test, y_train, y_test = d2_imgs, d2_t2_imgs, trn_targets, test_targets

# Compute a PCA
# reduce dimensionality to n_components
n_components = 48
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# print ("PCA variance retained", np.cumsum(pca.explained_variance_ratio_))
# apply PCA transformation
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

X_inv_proj = pca.inverse_transform(X_train_pca)
X_proj_img = np.reshape(X_inv_proj,(len(X_train),48,48))

# #Setup a figure 8 inches by 8 inches
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# plot the faces, each image is 64 by 64 dimension but 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(imgs[i], cmap=plt.cm.bone, interpolation='nearest')

#Setup a figure 8 inches by 8 inches
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# plot the faces, each image is 64 by 64 dimension but 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(X_proj_img[i], cmap=plt.cm.bone, interpolation='nearest')

# reshape reduced dataset for kmeans
nsamples, nx, ny = X_proj_img.shape
d2_X_proj = X_proj_img.reshape(nsamples, nx*ny)

kmeans = KMeans(n_clusters=n, random_state=0)
clusters = kmeans.fit_predict(d2_X_proj)

# print("Cluster faces post-PCA")
fig, ax = plt.subplots(1, n, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(n, 48, 48)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

labels = np.zeros_like(clusters)
mask = (clusters == 0 )
labels[mask] = mode(y_train[mask])[0]

for i in range(n):
    mask = (clusters == i)
    labels[mask] = mode(y_train[mask])[0]
labels = np.array(labels, str)
results=accuracy_score(y_train, labels)
print(results)

# train a neural network
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)
target_names = list(map(str, range(7)))
print(classification_report(y_test, y_pred, target_names=target_names))
