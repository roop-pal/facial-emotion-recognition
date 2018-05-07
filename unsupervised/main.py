from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import mode
import matplotlib.pyplot as plt
import numpy as np
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

    trn_targets  = np.array([row[0] for row in rows if row[-1] == 'Training'], int)
    tst_targets  = np.array([row[0] for row in rows if row[-1] == 'PublicTest'], int)
    tst2_targets = np.array([row[0] for row in rows if row[-1] == 'PrivateTest'], int)

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

    return d2_imgs

# pre-process training data
def pca_and_whiten(data):
    # translate arrays to 2d
    d2_imgs = vector_to_2d_array(data)

    # Randomized PCA with whitening -- retains image features better
    pca = PCA(n_components=2304, whiten= True, svd_solver='randomized')
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

t1, t2, t3, trn_targets, tst_targets, tst2_targets = d.load_data()
test = t2 + t3
test_targets = list(tst_targets) + list(tst2_targets)
# reshape to image dimensions 48 x 48
d2_imgs    = vector_to_2d_array(t1)
d2_t2_imgs = vector_to_2d_array(test)

# trn  = pca_and_whiten(t1)
# tst  = pca_and_whiten(t2)
# tst2 = pca_and_whiten(t3)


# print("NO PCA")
# kmeans = KMeans(n_clusters=n, random_state=0)
# clusters = kmeans.fit_predict(d2_imgs)

# labels = np.zeros_like(clusters)
# mask = (clusters == 0 )
# labels[mask] = mode(trn_targets[mask])[0]

# for i in range(n):
#     mask = (clusters == i)
#     labels[mask] = mode(trn_targets[mask])[0]

# results=accuracy_score(trn_targets, labels)
# print(results)

# print("WITH PCA")
# kmeans = KMeans(n_clusters=n, random_state=0)
# clusters = kmeans.fit_predict(trn)

# fig, ax = plt.subplots(1, n, figsize=(8, 3))
# centers = kmeans.cluster_centers_.reshape(n, 48, 48)
# for axi, center in zip(ax.flat, centers):
#     axi.set(xticks=[], yticks=[])
#     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)


# labels = np.zeros_like(clusters)
# mask = (clusters == 0 )
# labels[mask] = mode(trn_targets[mask])[0]

# for i in range(n):
#     mask = (clusters == i)
#     labels[mask] = mode(trn_targets[mask])[0]

# results=accuracy_score(trn_targets, labels)
# print(results)

# NO PCA
# 0.2558431153993521
# WITH PCA
# 0.25141941551429864

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

print("starting")
X_train, X_test, y_train, y_test = d2_imgs, d2_t2_imgs, trn_targets, test_targets

trn_targets = np.array(trn_targets, str)
tst_targets= np.array(test_targets, str)
# Compute a PCA
# reduce dimensionality to n_components
n_components = 96
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
print ("post pCA")
# apply PCA transformation
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

kmeans = KMeans(n_clusters=n, random_state=0)
clusters = kmeans.fit_predict(trn)

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
clf = MLPClassifier(hidden_layer_sizes=(1000,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)
target_names = list(map(str, range(7)))
print(classification_report(y_test, y_pred, target_names=target_names))
