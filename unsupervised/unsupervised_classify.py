from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import csv

class UnsupervisedClassification():
    def load_data(self, live_capture):
        im = np.fromstring( live_capture, dtype=int, sep=" ").reshape(1,-1)

        # Restore tuple
        filename = 'finalized_tuple_model.sav'
        loaded_model, loaded_X_train, loaded_y_train, loaded_score = joblib.load(open(filename, 'rb'))

        return im, loaded_X_train

    def label_image(self, live_capture):
        im, X_train = self.load_data(live_capture)
        X_test = im

        # Compute a PCA-- reduce dimensionality to n_components
        n_components = 48
        pca = PCA(n_components=n_components, whiten= True, svd_solver='randomized').fit(X_train)
        # print ("PCA variance retained", np.cumsum(pca.explained_variance_ratio_))
        # apply PCA transformation
        X_test_pca = pca.transform(X_test)
        y_pred = loaded_model.predict(X_test_pca)

        return (int(y_pred))
