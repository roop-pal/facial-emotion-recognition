from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import numpy as np
import pickle
import csv

# Restore tuple
filename = './unsupervised/finalized_tuple_model.pkl'
loaded_model, loaded_X_train, loaded_y_train, loaded_score = pickle.load(open(filename, 'rb'))

class UnsupervisedClassification():
	def label_image(self, live_capture):
		X_train = loaded_X_train
		imgs = np.array(live_capture)
		nsamples, nx, ny = imgs.shape
		X_test = imgs.reshape(nsamples, nx*ny)

		# Compute a PCA-- reduce dimensionality to n_components
		n_components = 48
		pca = PCA(n_components=n_components, whiten= True, svd_solver='randomized').fit(X_train)
		# print ("PCA variance retained", np.cumsum(pca.explained_variance_ratio_))
		# apply PCA transformation
		X_test_pca = pca.transform(X_test)
		y_pred = loaded_model.predict(X_test_pca)

		return (int(y_pred))
