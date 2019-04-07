import numpy as np


class KNearestNeighbor(object):
    
    def train(self, X, y):
    
        self.X_train = X
        self.y_train = y
    
    
    def test(self, X, k=1):

        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)


    def compute_distances(self, X):
        
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis = 1))
        return dists

    
    def predict_labels(self, dists, k=1):
    
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, dtype=np.int)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred
