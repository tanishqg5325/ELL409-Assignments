import numpy as np


class K_means(object):
    
    def __init__(self, input_shape, K):
        self.K = K
        self.centroids = np.zeros((K, input_shape))
        self.cluster_labels = np.zeros(K)
    
    
    def findClosestCentroids(self, X, centroids):
        N = X.shape[0]
        idx = np.zeros(N)

        for i, x in enumerate(X):
            idx[i] = np.argmin(np.sum((x - centroids) ** 2, axis=1))
        
        idx = idx.astype(np.int)
        return idx
    
    
    def computeCentroids(self, X, idx):
        N, D = X.shape
        self.centroids = np.zeros((self.K, D))
        to_be_deleted = []
        for i in range(self.K):
            if X[idx == i].shape[0] == 0:
                to_be_deleted += [i]
            else:
                self.centroids[i] = np.mean(X[idx == i], axis=0)
                
        self.centroids = np.delete(self.centroids, to_be_deleted, axis=0)
        self.K -= len(to_be_deleted)
        self.cluster_labels = np.zeros(self.K)
                
    
    def train(self, X, y, num_iters=10):
        N = X.shape[0]
        self.centroids = X[:self.K]
        
        for i in range(num_iters):
            idx = self.findClosestCentroids(X, self.centroids)
            self.computeCentroids(X, idx)
        
        for i in range(self.K):
            self.cluster_labels[i] = np.argmax(np.bincount(y[idx == i]))
            
    
    def test(self, X_test):
        
        return self.cluster_labels[self.findClosestCentroids(X_test, self.centroids)]

