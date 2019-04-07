import numpy as np


class Naive_Bayes(object):
    
    def __init__(self, input_shape, num_labels):
        
        self.num_classes = num_labels
        self.priors = np.zeros(num_labels)
        self.means = np.zeros((num_labels, input_shape))
        self.std = np.zeros((num_labels, input_shape))
    
    
    def train(self, X, y):
        
        N = X.shape[0]
        self.priors = np.bincount(y) / N
        for i in range(self.num_classes):
            self.means[i] = np.mean(X[y == i], axis=0)
            self.std[i] = np.std(X[y == i], axis=0)
        
    
    def test(self, X_test):
        
        N_test = X_test.shape[0]
        y_test = np.zeros(N_test)
        for i, x in enumerate(X_test):
            y_test[i] = np.argmin(np.sum(((x - self.means) ** 2) / (2 * self.std ** 2) + np.log(self.std), axis=1) - np.log(self.priors))
        y_test = y_test.astype(np.int)
        return y_test
        
