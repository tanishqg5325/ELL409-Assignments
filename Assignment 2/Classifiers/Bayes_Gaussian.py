import numpy as np


class Bayes_Gaussian(object):
    
    def __init__(self, input_shape, num_labels):
        
        self.num_classes = num_labels
        self.priors = np.zeros(num_labels)
        self.means = np.zeros((num_labels, input_shape))
        self.cov = np.zeros((num_labels, input_shape, input_shape))
    
    
    def train(self, X, y):
        
        N = X.shape[0]
        self.priors = np.bincount(y) / N
        for i in range(self.num_classes):
            x_i = X[y == i]
            self.means[i] = np.mean(x_i, axis=0)
            x_i = x_i - self.means[i]
            self.cov[i] = x_i.T.dot(x_i) / x_i.shape[0]
        self.cov[self.cov == 0] = 1e-5
        
    
    def test(self, X_test):
        
        N_test = X_test.shape[0]
        y_test = np.zeros(N_test)
        for i, x in enumerate(X_test):
            log_probs = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                log_probs[c] = (np.dot([x - self.means[c]], np.linalg.inv(self.cov[c])).dot(x - self.means[c]) + np.log(np.linalg.det(self.cov[c]))) / 2 - np.log(self.priors[c])
            
            y_test[i] = np.argmin(log_probs)
        return y_test
        
