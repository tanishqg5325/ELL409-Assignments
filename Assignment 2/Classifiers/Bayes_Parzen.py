import numpy as np


class Bayes_Parzen(object):
    
    def __init__(self, num_labels):
        
        self.num_classes = num_labels
        self.priors = np.zeros(num_labels)
    
    
    def train(self, X, y):
        
        self.X = X
        self.y = y
        N = X.shape[0]
        self.priors = np.bincount(y) / N
        
    
    def test(self, X_test, h):
        
        N_test = X_test.shape[0]
        y_test = np.zeros(N_test)
        for i, x in enumerate(X_test):
            probs = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                x_c = self.X[self.y == c]
                probs[c] =  self.priors[c] * np.sum(np.exp(-np.sum((x_c - x) ** 2, axis=1) / (2 * h * h)))
            
            y_test[i] = np.argmax(probs)
        return y_test
        
