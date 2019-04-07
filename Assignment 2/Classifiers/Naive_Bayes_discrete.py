import numpy as np


class Naive_Bayes(object):
    
    def __init__(self, input_shape, num_labels):
        
        self.num_classes = num_labels
        self.priors = np.zeros(num_labels)
        self.params = {}
    
    
    def train(self, X, y):
        
        N, D = X.shape
        self.priors = np.bincount(y) / N
        for i in range(self.num_classes):
            for j in range(D):
                self.params[(i, j)] = np.bincount(X[y == i][:, j]) / X[y == i].shape[0]
        
    
    def test(self, X_test):
        
        N_test, D = X_test.shape
        y_test = np.zeros(N_test)
        for i, x in enumerate(X_test):
            probs = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                probs[c] = self.priors[c]
                for d in range(D):
                    if X_test[i, d] >= self.params[(c, d)].shape:
                        probs[c] = 0
                        break
                    else:
                        probs[c] = probs[c] * self.params[(c, d)][X_test[i, d]]
            y_test[i] = np.argmax(probs)
        y_test = y_test.astype(np.int)
        return y_test
   
