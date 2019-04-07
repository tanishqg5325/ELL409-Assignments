import numpy as np


class Naive_Bayes(object):
    
    def __init__(self, input_shape, num_labels, num_bins):
        
        self.num_classes = num_labels
        self.priors = np.zeros(num_labels)
        self.num_bins = num_bins
        self.prob = np.zeros([input_shape, num_bins, num_labels])
        self.binMaxs = np.zeros([input_shape, num_bins])
    
    
    def train(self, X, y):
        
        N, D = X.shape
        self.priors = np.bincount(y) / N
        X_t = X.T
        B = self.num_bins
        for i in range(D):
            inc = (np.max(X_t[i]) - np.min(X_t[i])) / B
            bin_Max = np.min(X_t[i])
            for b in range(B):
                bin_Max += inc
                self.binMaxs[i, b] = bin_Max
                indexes = np.where((X_t[i] >= (bin_Max - inc)) & (X_t[i] <= bin_Max))[0]
                if indexes.shape[0] == 0:
                    self.prob[i, b] = np.ones(self.num_classes)
                else:
                    for c in range(self.num_classes):
                        self.prob[i, b, c] = np.sum(y[indexes] == c) / indexes.shape[0]
        
    
    def test(self, X_test):
        
        N_test, D = X_test.shape
        y_test = np.zeros(N_test)
        for i, x in enumerate(X_test):
            probs = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                probs[c] = self.priors[c]
                for d in range(D):
                    bI = 0
                    while bI < self.num_bins - 1 and x[d] > self.binMaxs[d, bI]:
                        bI += 1
                    probs[c] *= self.prob[d, bI, c]
            y_test[i] = np.argmax(probs)
        y_test = y_test.astype(np.int)
        return y_test
   
