import numpy as np
from .layers import *


class Logistic(object):

    def __init__(self, input_dim, num_classes, weight_scale=1e-3, initial_weights=None, initial_b=None, reg=0.0):
        self.params = {}
        self.reg = reg
        if initial_weights is not None:
            self.params['W'] = initial_weights
            self.params['b'] = initial_b.reshape(-1)
        else:
            self.params['W'] = weight_scale * np.random.randn(input_dim, num_classes)
            self.params['b'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        N = X.shape[0]
        W, b = self.params['W'], self.params['b']
        scores, cache = affine_forward(X, W, b)

        if y is None:
            return scores

        grads = {}
        loss, up_grad = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(W ** 2)
        up_grad, grads['W'], grads['b'] = affine_backward(up_grad, cache)
        grads['W'] += self.reg * W
        return loss, grads
    
    
    def test(self, X):
        return np.argmax(self.loss(X), axis=1)

