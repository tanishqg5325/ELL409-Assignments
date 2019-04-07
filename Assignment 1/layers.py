import numpy as np


def affine_forward(x, w, b):
    N = x.shape[0]
    X = x.reshape(N, -1)
    out = X.dot(w) + b;
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    N = x.shape[0]
    db = np.sum(dout, axis=0);
    X = x.reshape(N, -1)
    dX = dout.dot(w.T)
    dw = X.T.dot(dout)
    dx = dX.reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx


def sigmoid_forward(x):
    out = 1./(1 + np.exp(-x))
    cache = out
    return out, cache


def sigmoid_backward(dout, cache):
    out = cache
    dx = dout * out * (1 - out)
    return dx


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_sigmoid_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, sigm_cache = sigmoid_forward(a)
    cache = (fc_cache, sigm_cache)
    return out, cache


def affine_sigmoid_backward(dout, cache):
    fc_cache, sigm_cache = cache
    da = sigmoid_backward(dout, sigm_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def svm_loss(x, y):
    N = x.shape[0]
    correct_class_scores = x[list(range(N)), y]
    margin = x - correct_class_scores[:, np.newaxis] + 1
    loss = np.sum(np.maximum(margin, 0)) - N
    loss /= N
    dx = np.float64(margin > 0)
    dx[list(range(N)), y] -= np.sum(dx, axis=1)
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    N = x.shape[0]
    scores = x - np.max(x, axis=1)[:, np.newaxis]
    scores = np.exp(scores)
    scores /= np.sum(scores, axis=1)[:, np.newaxis]
    loss = -np.sum(np.log(scores[list(range(N)), y])) / N
    dx = scores.copy()
    dx[list(range(N)), y] -= 1
    dx /= N
    return loss, dx
