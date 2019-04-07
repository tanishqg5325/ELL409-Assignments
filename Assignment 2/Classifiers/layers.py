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
