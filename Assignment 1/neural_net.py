import numpy as np
from layers import *


class NeuralNet(object):

    def __init__(self, hidden_dims, input_dim=28*28, num_classes=10, reg=0.0,
                 weight_scale=1e-2, activation='relu', loss_fn='softmax', dtype=np.float32):
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        
        if activation not in ['relu', 'sigmoid']:
            print("Invalid activation")
            return
        
        self.activation = activation
        
        if loss_fn not in ['svm', 'softmax']:
            print("Invalid Loss Function")
            return
        
        self.loss_fn = loss_fn
        
        self.params = {}
        
        dims = [input_dim] +  hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W' + str(i+1)] = weight_scale * np.random.randn(dims[i], dims[i+1])
            self.params['b' + str(i+1)] = np.zeros(dims[i+1])

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        cache = []
        H = X
        reg_loss = 0
        
        for i in range(self.num_layers - 1):
            if self.activation == 'sigmoid':
                H, new_cache = affine_sigmoid_forward(H, self.params['W' + str(i+1)], self.params['b' + str(i+1)])
            elif self.activation == 'relu':
                H, new_cache = affine_relu_forward(H, self.params['W' + str(i+1)], self.params['b' + str(i+1)])
            cache.append(new_cache) 
            reg_loss += np.sum(self.params['W' + str(i+1)] ** 2)
            
        scores, new_cache = affine_forward(H, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        cache.append(new_cache)
        reg_loss += np.sum(self.params['W' + str(self.num_layers)] ** 2)

        if mode == 'test':
            return scores

        loss, grads, upgrad = 0.0, {}, 0.0
        if self.loss_fn == 'softmax':
            loss, up_grad = softmax_loss(scores, y)
        elif self.loss_fn == 'svm':
            loss, up_grad = svm_loss(scores, y)
        
        loss += 0.5 * self.reg * reg_loss
        
        up_grad, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] = affine_backward(up_grad, cache.pop())
        grads['W' + str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]
        
        for i in range(self.num_layers - 1, 0, -1):
            if self.activation == 'sigmoid':
                up_grad, grads['W' + str(i)], grads['b' + str(i)] = affine_sigmoid_backward(up_grad, cache.pop())
            elif self.activation == 'relu':
                up_grad, grads['W' + str(i)], grads['b' + str(i)] = affine_relu_backward(up_grad, cache.pop())
            grads['W' + str(i)] += self.reg * self.params['W' + str(i)]

        return loss, grads
    
    
    def find_output(self, x):
        output = {}
        output[0] = x.reshape(1, -1)
        for i in range(self.num_layers - 1):
            if self.activation == 'sigmoid':
                output[i+1], _ = affine_sigmoid_forward(output[i], self.params['W' + str(i+1)], self.params['b' + str(i+1)])
            elif self.activation == 'relu':
                output[i+1], _ = affine_relu_forward(output[i], self.params['W' + str(i+1)], self.params['b' + str(i+1)])
        
        output[self.num_layers], _ = affine_forward(output[self.num_layers-1], 
                                                 self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        for i in range(self.num_layers + 1):
            output[i] = output[i][0]
        return output
