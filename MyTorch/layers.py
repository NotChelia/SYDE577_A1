from .tensor import Tensor
import numpy as np

class Linear:
    def __init__(self, in_features, out_features, weight=None, bias=None):
        if weight is None:
            weight = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        if bias is None:
            bias = np.zeros(out_features)
        self.weight = Tensor(weight.T, requires_grad=True) #transpose to match the shape
        self.bias = Tensor(bias, requires_grad=True)

    def __call__(self, x):
        return x @ self.weight + self.bias

    @property
    def parameters(self):
        return [self.weight, self.bias]

def relu(x):
    return x.relu()

def sigmoid(x):
    out = Tensor(1 / (1 + np.exp(-x.data)), requires_grad=x.requires_grad)
    out._prev = [x]
    out._op = 'sigmoid'

    def _backward(grad_output):
        if x.requires_grad:
            sigmoid_grad = out.data * (1 - out.data)
            grad_self = grad_output * sigmoid_grad
            grad_self = x.reduce_grad(grad_self)
            if x.grad is None:
                x.grad = grad_self
            else:
                x.grad += grad_self

    out._backward = _backward
    return out