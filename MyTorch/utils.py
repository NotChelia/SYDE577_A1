from .tensor import Tensor
import numpy as np

def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    loss = (diff * diff).mean() * 0.5
    return loss

def binary_cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-12
    y_pred_clipped = Tensor(np.clip(y_pred.data, epsilon, 1 - epsilon), requires_grad=y_pred.requires_grad)
    y_pred_clipped._prev = [y_pred]
    y_pred_clipped._op = 'clip'

    term1 = y_true * np.log(y_pred_clipped.data)
    term2 = (1 - y_true) * np.log(1 - y_pred_clipped.data)
    loss = Tensor(-(term1 + term2).mean(), requires_grad=y_pred.requires_grad)
    loss._prev = [y_pred_clipped]
    loss._op = 'binary_cross_entropy'

    def _backward(grad_output):
        if y_pred.requires_grad:
            grad = -(y_true / y_pred_clipped.data) + (1 - y_true) / (1 - y_pred_clipped.data)
            grad = grad * grad_output / y_true.size
            grad = y_pred.reduce_grad(grad)
            if y_pred.grad is None:
                y_pred.grad = grad
            else:
                y_pred.grad += grad

    loss._backward = _backward
    return loss