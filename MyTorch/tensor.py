import numpy as np

#notes:
# a computational graph representation would be better, introducing a node class
# and seperating the backward into a seperate method makes more sense

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda grad_output: None
        self._prev = []
        self._op = ''

    def reduce_grad(self, grad_output):
        grad = grad_output
        while grad.ndim > self.data.ndim:
            grad = grad.sum(axis=0)
        for axis, size in enumerate(self.data.shape):
            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)
        out._op = 'add'

        def _backward(grad_output):
            if self.requires_grad:
                grad_self = self.reduce_grad(grad_output)
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = other.reduce_grad(grad_output)
                if other.grad is None:
                    other.grad = grad_other
                else:
                    other.grad += grad_other

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)
        out._op = 'mul'

        def _backward(grad_output):
            if grad_output is None:
                grad_output = np.ones_like(out.data)

            grad_self = other.data * grad_output
            grad_other = self.data * grad_output

            if self.requires_grad:
                grad_self = self.reduce_grad(grad_self)
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = other.reduce_grad(grad_other)
                if other.grad is None:
                    other.grad = grad_other
                else:
                    other.grad += grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = (self, other)
        out._op = 'matmul'

        def _backward(grad_output):
            if self.requires_grad:
                grad_self = grad_output @ other.data.T
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self
            if other.requires_grad:
                grad_other = self.data.T @ grad_output
                if other.grad is None:
                    other.grad = grad_other
                else:
                    other.grad += grad_other

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        out._prev = [self]
        out._op = 'relu'

        def _backward(grad_output):
            if self.requires_grad:
                grad_self = grad_output * (self.data > 0)
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        out._prev = [self]
        out._op = 'sum'

        def _backward(grad_output):
            if self.requires_grad:
                grad = grad_output * np.ones_like(self.data)
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(np.mean(self.data), requires_grad=self.requires_grad)
        out._prev = [self]
        out._op = 'mean'

        def _backward(grad_output):
            if self.requires_grad:
                grad = grad_output * np.ones_like(self.data) / self.data.size
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        return out

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)

        self.grad = grad

        topo_order = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for prev in tensor._prev:
                    build_topo(prev)
                topo_order.append(tensor)

        build_topo(self)

        for tensor in reversed(topo_order):
            tensor._backward(tensor.grad)

    def zero_grad(self):
        self.grad = None