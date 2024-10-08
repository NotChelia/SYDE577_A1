from .layers import Linear, relu, sigmoid

class Network:
    def __init__(self, initial_params=None, task='regression'):
        if initial_params is not None:
            self.layer1 = Linear(2, 10, weight=initial_params['weights1'], bias=initial_params['biases1'])
            self.layer2 = Linear(10, 10, weight=initial_params['weights2'], bias=initial_params['biases2'])
            self.layer3 = Linear(10, 1, weight=initial_params['weights3'], bias=initial_params['biases3'])
        else:
            self.layer1 = Linear(2, 10)
            self.layer2 = Linear(10, 10)
            self.layer3 = Linear(10, 1)
        
        self.task = task

    def forward(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = self.layer3(x)
        if self.task == 'binary_classification':
            x = sigmoid(x)
        return x

    @property
    def parameters(self):
        return self.layer1.parameters + self.layer2.parameters + self.layer3.parameters

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()