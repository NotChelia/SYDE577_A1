MyTorch is a simple autodifferentiation package based on Numpy, with a simple set of features including, custom implementations of gradient computation, ReLU and sigmoid nonlinearities, and a mean squared error (MSE) and binary cross-entropy (BCE) loss function

# Features
**Autodifferentiation Support**

Compute gradients for multi-layer FC NN

**Activation Functions**

- ReLU
- Sigmoid for Binary Classification

**Loss Functions**

- Mean Squared Error (MSE) for regression
- Binary Cross-Entropy (BCE) for binary classification tasks

**Multi-layer FC NN**

- Exmaple network config provided: 2-inputs, 2 hidden layers of 10 neurons and 1 output

# Usage
To use MyTorch import the components as shown below
```Python
from MyTorch import Tensor, Network, mse_loss, binary_cross_entropy_loss, relu, sigmoid
```

# Requirements
```Python
pip install numpy
```
or with the provided Requirements file:
```Python
pip install requirements.txt
```

# Example Network
`training_example.py` is a provided example of how the packaged is used for a network with two inputs, two hidden layers with ten neurons each, and a single output using, weights, biases, inputs, and targets in
the file assignment-one-test-parameters.pkl. Comparing results of the first pass gradient with PyTorch, we can see that:

grad of first layer weight for first sample of MyTorch
```
[[-0.0210035  -0.09259178]
 [-0.0184767  -0.08145263]
 [ 0.          0.        ]
 [ 0.01776593  0.0783193 ]
 [ 0.          0.        ]
 [ 0.          0.        ]
 [-0.00974717 -0.04296942]
 [ 0.          0.        ]
 [ 0.          0.        ]
 [ 0.          0.        ]]
grad of first layer bias for first sample
[-0.21514022 -0.18925803  0.          0.18197762  0.          0.
 -0.09984094  0.          0.          0.        ]
```

vs PyTorch

```
tensor([[-0.0210, -0.0926],
        [-0.0185, -0.0815],
        [ 0.0000,  0.0000],
        [ 0.0178,  0.0783],
        [ 0.0000,  0.0000],
        [ 0.0000,  0.0000],
        [-0.0097, -0.0430],
        [ 0.0000,  0.0000],
        [ 0.0000,  0.0000],
        [ 0.0000,  0.0000]])
tensor([-0.2151, -0.1893,  0.0000,  0.1820,  0.0000,  0.0000, -0.0998,  0.0000,
         0.0000,  0.0000])
```

We can see that our implementation reflects the same functionality. The training code should also demonstrate the following plot of training loss vs epoch

![chart](https://i.imgur.com/ZxbzIr1.png)
