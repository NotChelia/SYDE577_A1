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
