import numpy as np
import pickle
from MyTorch import Tensor, Network, mse_loss, binary_cross_entropy_loss, relu, sigmoid
import matplotlib.pyplot as plt

#set a seed for reproducibility
np.random.seed(18945)

#loadpickle
with open('assignment-one-test-parameters.pkl', 'rb') as f:
    params = pickle.load(f)


initial_weights1 = params['w1']
initial_weights2 = params['w2']
initial_weights3 = params['w3']
initial_bias1 = params['b1']
initial_bias2 = params['b2']
initial_bias3 = params['b3']

inputs = params['inputs']
targets = params['targets'].reshape(-1, 1)


initial_params = {
    'weights1': initial_weights1,
    'biases1': initial_bias1,
    'weights2': initial_weights2,
    'biases2': initial_bias2,
    'weights3': initial_weights3,
    'biases3': initial_bias3
}

task = 'regression'
net = Network(initial_params=initial_params, task=task)

if task == 'regression':
    loss_fn = mse_loss
elif task == 'binary_classification':
    loss_fn = binary_cross_entropy_loss

#first sample compute and print grad
first_input = inputs[0]
first_target = targets[0]

x0 = Tensor(first_input.reshape(1, -1), requires_grad=False)
y0 = Tensor(first_target.reshape(1, -1), requires_grad=False)

y_pred0 = net.forward(x0)
loss0 = mse_loss(y_pred0, y0)

loss0.backward()

print("Gradient of first layer weight for first sample")
print(net.layer1.weight.grad.T)
print("Gradient of first layer bias for first sample")
print(net.layer1.bias.grad)


#training settings as outlined in assignment
learning_rate = 1/100
epochs = 5
loss_history = []

for epoch in range(0, epochs + 1):
    net.zero_grad()
    accum_loss = 0
    accum_grads = [np.zeros_like(p.data) for p in net.parameters]

    for x, y in zip(inputs, targets):
        x_tensor = Tensor(x.reshape(1, -1), requires_grad=False)
        y_tensor = Tensor(y.reshape(1, -1), requires_grad=False)
        y_pred = net.forward(x_tensor)
        loss = mse_loss(y_pred, y_tensor)

        accum_loss += loss.data

        loss.backward()

        for i, p in enumerate(net.parameters):
            if p.grad is not None:
                accum_grads[i] += p.grad
            p.zero_grad()

    avg_grads = [g / len(inputs) for g in accum_grads]

    for p, g in zip(net.parameters, avg_grads):
        p.data -= learning_rate * g

    accum_loss = 0
    for x, y in zip(inputs, targets):
        x_tensor = Tensor(x.reshape(1, -1), requires_grad=False)
        y_tensor = Tensor(y.reshape(1, -1), requires_grad=False)
        y_pred = net.forward(x_tensor)
        loss = mse_loss(y_pred, y_tensor)
        accum_loss += loss.data

    avg_loss = accum_loss / len(inputs)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch}: Ave Loss = {avg_loss}")

plt.figure(figsize=(8, 6))
plt.plot(range(len(loss_history)), loss_history, marker='o', linestyle='-')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.show()