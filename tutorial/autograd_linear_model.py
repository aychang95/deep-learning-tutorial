# Example of three-layer network using pytorch tensors along with autograd

import torch

device = torch.device("cpu")
device = torch.device("cuda")

# Specify the size of your batch, data, hidden layers, and labels
N = 128  # Batch size: How many data points you feed in at a time
data_dim = 1000  # Dimension of data vector, remember one data point is one dimensional
H_1 = 2000  # Dimension of first hidden layer
H_2 = 100  # Dimension of second hidden layer
label_dim = 10  # Dimension of label, the output/answer corresponding to your initial data of dim of 1000
learning_rate = 1e-5

# Set data
x = torch.randn(N, data_dim, device=device)
y = torch.randn(N, label_dim, device=device)


def initialize_uniform_weights(weights, in_dim):
    k_sqrt = in_dim ** (-.5)
    return weights * (2 * k_sqrt) - k_sqrt


# Set weights
w_1 = torch.rand(data_dim, H_1, device=device, requires_grad=True)  # Shape: 128 X 2000
w_2 = torch.rand(H_1, H_2, device=device, requires_grad=True)  # Shape: 2000 X 100
w_3 = torch.rand(H_2, label_dim, device=device, requires_grad=True)  # Shape: 100 X 10

with torch.no_grad():
    w_1 = initialize_uniform_weights(w_1, data_dim)
    w_2 = initialize_uniform_weights(w_2, H_1)
    w_3 = initialize_uniform_weights(w_3, H_2)

w_1.requires_grad = True
w_2.requires_grad = True
w_3.requires_grad = True


# Start Training
for i in range(1000):
    # Start forward pass
    y_pred = x.mm(w_1).clamp(min=0).mm(w_2).clamp(min=0).mm(w_3)

    # Compute loss
    loss = (y_pred - y).pow(2).sum()
    print(i, loss.item())

    # Use autograd to compute pass
    loss.backward()

    with torch.no_grad():
        w_1 -= learning_rate * w_1.grad
        w_2 -= learning_rate * w_2.grad
        w_3 -= learning_rate * w_3.grad

        # Manually zero gradients after running backward pass
        w_1.grad.zero_()
        w_2.grad.zero_()
        w_3.grad.zero_()

print(y_pred[0])
print(y[0])
