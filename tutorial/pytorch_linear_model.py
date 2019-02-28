# Example of three-layer network using pytorch tensors

import torch

device = torch.device("cpu")
device = torch.device("cuda")

# Specify the size of your batch, data, hidden layers, and labels
N = 128  # Batch size: How many data points you feed in at a time
data_dim = 1000  # Dimension of data vector, remember one data point is one dimensional
H_1 = 2000  # Dimension of first hidden layer
H_2 = 100  # Dimension of second hidden layer
label_dim = 10  # Dimension of label, the output/answer corresponding to your initial data of dim of 1000
learning_rate = 1e-6

#Set data
x = torch.randn(N, data_dim)
y = torch.randn(N, label_dim)

#Set weights
w_1 = torch.randn(data_dim, H_1)  # Shape: 128 X 2000
w_2 = torch.randn(H_1, H_2)  # Shape: 2000 X 100
w_3 = torch.randn(H_2, label_dim)  # Shape: 100 X 10

#Start Training
for i in range(500):
    h_1 = x.mm(w_1)  # Shape: 128 X 2000
    h_1_relu = h_1.clamp(min=0)  # Shape: 128 X 2000
    h_2 = h_1_relu.mm(w_2)  # Shape: 128 X 100
    h_2_relu = h_1_relu.clamp(min=0)  # Shape: 128 X 100
    y_pred = h_2_relu.mm(w_3)  # Shape: 128 X 10

    loss = (y_pred-y).pow(2).sum()
    print(f"This is loss {loss}")
    print(f"This is iter {i}")

    #Backprop
    y_pred_gradient = 2




