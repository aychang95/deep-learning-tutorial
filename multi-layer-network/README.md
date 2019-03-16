### Multi-Layer Network
This tutorial is based on the construction of a three-layer network that learns to classify a 1000 dimension data point as a
dimension label.  This tutorial will detail the following:
- A raw implementation of three-layer network, in this case all fully connected layers, using just numpy
- Manual gradient computation for backpropogation with gradient descent
- Weight initialization techniques to counter vanishing gradient issues
- Use of Pytorch tensors and modules for very building and training


## Numpy Implementation

```python
# Code in file numpy_linear_model.py
# Example of three-layer network using only numpy

import numpy as np

# Specify the size of your batch, data, hidden layers, and labels
N = 128  # Batch size: How many data points you feed in at a time
data_dim = 1000  # Dimension of data vector, remember one data point is one dimensional
H_1 = 100 # Dimension of first hidden layer
H_2 = 100  # Dimension of second hidden layer
label_dim = 10  # Dimension of label, the output/answer corresponding to your initial data of dim of 1000
learning_rate = 1e-6

# Create dummy data and labels
x = np.random.randn(N, data_dim)  # Our data in the shape of a 128 X 1000 tensor
y = np.random.randn(N, label_dim)  # Our corresponding labels in the shape of a 128 X 10 tensor

# Now we initialize our weights from a standard normal distribution (dummy data is also from SND but that is irrelevant)
w_1 = np.random.randn(data_dim, H_1)  # First layer in the shape of a 1000 X 2000 tensor
w_2 = np.random.randn(H_1, H_2)  # Second layer in the shape of a 2000 X 100 tensor
w_3 = np.random.randn(H_2, label_dim)  # Third layer in the shape of a 100 X 10 tensor

# On to the training
for i in range(1000):
    # Start with the forward pass
    h_1 = x.dot(w_1)  # MatMul between data and weights of the first layer with shape 128 X 2000
    h_1_relu = np.maximum(h_1, 0)  # Non-linear ReLU layer
    h_2 = h_1_relu.dot(w_2)  # Matmul between first hidden layer and second weight layer with shape 128 X 100
    h_2_relu = np.maximum(h_2, 0)  # Non-linear ReLU layer
    y_pred = h_2_relu.dot(w_3)  # Matmul between second hidden layer and third weight layer with shape 128 X 10


    # Use a loss function to see how well it did (in this case we use the residual sum of squares)
    loss = (np.square(y_pred - y).sum())  # This is a scalar representing our loss score...lower the better. shape: scalar
    print(f"Loss is: {loss}")
    print(f"Step is : {i}")

    # Time to backpropagate which will compute the gradients for our weights
    y_pred_gradient = (y_pred - y)  # Find derivative of loss in respect to y_pred: dloss/dy_pred with shape 128 X 10
    w_3_gradient = h_2_relu.T.dot(y_pred_gradient)  # Find derivative of y_pred in respect to w_3 and apply chain rule with shape 100 X 10

    h_2_relu_gradient = y_pred_gradient.dot(w_3.T)  # Find derivative of h_2_relu in respect to w_2 with shape 128 X 100
    h_2_relu_gradient[h_2 < 0] = 0  # Adjust to derivative of ReLU.  shape: 128 X 100
    w_2_gradient = h_1_relu.T.dot(h_2_relu_gradient)  # Chain rule applied to the derivative of h_2_relu with shape 2000 X 100

    h_2_relu_gradient = y_pred_gradient.dot(w_3.T)  # dloss/dy shape 128 X 10
    h_2_relu_gradient[h_2 < 0] = 0  # Adjust to derivative of ReLU. shape: 128 X 100
    h_1_relu_gradient = h_2_relu_gradient.dot(w_2.T)  # dloss/dy * dy/dh_2 * dh_2/dh_1 with shape 128 X 2000
    h_1_relu_gradient[h_1 < 0] = 0  # Adjust to derivative of ReLU. shape: 128 X 2000
    w_1_gradient = x.T.dot(h_1_relu_gradient)  # Chain rule applied all the way to the end with shape 1000 X 2000

    # Update weights at specified rate
    w_1 -= learning_rate*w_1_gradient
    w_2 -= learning_rate*w_2_gradient
    w_3 -= learning_rate*w_3_gradient


def forward_pass():
    h_1 = x.dot(w_1)
    h_1_relu = np.maximum(h_1, 0)
    h_2 = h_1_relu.dot(w_2)
    h_2_relu = np.maximum(h_2, 0)
    y_pred = h_2_relu.dot(w_3)
    return y_pred


print(f"The original label {y[0]}")
print(f"The learned prediction {forward_pass()[0]}")







```