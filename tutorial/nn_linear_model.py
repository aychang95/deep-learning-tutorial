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

model = torch.nn.Sequential(
    torch.nn.Linear(data_dim, H_1),
    torch.nn.ReLU(),
    torch.nn.Linear(H_1, H_2),
    torch.nn.ReLU(),
    torch.nn.Linear(H_2, label_dim)
).to(device)

loss_fn = torch.nn.MSELoss(reduction="sum")

for i in range(2000):
    y_pred = model(x)


    loss = loss_fn(y_pred, y)
    print(i, loss.item())
    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param.data -= learning_rate*param.grad

print(y[0])
print(y_pred[0])