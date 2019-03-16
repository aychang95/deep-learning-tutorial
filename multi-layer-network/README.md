### Multi-Layer Network
This tutorial is based on the construction of a three-layer network that learns to classify a 1000 dimension data point as a
dimension label.  This tutorial will detail the following:
- A raw implementation of three-layer network, in this case all fully connected layers, using just numpy
- Manual gradient computation for backpropogation with gradient descent
- Weight initialization techniques to counter vanishing gradient issues
- Use of Pytorch tensors and modules for very building and training


## Numpy Implementation

```python
:INCLUDE multi-layer-network/numpy_linear_model.py
```
