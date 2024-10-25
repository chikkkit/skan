# SKAN: Single-Parameterized KAN Network

<p align="center"><b>English</b> / <a href="https://github.com/chikkkit/SKAN/blob/main/README_zh.md">简体中文</a></p>

## Introduction
SKAN is an innovative KAN (Kolmogorov-Arnold Network) network, characterized by its core feature where each basis function depends on only one learnable parameter, as proposed in [`this paper`](https://arxiv.org/abs/2410.14951) [1]. This design enables SKAN to scale up to larger networks while maintaining the same number of parameters, thereby more effectively capturing complex interactions between parameters. This repository provides a complete code implementation of SKAN, including the construction of basic SKAN networks, SKAN networks with custom basis functions, and a series of learnable functions mentioned in paper [1]. The SKAN library is built on the PyTorch framework, with defined networks inheriting from PyTorch's `nn.Module`, ensuring full compatibility with the PyTorch ecosystem (including CUDA support).

The SKAN network also serves as an ideal example of the EKE Principle (Efficient KAN Extension Principle). The EKE Principle emphasizes that in KAN networks, network performance can be more effectively enhanced by increasing parameters rather than complicating basis functions.

## Usage Guide

### Installing SKAN
SKAN can be easily installed via PyPI using the following command:
```bash
pip install single-kan
```

### Building Basic Networks
Below is a basic example of using the SKAN network, demonstrating how to construct a SKAN network for MNIST handwritten digit classification:
```python
import torch
from skan import SKANNetwork

# Select device, prioritizing GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Construct SKAN network with 784 input nodes, 100 hidden nodes, and 10 output nodes
net = SKANNetwork([784, 100, 10]).to(device)
```
If `basis_function` is not specified, the `lshifted_softplus` function is used by default, which performed best in the tests of paper [1]. If the device supports GPU and the relevant drivers are installed, the network in the above code will perform computations on the GPU.

### Using Preset Basis Functions
The SKAN library provides multiple preset single-parameter learnable functions, which are mentioned in the paper. Here's an example of how to use a preset single-parameter learnable function:
```python
import torch
from skan import SKANNetwork
from skan.basis import lrelu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Construct SKAN network using lrelu as the basis function
net = SKANNetwork([784, 100, 10], basis_function=lrelu).to(device)
```

### Customizing SKAN Networks
The SKAN library supports user-defined basis functions. Here's an example of a custom basis function:
```python
import torch
import numpy as np
from skan import SKANNetwork

# Define custom basis function
def lshifted_softplus(x, k):
    return torch.log(1 + torch.exp(k*x)) - np.log(2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Construct SKAN network using the custom basis function
net = SKANNetwork([784, 100, 10], basis_function=lshifted_softplus).to(device)
```
Custom basis functions should accept two parameters: the input value `x` and a unique learnable parameter `k` (keep this order of parameters). It is important to ensure that the basis function supports NumPy broadcasting operations and only uses libraries built on NumPy (such as PyTorch).

### Reference
[1] Chen Z, Zhang X. LSS-SKAN: Efficient Kolmogorov-Arnold Networks based on Single-Parameterized Function[J]. arXiv preprint arXiv:2410.14951, 2024.
