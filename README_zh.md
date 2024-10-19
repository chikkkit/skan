# SKAN: 单参数KAN网络

## 简介
SKAN是一种创新的KAN网络，其核心特性在于每个基函数仅依赖于一个可学习的参数，在[1]中的提出。这种设计使得SKAN在保持参数数量不变的情况下，能够扩展出更大规模的网络，从而更有效地捕捉参数之间复杂的交互变化。本库提供了SKAN的完整代码实现，包括基础SKAN网络的构建、自定义基函数的SKAN网络，以及论文[1]中提到的一系列可学习函数。SKAN库基于PyTorch框架构建，定义的网络继承自PyTorch的`nn.Module`，完全兼容PyTorch的生态（包括CUDA支持）。

SKAN网络也是EKE Principle（高效的KAN扩展原理）的一个理想示例。EKE Principle强调，在KAN网络中，通过增加参数而非复杂化基函数，可以更有效地提升网络性能。

## 使用指南

### 安装SKAN
你可以通过PyPI轻松安装SKAN库：
```bash
pip install single-kan
```

### 构建基本网络
以下是SKAN网络的基本使用示例，展示了如何构建一个用于MNIST手写数字分类的SKAN网络：
```python
import torch
from skan import SKANNetwork

# 选择设备，优先使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 构建SKAN网络，输入层784个节点，隐藏层100个节点，输出层10个节点
net = SKANNetwork([784, 100, 10]).to(device)
```
如果未指定`basis_function`，则默认使用`lshifted_softplus`函数，这是论文测试中表现最佳的基函数。如果设备支持GPU并且已安装相关驱动，上述代码中的网络将在GPU上进行运算。

### 使用预设基函数
SKAN库提供了多种预设的单参数可学习函数，这些函数在论文中有所提及。以下是如何使用预置的单参数可学习函数的示例：
```python
import torch
from skan import SKANNetwork
from skan.basis import lrelu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 使用lrelu作为基函数构建SKAN网络
net = SKANNetwork([784, 100, 10], basis_function=lrelu).to(device)
```

### 自定义SKAN网络
SKAN库支持用户自定义基函数。以下是一个自定义基函数的示例：
```python
import torch
import numpy as np

# 定义自定义基函数
def lshifted_softplus(x, k):
    return torch.log(1 + torch.exp(k*x)) - np.log(2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 使用自定义基函数构建SKAN网络
net = SKANNetwork([784, 100, 10], basis_function=lshifted_softplus).to(device)
```
自定义基函数需要接收两个参数：输入值`x`和唯一的可学习参数`k`(保持这个参数顺序)。请确保基函数支持NumPy的广播运算，并仅使用基于NumPy构建的库（如PyTorch）。

### 参考文献
[1] LSS-SKAN: Efficient Kolmogorov–Arnold Networks based on Single-Parameterized Function(submited to arxiv)
