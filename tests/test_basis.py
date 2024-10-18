import torch
from skan import SKANNetwork
import torchvision
from torch.utils.data import DataLoader
from skan.basis import lrelu

# 选择设备，优先使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 构建SKAN网络，输入层784个节点，隐藏层100个节点，输出层10个节点
net = SKANNetwork([784, 100, 10], basis_function=lrelu).to(device)

# 使用MNIST数据集
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

lr = 0.001

# 选择Adam优化器
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# 选择交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()

# 训练网络
for epoch in range(10):
    net.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))