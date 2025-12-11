import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------
# LeNet-5
# -----------------------
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)      # Change to 3 channels if CIFAR-10
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))           # 28->24
        x = F.max_pool2d(x, 2)              # 24->12
        x = F.relu(self.conv2(x))           # 12->8
        x = F.max_pool2d(x, 2)              # 8->4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------
# MNIST loader
# -----------------------
def load_mnist(batch_size=32):
    transform = transforms.ToTensor()
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    testset  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True),
        DataLoader(testset, batch_size=batch_size),
    )
