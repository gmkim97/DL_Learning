"""
### CNN : MNIST application example (Pytorch)

1. Import proper libraries
2. Variable initialization
3. Define layers
    (1) Convection Layer
    (2) Pooling Layer
    (3) Fully Connected Layer
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def displayMNIST(images, labels, num):
    for i in range(num):
        plt.subplot(1,num, i + 1)
        plt.imshow(images[i].reshape(28,28), cmap="gray")
        plt.title(labels[i].item())
        plt.axis("off")      

    plt.show()

# Check whether GPU Acceleration is possible
## For MacOS (mps)

able_gpu = torch.backends.mps.is_available()
if able_gpu :
    device = torch.device('mps')
    print("MPS Available")
else :
    device = torch.device('cpu')
    print("MPS NOT Available")

# Download MNIST dataset

path = "./dataset"

train_data = datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(),download=True)
test_data = datasets.MNIST(root=path, train=False, transform=transforms.ToTensor(),download=True)

train_loader = DataLoader(train_data, batch_size=100, shuffle=None)

images, labels = next(iter(train_loader))

displayMNIST(images, labels, 10)


# In-progress
