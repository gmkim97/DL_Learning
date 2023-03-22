"""
### CNN : MNIST application example (Pytorch)

1. Import proper libraries
2. Variable initialization
3. Define layers
    (1) Convection Layer
    (2) Pooling Layer
    (3) Fully Connected Layer
"""

"""
### Given Architecture of CNN
# 1번 레이어 : 합성곱층(Convolutional layer)
합성곱(in_channel = 1, out_channel = 32, kernel_size=3, stride=1, padding=1) + 활성화 함수 ReLU
맥스풀링(kernel_size=2, stride=2))

# 2번 레이어 : 합성곱층(Convolutional layer)
합성곱(in_channel = 32, out_channel = 64, kernel_size=3, stride=1, padding=1) + 활성화 함수 ReLU
맥스풀링(kernel_size=2, stride=2))

# 3번 레이어 : 합성곱층(Convolutional layer)
합성곱(in_channel = 64, out_channel = 128, kernel_size=3, stride=1, padding=1) + 활성화 함수 ReLU
맥스풀링(kernel_size=2, stride=2, padding=1))

# 4번 레이어 : 전결합층(Fully-Connected layer)
특성맵을 펼친다. # batch_size x 4 x 4 x 128 → batch_size x 2048
전결합층(뉴런 625개) + 활성화 함수 ReLU

# 5번 레이어 : 전결합층(Fully-Connected layer)
전결합층(뉴런 10개) + 활성화 함수 Softmax
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def displayMNIST(train_loader, num):
    images, labels = next(iter(train_loader))

    for i in range(num):
        plt.subplot(1,num, i + 1)
        plt.imshow(images[i].reshape(28,28), cmap="gray")
        plt.title(labels[i].item())
        plt.axis("off")      

    plt.show()

def checkGPU():
    # Check whether GPU Acceleration is possible
    ## For MacOS (mps)
    ## For Linux (gpu)
    able_mps = torch.backends.mps.is_available()
    able_gpu = torch.cuda.is_available()
    if able_mps :
        device = torch.device('mps')
        print("MPS Available")
    elif able_gpu :
        device = torch.device('cuda:0')
        print("GPU CUDA Available")
    else :
        device = torch.device('cpu')
        print("MPS NOT Available")

    return device


# Define a Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4*4*128, out_features=625, bias=True),
            nn.ReLU()  
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=625, out_features=10, bias=True),
            nn.ReLU()
        )
        
    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_pool1 = self.pool1(out_conv1)
        out_conv2 = self.conv2(out_pool1)
        out_pool2 = self.pool2(out_conv2)
        out_conv3 = self.conv3(out_pool2)
        out_pool3 = self.pool3(out_conv3)
        out_flatten = torch.flatten(out_pool3, 1)
        out_fc1 = self.fc1(out_flatten)
        out_fc2 = self.fc2(out_fc1)

        return out_fc2

# Download MNIST dataset

path = "./dataset"
device = checkGPU()
learn_rate = 0.001
batch_size = 100
total_epochs = 15
cnn = CNN().to(device)

train_data = datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(),download=True)
test_data = datasets.MNIST(root=path, train=False, transform=transforms.ToTensor(),download=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

# Define a Loss function and Optimizer

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=learn_rate)

# Train the network

for epochs in range(total_epochs):

    loss_avg = 0.0

    for index, data in enumerate(train_loader):
        
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad() ## Initializing the gradient

        hypothesis = cnn(inputs) ## Forward
        loss = criterion(hypothesis, labels)
        loss.backward() ## Backward

        optimizer.step() ## Optimization

        # Displaying the training progress
        loss_avg += loss.item() / len(train_loader)

        if (index % 100) == 99:
            print("\n[Epoch : {}  Index : {}]  Average Loss : {:.6f}\n".format(epochs+1,index+1, loss_avg))
            loss_avg = 0.0

# Test



### In Progress ###