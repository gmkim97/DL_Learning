"""
### CNN : MNIST application example (Pytorch)

1. Checking device
2. Setting datasets and data loader
3. Modeling NN
4. Defining loss(cost) function and optimizer
5. Training : forward + backward + optimization
6. Testing
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def displayMNIST(data_loader, num):
    images, labels = next(iter(data_loader))

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
        print("MPS Available\n")
    elif able_gpu :
        device = torch.device('cuda:0')
        print("GPU CUDA Available\n")
    else :
        device = torch.device('cpu')
        print("MPS NOT Available\n")

    return device

def trainMNIST(data_loader, optimizer, criterion, epochs, device):
    
    cnn.train()

    avg_loss = 0
    
    for index, data in enumerate(data_loader): ## <enumerate func.> returns (1) the index, and (2) the components of collection in tuple.
    
        inputs, labels = data[0].to(device), data[1].to(device) ## MNIST data in data_loader consists of (input) image data with each labels

        optimizer.zero_grad() ## Initializing the gradient

        hypothesis = cnn(inputs) ## Forward
        loss = criterion(hypothesis, labels)
        loss.backward() ## Backward

        optimizer.step() ## Optimizations

        avg_loss += loss / len(data_loader)
    
    print("Completed training for Epoch : {}".format(epochs+1))
    print("[Epoch : {}] Average Loss : {:.6f}\n".format(epochs+1, avg_loss))

def testMNIST(test_data, device):

    cnn.eval()

    with torch.no_grad():
        X_test = test_data.test_data.view(len(test_data), 1, 28,28).float().to(device)
        Y_test = test_data.test_labels.to(device)

        hypothesis = cnn(X_test)
        correct_hypo = torch.argmax(hypothesis,1) == Y_test
        accuracy = correct_hypo.float().mean()
        print("Accuracy : {:.6f}".format(accuracy.item()))

# Define a Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential( ## One layer = Conv. layer + Pooling layer
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), ## ReLU Activation function
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        )

        self.fc1 = nn.Linear(in_features=4*4*128, out_features=625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight) ## Weight initialization to improve the learning process.
        
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=0.5)  ## Using Dropout Regularization at hidden layer of FC for avoiding overfit, or improving the process.
        )

        self.fc2 = nn.Linear(in_features=625, out_features=10, bias=True) ## Since cost function already has Softmax, don't need to activate in here.
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):
        out_layer1 = self.layer1(x)
        out_layer2 = self.layer2(out_layer1)
        out_layer3 = self.layer3(out_layer2)
        out_flatten = torch.flatten(out_layer3, 1) ## Before going through fc layers, should flatten the output.
        out_layer4 = self.layer4(out_flatten)
        out = self.fc2(out_layer4)

        return out

# Download MNIST dataset

path = "./dataset"
device = checkGPU()
learn_rate = 0.001
batch_size = 100
total_epochs = 15

train_data = datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(),download=True)
test_data = datasets.MNIST(root=path, train=False, transform=transforms.ToTensor(),download=True)

print("Total numbers of training data : {}\n".format(len(train_data)))

data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

print("Batch size is {}, then numbers of batches : {}\n".format(batch_size, int(len(train_data) / batch_size)))

# Modeling NN

cnn = CNN().to(device)

# Define a Loss function and Optimizer

criterion = nn.CrossEntropyLoss().to(device) ## CrossEntropyLoss() cost function includes Softmax activation function.
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=learn_rate) ## Adam() optimizer

# Train the network

for epochs in range(total_epochs):
    trainMNIST(data_loader, optimizer, criterion, epochs, device)


# Test

testMNIST(test_data, device)
