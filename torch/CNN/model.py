import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms

from data_loader import train_loader, valid_loader
class Classifier(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(Classifier, self).__init__()
        # Input Image Dimension ( 28, 28 )
        conv1 = nn.Conv2d( in_channels=1,
                            out_channels=6, 
                            kernel_size=5, 
                            stride=1) # 6 @ 24 * 24
        # activation Relu
        pool1 = nn.MaxPool2d(2) # 6 @ 12 * 12
        conv2 = nn.Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            stride=1)  # 16 @ 8 * 8
        pool2 = nn.MaxPool2d(kernel_size=2) # 16 @ 4 * 4

        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            pool1,
            conv2,
            nn.ReLU(),
            pool2
        )

        fc1 = nn.Linear(in_features=16*4*4,
                        out_features=120)
        # activation Relu
        fc2 = nn.Linear(in_features=120,
                        out_features=84)
        # activation Relu
        fc3 = nn.Linear(in_features=84,
                        out_features=10)
        
        self.fc_module = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )

    def forward(self, x):
        out = self.conv_module(x) # 16 * 4 * 4
        # make linear
        dim = 1
        for d in out.size()[1:]: # 16 * 4 * 4
            dim = dim * d
        out = out.view(-1, dim) 
        out = self.fc_module(out)
        return F.softmax(out, dim=1)
