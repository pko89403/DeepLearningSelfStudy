import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms 

train_dataset = datasets.MNIST("./mnist_data/",
                                download=True,
                                train=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(), # Image to Tensor
                                    transforms.Normalize((0.1307,), (0.3081,)) # Image, Label
                                ]))
valid_datset = datasets.MNIST("./mnist_data/",
                                    download=True,
                                    train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081, ))
                                    ]))