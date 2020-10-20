import torch
import numpy as np 
import torchvision
import os

from torch import nn 
from torch import optim 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms 
from matplotlib import pyplot as plt

def sample_show():
    # get a batch of training data
    inp, classes = next(iter(train_loader))
    title = [ train_data.classes[i] for i in classes]
    # Make a grid from batch
    inp = torchvision.utils.make_grid(inp, nrow=8)
     
#
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inpt = std * inp + mean
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(3)

# HyperParameter
USE_CUDA = torch.cuda.is_available()
BATCH_SIZE = 6

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=300),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(size=300) ,
    transforms.CenterCrop(size=300),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("/Users/amore/Pytorch_SelfStudy/TransferLearning/hymenoptera_data/train/", train_transform)
test_data = datasets.ImageFolder("/Users/amore/Pytorch_SelfStudy/TransferLearning/hymenoptera_data/val", test_transform)

train_loader = DataLoader(dataset = train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=4)
test_loader = DataLoader(dataset = test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=4)

# sample_show()