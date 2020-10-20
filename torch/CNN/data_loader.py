import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms 
from mnist_data import train_dataset, valid_datset

batch_size = 64

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_datset,
                                                batch_size=batch_size,
                                                shuffle=True)
