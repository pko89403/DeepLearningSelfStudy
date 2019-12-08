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

from DataLoader import train_loader, test_loader

FINE_TUNE = False  

inception = models.inception_v3(pretrained=True)

# Auxiliary 를 사용하지 않으면 inception v2와 동일
inception.aux_logits = False

# 일단 모든 layers를 requires_grad=False 를 통해서 학습이 안되도록 막는다.
if not FINE_TUNE: 
    for parameter in inception.parameters():
        parameter.requires_grad = False 

# 새로운 fully-connected classifier layer 를 만들어준다. ( requires_grad=True )
# in_features = 2048 : in 으로 들어오는 feature 의 갯수
n_features = inception.fc.in_features
inception.fc = nn.Linear(n_features, 2)

criterion = nn.CrossEntropyLoss()

# Optimizer 에는 requires_grad=True 인 parameters들만 들어갈수 있다.
optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, inception.parameters()), lr=0.001)

def train_model(model, criterion, optimizer, epochs=30):
    for epoch in range(epochs):
        epoch_loss = 0
        for step, (inputs, y_true) in enumerate(train_loader):
            x_sample, y_true = Variable(inputs), Variable(y_true)

            # parameter gradients들은 0으로 초기화 한다.
            optimizer.zero_grad()

            # FeedForward
            y_pred =  inception(x_sample)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            _loss = loss.item()
            epoch_loss += _loss
        
        print(f'[{epoch+1}] loss : {epoch_loss/step:.4}')

def validate(model, epochs=1):
    model.train(False)
    n_total_correct = 0
    for step, (inputs, y_true) in enumerate(test_loader):
        x_sample, y_true = Variable(x_sample), Variable(y_true)

        y_pred = model(x_sample)
        _, y_pred = torch.max(y_pred.data, 1)

        n_correct = torch.sum(y_pred == y_true.data)
        n_total_correct += n_correct
    
    print('Accuracy : ', n_total_correct / len(test_loader.dataset))

train_model(inception, criterion, optimizer)
validate(inception)