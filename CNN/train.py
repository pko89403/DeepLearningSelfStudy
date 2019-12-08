import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms

from data_loader import train_loader, valid_loader
from model import Classifier

# cnn classifier
cnn = Classifier()
# loss 
criterion = nn.CrossEntropyLoss()
# backpropagation method
learning_rate = 1e-3
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# hyper-parameters
num_epochs = 2
num_batches = len(train_loader)

train_loss_list = []
valid_loss_list = []

for epoch in range(num_epochs):
    train_loss = 0.0
    
    for i, data in enumerate(train_loader): # returns ( Index, List[Index] )
        x, label = data
        # grad init
        optimizer.zero_grad()
        # forward propagation
        model_output = cnn( x )
        # calculate loss
        loss = criterion(model_output, label)
        # back propagation
        loss.backward()
        # weight update
        optimizer.step()

        # train_loss summary
        train_loss += loss.item()
        # del (memory issue)
        del loss
        del model_output

        # printout training steps
        if ( i + 1 ) % 100 == 0 : # every 100 mini-batches
            with torch.no_grad(): # very Important!
                val_losses = 0.0
                for j, val in enumerate( valid_loader ):
                    val_x, val_label = val
                    val_output = cnn(val_x)
                    val_loss = criterion(val_output, val_label)
                    val_losses += val_loss
            
            print("epoch : {}/{} | step: {}/{}  | train loss : {:.4f} | val loss: {:.4f}".format(
                epoch+1, num_epochs, i+1, num_batches, train_loss / 100, val_loss / len(valid_loader)
            ))

            train_loss_list.append( train_loss / 100 )
            valid_loss_list.append( val_losses / len(valid_loader ))
            train_loss = 0.0