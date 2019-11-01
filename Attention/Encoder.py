import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional = True):
        super(Encoder, self).__init__()
        # Store parameters
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        # Create LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional)

    def forward(self, inputs, hidden):
        output, hidden = self.lstm(inputs.view(1, 1, self.input_size), hidden)
        return output, hidden

    def init_hidden(self):
        # To be called before passing sentence through the LSTM to initialize the hidden state
        # hidden state has to be two vectors, as LSTMs have two vectors ( Hidden Activation + Memory Cell )
        # 1st dim of hidden state is 2 for bidirectional LSTM
        # 2nd dim is the batch size
        return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
            torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))
