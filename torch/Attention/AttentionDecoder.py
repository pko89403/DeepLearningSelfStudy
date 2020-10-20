import torch
from torch import nn
import torch.nn.functional as func
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.attn = nn.Linear(hidden_size + output_size, 1)
        # attn layer is used to calculate the value of e, which is small neural network
        self.lstm = nn.LSTM(hidden_size + vocab_size, output_size)
        self.final = nn.Linear(output_size, vocab_size)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.output_size),
            torch.zeros(1, 1, self.output_size))

    def forward(self, decorder_hidden, encoder_outputs, input):
    # takes the decorder's previous hidden state, encoder_outputs and the previous word outputted.
    
        weights = []
        # 'weights' list is used to store the attention weights. 

        for i in range(len(encoder_outputs)):
            # calculate attention weight for each encoder output, pass them through the attention layer.

            print(decorder_hidden[0][0].shape)
            print(encoder_outputs[0].shape)
            weights.append(self.attn(torch.cat((decorder_hidden[0][0],
                                                encoder_outputs[i]), dim = 1)))
            # attn layer calcuates the importance of that word, by using previous decoder hidden state
            # and the hidden state of the encoder at that particular time step.

        normalized_weights = func.softmax(torch.cat(weights, 1), 1)
        # scale weights in range(0, 1) by applying softmax activation

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                                 encoder_outputs.view(1, -1, self.hidden_size))
        # use batch matrix multiplication to multipy attention vector of size 
        #   attention vector size (1, 1, len(encoder_outputs))
        #   encoder_outputs of size (1, len(encoder_outputs), hidden_size)
        # obtaining the size of vector hidden_size is the weighted sum.

        input_lstm = torch.cat((attn_applied[0], input[0]), dim = 1) # you can use embedding of input change this part.
        # concatenation of vector obtained by having a weighted sum according to attention weights and the previous word outputeed

        output, hidden = self.lstm(input_lstm.unsqueeze(0), decorder_hidden)
        # lstm layer takes in concatenation of the vector
        

        output = self.final(output[0])
        # the final layter is added to map the output features space into the size of vocabulary,
        # and also add some non-linearity while outputting the word.
        return output, hidden, normalized_weights 