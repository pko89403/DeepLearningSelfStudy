import torch
from AttentionDecoder import *
from Encoder import *

bidirectional = True
c = Encoder(10, 20, bidirectional)
a, b = c.forward(torch.randn(10), c.init_hidden())

print(a.shape)
print(b[0].shape)
print(b[1].shape)

x = AttentionDecoder(20 * (1 + bidirectional), 25, 30)
y, z, w = x.forward(x.init_hidden(), torch.cat((a,a)), torch.zeros(1, 1, 30))

print(y.shape)
print(z[0].shape)
print(z[1].shape)
print(w)
