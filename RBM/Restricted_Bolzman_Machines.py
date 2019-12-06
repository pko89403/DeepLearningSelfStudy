# Restricted Boltzmann Machines
# In order to increase the expressive power of the model
# We introduce some non-observed variables.
# https://github.com/bacnguyencong/rbm-pytorch/blob/master/Notebook.ipynb

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F 
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import RBM
from model_libs import train, show_and_save

# Hyper-Parameter Settings
batch_size = 64
n_epochs = 10
learning_rate = 0.01
n_input_dim = 784
n_hidden_dim = 128

# Create a RBM model
model = RBM(n_input_dim=n_input_dim,
            n_hidden_dim=n_hidden_dim,
            n_gibbs=1)

# Prepare the Dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./output',
                    train=True,
                    download=True,
                    transform=transforms.Compose([ transforms.ToTensor()
                    ])),
    batch_size=batch_size
)

# Then train the model
model = train(model, train_loader, n_epochs=n_epochs, lr=learning_rate)

images = next(iter(train_loader))[0]
input, input_gibbs = model(images.view(-1, 784))

# show the real images
show_and_save(make_grid(input.view(batch_size, 1, 28, 28).data), 'output/real')

# show the generated images
show_and_save(make_grid(input_gibbs.view(batch_size, 1, 28, 28).data), 'output/fake')

# How one image is factorized through the hidden variables
n_sample = 4
kth = 18
d = images[kth:kth+1]

V = torch.sigmoid(F.linear(d.view(1, -1), model.weight, model.hidden))
v, o = torch.sort(V.view(-1))

fig, ax = plt.subplots(1, n_sample + 1, figsize=(3*(1 + n_sample),3))
ax[0].imshow(d.view(28, 28).numpy(), cmap='gray')
ax[0].set_title('Original image')

for k, i in enumerate(o[-n_sample:].numpy()):
    f = model.weight[i].view(28, 28).data.numpy()
    ax[k + 1].imshow(f, cmap='gray')
    ax[k + 1].set_title('p=%.2f'% V[0][i].item())
    
plt.savefig('output/factor.png', dpi=200)