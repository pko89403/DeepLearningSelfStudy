import matplotlib.pyplot as plt
import numpy as np 
import torch.optim as optim 

def show_and_save(img, filename):
    """ show and save the image.

    Args : img (Tensor) : The image.
           file_name (String) : The destination.
    """
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % filename
    plt.imshow(npimg, cmap='gray')
    plt.imsave(f, npimg)

def train(model, train_loader, n_epochs, lr=1e-2):
    """ Train a RBM Model.

    Args : model
           train_loader (DataLoader)
           n_epochs (int, optional) : The number of epochs
           lr (Fload, optional) : The learning rate
    
    Returns : The trained model.
    """
    # optimzer
    train_op = optim.Adam(model.parameters(), lr)

    # train the RBM model
    model.train()

    for epoch in range(n_epochs):
        loss_ = []
        for _, (data, target) in enumerate(train_loader):
            # tensor.view() : Resize tensor shape
            input, input_gibbs = model(data.view(-1, 784))
            loss = model.free_energy(input) - model.free_energy(input_gibbs)
            # tensor.item() : Transform tensor to scala
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            # Optimzer's step() : update model parameters.
            train_op.step()
        print(f"Epoch - {epoch} : Loss = {np.mean(loss_)}\t")

    return model