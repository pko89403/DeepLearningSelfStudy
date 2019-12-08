import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class RBM(nn.Module):
    """
    Args:
        n_input_dim (int, optional) : The size of input layer
        n_hidden_dim (int, optional) : The size of hidden layer
        n_gibbs (int, optional) : The numbers of Gibbs sampling
    """

    def __init__(self, n_input_dim=784, n_hidden_dim=128, n_gibbs=1):
        """ Create a RBM """
        super(RBM, self).__init__()
        self.input = nn.Parameter(torch.randn(1, n_input_dim)) # input bias
        self.hidden = nn.Parameter(torch.randn(1, n_hidden_dim)) # hidden bias
        self.weight = nn.Parameter(torch.randn(n_hidden_dim, n_input_dim))
        self.gibbs = n_gibbs

    def input_to_hidden(self, input):
        """ conditional sampling a hidden variable given a visible variable.

        Args : input ( Tensor ) : Input Variable.
        
        bernoulli(prob=None, logits=None, validate_args=None) 
        - from torch import distributions.bernoulli
        - Samples are binary ( 0 or 1 )
        -- take the value '1' with probability 'p' and '0' with probability '1-p' 

        linear(input, wegith, bias)
        - from torch import nn.functional.linear

        Returns : Tensor : Hidden Variable.
        """
        p = torch.sigmoid(F.linear(input, self.weight, self.hidden))
        return p.bernoulli()
    
    def hidden_to_input(self, hidden):
        """ conditional sampling a input variable given a hidden variable.

        Args : hidden ( Tensor ) : Hidden Variable.

        Returns : Tensor : Input Variable.
        """
        # tensor.t() -> transpose()
        p = torch.sigmoid(F.linear(hidden, self.weight.t(), self.input))
        return p.bernoulli()
    
    def free_energy(self, input):
        """ Free Energy Function
        ... math::
            \ begin{align}
                F(x) &= -\log \sum_h \exp (-E(x, h)) \\
                &= -a^\top x - \sum_j \log (1 + \exp(weight^{\top}_jx + b_j))\,.
            \end{align}
        
        Args:
            input ( Tensor ) : Input Variable.
        
        Returns: FloatTensor : The free energy value. 
        """
        input_term = torch.matmul(input, self.input.t())
        weight_x_hidden = F.linear(input, self.weight, self.hidden)
        hidden_term = torch.sum(F.softplus(weight_x_hidden), dim=1)
        return torch.mean(- hidden_term - input_term )
    
    def forward(self, input):
        """ compute the readl and generated examples.
        
        Args : input ( Tensor ) : Input Variable.

        Returns : ( Tensor, Tensor ) : Real & Generated Variables.
        """
        hidden = self.input_to_hidden(input)
        for _ in range(self.gibbs):
            input_gibbs = self.hidden_to_input(hidden)
            hidden = self.input_to_hidden(input_gibbs)
        return input, input_gibbs
