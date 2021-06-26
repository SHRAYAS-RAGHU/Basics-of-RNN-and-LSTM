import torch as T
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims):
        super().__init__()

        self.h_dims = h_dims
        self.ip_to_h = nn.Linear(in_dims + h_dims, h_dims)
        self.ip_to_op = nn.Linear(in_dims + h_dims, out_dims)
        self.activation = nn.LogSoftmax(dim=1)
    
    def forward(self, inp, hidden):
        inp = T.cat((inp, hidden), dim = 1)

        h = self.ip_to_h(inp)
        out = self.activation(self.ip_to_op(inp))

        return out, h
    
    def init_hidden(self):
        return T.zeros(1, self.h_dims)
