import torch as T
import torch.nn as nn
from torch.nn.modules.rnn import RNN

class  LSTM(nn.Module):
    def __init__(self, in_dims, hid_dims, n_layers, out_dims):
        super().__init__()

        self.in_dims = in_dims
        self.hid_dims = hid_dims
        self.n_layers = n_layers
        self.out_dims = out_dims
        self.drop_prob=0.5
        self.lr=0.001
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        self.LSTM = nn.LSTM(self.in_dims, self.hid_dims, n_layers,
                            dropout=self.drop_prob, batch_first=True)

        self.dropout = nn.Dropout(self.drop_prob)

        self.fc = nn.Linear(self.hid_dims, self.out_dims)

    def forward(self, x, hidden):
        
        out, hidden = self.LSTM(x, hidden)

        out = self.dropout(out)

        out = out.contiguous().view(-1, self.hid_dims)

        out = self.fc(out)

        return out, hidden


    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hid_dims).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hid_dims).zero_().to(self.device))
                

        return hidden
        
