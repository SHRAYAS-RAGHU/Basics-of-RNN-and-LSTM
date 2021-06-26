import torch as T
import torch.nn as nn

class LSTM_SENTIMENT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hid_dims, n_layers, out_dims):
        super().__init__()
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.n_layers = n_layers

        self.drop_prob = 0.5
        self.lr = 0.0008
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hid_dims, n_layers,
                            dropout=self.drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(self.drop_prob)

        self.fc = nn.Linear(self.hid_dims, self.out_dims)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, h):

        batch_size = x.shape[0]

        embed = self.embedding(x.long()).to(self.device)
        #print(h[0].shape)
        o, h = self.lstm(embed, h)

        o = o[:, -1, :]

        o = self.dropout(o)

        o = self.sigmoid(self.fc(o))

        return o, h
    
    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data

        hidden = (weight.new_zeros(self.n_layers, batch_size, self.hid_dims).to(self.device),
                  weight.new_zeros(self.n_layers, batch_size, self.hid_dims).to(self.device))

        return hidden


