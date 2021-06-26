import torch as T
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from Dataloader import DATA_LOADER
dat = DATA_LOADER()
train, val, test = dat.sentiment_dataloader()

batch_size = 50
vocab_size = len(dat.words_to_int) + 1
out_dim = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

from LSTM_model import LSTM_SENTIMENT
lstm = LSTM_SENTIMENT(vocab_size, embedding_dim,
                      hidden_dim, n_layers, out_dim)
#print(lstm)

loss_fn = nn.BCELoss()
optimizer = Adam(lstm.parameters(), lr = 0.001)

ctr = 0

print_every = 100

for e in range(4):
    hidden = lstm.init_hidden(batch_size)

    for inp, tgt in train:
        ctr += 1

        hidden = tuple([x.data() for x in hidden]) 

        lstm.zero_grad()

        inp, tgt = inp.to(lstm.device), tgt.to(lstm.device)

        out, hidden = lstm(inp, hidden)

        loss = loss_fn(out.squeeze(), tgt.float())

        loss.backward()

        nn.utils.clip_grad_norm_(lstm.parameters(), 5)

        optimizer.step()

        if ctr % print_every == 0:
            val_loss = []

            val_h = lstm.init_hidden(batch_size)
            
            lstm.eval()

            for val_x, val_y in val:
                
                val_h = tuple([x.data() for x in val_h])

                val_x, val_y = val_x.to(lstm.device), val_y.to(lstm.device)

                val_out, val_h = lstm(val_x, val_h)

                val_l = loss_fn(val_out.squeeze(), val_y.float())
                val_loss.append(val_l.item())

            lstm.train()

            print(f'\
                EPOCH : {e:>2}\
                STEP :  {ctr:>2}\
                VAL_LOSS : {np.mean(val_loss):>8.6f}\
                LOSS : {loss.item():>8.6f}')

T.save(lstm.state_dict(), 'chkpt.pth')
