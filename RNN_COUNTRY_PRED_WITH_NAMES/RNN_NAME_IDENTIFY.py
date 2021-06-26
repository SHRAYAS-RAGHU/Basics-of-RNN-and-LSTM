import string
import torch as T
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from RNN import RNN
from Data_Loader import  load_dataset
from Data_Loader import  obtain_sample

LETTERS = string.ascii_letters + ",.;'"
N_LETTERS = len(LETTERS)

dataset, labels = load_dataset()

rnn = RNN(57, 128, len(labels))
rnn.load_state_dict(T.load(r'C:\Users\krshr\Desktop\Files\Deep_learning\NLP_RNN_LSTM\RNN_COUNTRY_PRED_WITH_NAMES\Model.pth'))

loss_fn = nn.NLLLoss()
optimizer = SGD(rnn.parameters(), lr = 0.005)

def train(inp, target):                 # inp is a tensor of string like 'Albert' of shape(len, 1, 57)
    hid = rnn.init_hidden()
    for i in range(inp.shape[0]):
        out, hid= rnn(inp[i], hid)

    loss = loss_fn(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return out, loss.item()

runnin_loss = 0
loss = []
prt_every = 500

for i in range(100000):
    data, label = obtain_sample(dataset, labels)

    out, l = train(data, label)

    runnin_loss += l
    loss.append(l)

    if i % prt_every == 0:
        print(f'{i} th EPISODE LOSS: {l} MEAN:{np.average(loss)}')

T.save(rnn.state_dict(), 'Model.pth')

