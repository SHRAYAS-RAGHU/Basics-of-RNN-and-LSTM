import torch.nn as nn
import torch

rnn = nn.LSTM(2, 4, 1, batch_first = True)
input = torch.randn(1, 5, 2)
h0 = torch.randn(1, 1, 4)
c0 = torch.randn(1, 1, 4)
Output, (hn, cn) = rnn(input, (h0, c0))
print(
    f'INPUT : {input.shape}\nSTM : {hn.shape}\nLTM : {cn.shape}\nOUTPUT : {Output.shape}')
print(
    f'INPUT : {input}\nSTM : {hn}\nLTM : {cn}\nOUTPUT : {Output}')
