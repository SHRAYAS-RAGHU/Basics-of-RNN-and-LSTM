import numpy as np
import torch as T
import torch.nn.functional as F
from DATA_LOADER import DATA_LOADER
from RNN import LSTM

data_loader = DATA_LOADER(
    r'C:\Users\krshr\Desktop\Files\Deep_learning\NLP_RNN_LSTM\CHARACTER_LEVEL_RNN\data.txt')

lstm = LSTM(data_loader.n_characters, 512, 2, data_loader.n_characters)
state_dict = T.load(r'C:\Users\krshr\Desktop\Files\Deep_learning\NLP_RNN_LSTM\CHARACTER_LEVEL_RNN\checkpoint.pth', map_location = 'cpu')

lstm.load_state_dict(state_dict)
lstm.eval()

prime = 'Anna'

# TO GIVE A PROPER START TO THE PREDICTION
doc = [ch for ch in prime]

h = lstm.init_hidden(1)

TOP_K = 5


def prediction(dataloader, lstm, character, h):
    h = tuple([each.data for each in h])

    character = np.array([[dataloader.C2I[character]]])
    character = dataloader.one_hot(character, dataloader.n_characters)
    character = T.Tensor(character)

    character = character

    out, h = lstm(character, h)

    prob = F.softmax(out, dim=1).data

    prob, top_ch = prob.topk(TOP_K)
    top_ch, prob = top_ch.numpy().squeeze(), prob.numpy().squeeze()

    char = np.random.choice(top_ch, p=prob/prob.sum())

    return dataloader.I2C[char], h


for ch in doc:
    char, h = prediction(data_loader, lstm, ch, h)

doc.append(char)

for i in range(1000):
    char, h = prediction(data_loader, lstm, doc[-1], h)
    doc.append(char)

print(''.join(doc))


