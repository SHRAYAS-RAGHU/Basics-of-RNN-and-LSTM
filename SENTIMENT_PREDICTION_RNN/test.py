import torch.nn as nn
import torch as T
import numpy as np
from Dataloader import DATA_LOADER
d = DATA_LOADER()

vocab_size = len(d.words_to_int) + 1
out_dim = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2
from LSTM_model import LSTM_SENTIMENT
lstm = LSTM_SENTIMENT(vocab_size, embedding_dim,
                      hidden_dim, n_layers, out_dim)
lstm.load_state_dict(T.load(r'C:\Users\krshr\Desktop\Files\Deep_learning\NLP_RNN_LSTM\SENTIMENT_PREDICTION_RNN\RNN_STATE_DICT.pth', map_location='cpu'))
lstm.eval()

test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'

from string import punctuation
test_review = test_review_neg.lower()
test_review = ''.join([i for i in test_review if i not in punctuation])
test_review = test_review.split()
encoded_test_review = [[d.words_to_int[i] for i in test_review]]
features = d.std_length(encoded_test_review, 241)
features = T.from_numpy(features)

batch_size = features.shape[0]

h = lstm.init_hidden(batch_size)
output, h = lstm(features, h)
pred = T.round(output.squeeze())

print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

if(pred.item() == 1):
    print("Positive review detected!")
else:
    print("Negative review detected.")
