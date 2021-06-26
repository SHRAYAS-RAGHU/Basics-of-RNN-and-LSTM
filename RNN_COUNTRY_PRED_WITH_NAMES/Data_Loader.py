import matplotlib.pyplot as plt
import os
import io
import unicodedata
import string
from pathlib import Path
import torch as T
import numpy as np

LETTERS = string.ascii_letters + ",.;'"
N_LETTERS = len(LETTERS)

path = Path(
    r'C:\Users\krshr\Desktop\Files\Deep_learning\NLP_RNN_LSTM\RNN_COUNTRY_PRED_WITH_NAMES\data\names')
files = os.listdir(path)
#  Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427

def U_to_A(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in LETTERS
    )

def Line_encoding(line):
    a = T.zeros((len(line), 1, 57))
    for i, j in enumerate(line):
        a[i, 0, LETTERS.find(j)] = 1
    return a

def load_dataset():
    dataset = {}
    labels = []

    for file in files:
        index = file.strip('.txt')
        a = io.open(path / file, encoding='utf-8')
        dataset[index] = [U_to_A(i) for i in a]
        labels.append(index)
    
    return dataset, labels

def obtain_sample(dataset, labels):    
    choice = lambda x: x[np.random.randint(0, len(x))]

    label = choice(labels)
    data = choice(dataset[label])
    data = Line_encoding(data)
    label = T.tensor([labels.index(label)], dtype=T.long)

    return data, label
