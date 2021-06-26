from RNN import RNN
import torch as T
from Data_Loader import load_dataset
from Data_Loader import obtain_sample

rnn = RNN(57, 128, 18)
rnn.load_state_dict(
    T.load(r'C:\Users\krshr\Desktop\Files\Deep_learning\NLP_RNN_LSTM\RNN_COUNTRY_PRED_WITH_NAMES\Model.pth'))
a,b = load_dataset()

for j in range(10):
    x, y = obtain_sample(a, b)
    hid = rnn.init_hidden()
    for i in range(len(x)):
        out, hid = rnn(x[i], hid)
    print(
        f"Pred : {b[T.argmax(out)]:>10}, Truth : {b[y]:>10}, OUTPUT : {(T.argmax(out)==y).item():>2} Prob : {T.exp(T.max(out)):.2f}")
