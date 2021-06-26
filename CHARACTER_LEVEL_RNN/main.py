from DATA_LOADER import DATA_LOADER
from RNN import LSTM
import torch as T
from torch.optim import Adam
import torch.nn as nn
import numpy as np

def train(data_loader, net_object, dataset, epochs=10, batch_size=8, 
                seq_length=50, lr=0.001, clip_grad_value=5, val_frac=0.1, print_every=10, device = 'cpu'):
    net_object.train()

    data, no_of_char = dataset

    optimizer = Adam(net_object.parameters(), lr = lr)
    loss_fn = nn.CrossEntropyLoss()

    # VALIDATION SPLIT OF 0.1 CAN BE DONE BY FINDING THE VAL_INDEX AND SPLITTING IT
    val_index = int(len(data) * (1 - val_frac))
    train_data, val_data = data[:val_index], data[val_index:]

    ctr = 0
    min_val_loss = 10
    for e in range(epochs):
        
        # CREATES HIDDEN LAYER (H, C) FOR LTM AND STM
        h = net_object.init_hidden(batch_size)

        for x, y in data_loader.get_batch(train_data, batch_size, seq_length):

            h = tuple([hh.data for hh in h])
            net_object.zero_grad()
            x = data_loader.one_hot(x, no_of_char)
            x = T.Tensor(x).to(device)
            # PYTORCH EXPECTS TARGET OF TYPE LONG FOR CE-LOSS
            y = T.Tensor(y).view(batch_size*seq_length).long().to(device)
            out, h = net_object(x, h)
            #print(out.shape, y.shape, hidden[0].shape, hidden[1].shape)
            loss = loss_fn(out, y)
            loss.backward()

            # TO PREVENT THE PROBLEM OF EXPLODING GRADIENTS WE CLIP THE GRADIENTS
            nn.utils.clip_grad.clip_grad_norm_(net_object.parameters(), clip_grad_value)
            optimizer.step()
            
            if ctr % print_every == 0:
                # Get validation loss
                val_h = net_object.init_hidden(batch_size)
                val_losses = []
                net_object.eval() # NETWORK SET FOR EVALUVATION TO STOP BACK PROP
                for x, y in data_loader.get_batch(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = data_loader.one_hot(x, no_of_char)
                    x, y = T.Tensor(x), T.Tensor(y).view(batch_size*seq_length).long()

                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x.to(device), y.to(device)
                    

                    output, val_h = net_object(inputs, val_h)
                    val_loss = loss_fn(output, targets)

                    val_losses.append(val_loss.item())

                net_object.train()  

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(ctr),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

                min_val_loss = min(val_loss, min_val_loss)
                
                if val_loss < loss and val_loss <= min_val_loss:
                    T.save(net_object.state_dict(), r'/content/drive/MyDrive/CHARACTER_LEVEL_RNN/checkpoint.pth')

data_loader = DATA_LOADER(r'C:\Users\krshr\Desktop\Files\Deep_learning\NLP_RNN_LSTM\CHARACTER_LEVEL_RNN\data.txt')
encoded_text, no_of_char = data_loader.load_data()

lstm = LSTM(no_of_char, 512, 2, no_of_char)
lstm.to(lstm.device)
#print(lstm)
batch_size = 128
seq_length = 100
n_epochs = 20  # start smaller if you are just testing initial behavior

# train the model
train(data_loader, lstm, (encoded_text, no_of_char), epochs=n_epochs, batch_size=batch_size,
      seq_length=seq_length, lr=0.001, print_every=10, device='cpu')


