import torch as T
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.parameter import Parameter
import math

class LEARN_GATE(nn.Module):
    def __init__(self, in_dims, hid_dims, out_dims):
        super().__init__()

        self.hid_dims = hid_dims
        self.in_dims = in_dims
        self.out_dims = out_dims

        # INPUT SHAPE (BATCH, SEQ, INP_DIM)             E.G (1, 5, 57)
        # HIDDEN SHAPE (NUM_LAYERS, BATCH, HIDDEN_DIM)  E.G (1, 1, 128)

        # THOUGH INPUT IS A SEQUENCE OF LENGTH 5 ITS PASSED ONE INSTANCE AT A TIME
        self.U1 = Parameter(
            T.Tensor(self.in_dims, self.hid_dims))      # (1, 1, 57) @ (57, 128) = (1, 1, 128)
        self.V1 = Parameter(
            T.Tensor(self.hid_dims, self.hid_dims))     # (1, 1, 128) @ (128, 128) = (1, 1, 128)
        self.B1 = Parameter(
            T.Tensor(self.hid_dims))                    # (128)
        
        self.U2 = Parameter(
            T.Tensor(self.in_dims, self.hid_dims))      # (1, 1, 57) @ (57, 128) = (1, 1, 128)
        self.V2 = Parameter(
            T.Tensor(self.hid_dims, self.hid_dims))     # (1, 1, 128) @ (128, 128) = (1, 1, 128)
        self.B2 = Parameter(
            T.Tensor(self.hid_dims))                    # (128)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inp, mem):
        b, s, _ = inp.shape

        if mem is None:
            h_t, c_t = (
                T.zeros(b, self.hidden_size),
                T.zeros(b, self.hidden_size)
            )
        else:
            h_t, _ = mem

        for i in range(s):
            x_t = inp[:, i, :]

            sigmoid = nn.Sigmoid()
            I_T = sigmoid(x_t @ self.U1 + h_t @ self.V1  + self.B1)

            tanh = nn.Tanh()
            N_T = tanh(x_t @ self.U2 + h_t @ self.V2 + self.B2)

            L_T = N_T * I_T

            print(L_T.shape)
            print(f'L_T : {L_T}')


a = LEARN_GATE(3, 5, 2)
inp = T.randn((1, 2, 3))
h0 = T.randn((1, 1, 5))
c0 = T.randn((1, 1, 5))
print(f'INPUT :{inp}\nSTM{h0}\nLTM{c0}\n')
a.forward(inp, (h0,c0))


"""
INPUT :tensor([[[ 0.7118, -0.3781, -0.6347],
         [-0.4634, -0.1693,  0.3809]]])
STMtensor([[[-1.1169,  1.7287,  0.6230,  0.4987,  0.1512]]])
LTMtensor([[[ 1.0684,  0.5852,  0.1419, -0.4173, -0.8350]]])

torch.Size([1, 1, 5])
L_T : tensor([[[0.0000e+00, 0.0000e+00, 1.7705e-38, 1.3437e-38, 0.0000e+00]]],
       grad_fn=<MulBackward0>)
torch.Size([1, 1, 5])
L_T : tensor([[[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.9964e-38]]],
       grad_fn=<MulBackward0>)
"""
