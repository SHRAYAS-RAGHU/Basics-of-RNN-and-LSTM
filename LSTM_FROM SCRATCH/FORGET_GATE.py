import torch as T
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.parameter import Parameter
import math

class Forget_GATE(nn.Module):
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
            h_t, c_t = mem

        for i in range(s):
            x_t = inp[:, i, :]

            sigmoid = nn.Sigmoid()
            F_T = sigmoid(x_t @ self.U1 + h_t @ self.V1  + self.B1)
            c_t = c_t * F_T

            print(F_T.shape, c_t.shape)
            print(f'F_T : {F_T}\nc_t : {c_t}')


a = Forget_GATE(3, 5, 2)
inp = T.randn((1, 2, 3))
h0 = T.randn((1, 1, 5))
c0 = T.randn((1, 1, 5))
print(f'INPUT :{inp}\nSTM{h0}\nLTM{c0}\n')
a.forward(inp, (h0,c0))


"""
INPUT :tensor([[[ 0.3994,  0.5193, -0.2878],
         [-0.2518,  0.2287, -1.4256]]])
STMtensor([[[-2.0669, -0.2781, -2.8618,  0.1783, -0.2322]]])
LTMtensor([[[ 0.4606, -0.6910, -0.6507, -0.4817,  0.0984]]])

torch.Size([1, 1, 5]) torch.Size([1, 1, 5])
F_T : tensor([[[0.5000, 0.5000, 0.5000, 0.5000, 0.5000]]], grad_fn=<SigmoidBackward>)
c_t : tensor([[[ 0.2303, -0.3455, -0.3254, -0.2409,  0.0492]]],       grad_fn=<MulBackward0>)
torch.Size([1, 1, 5]) torch.Size([1, 1, 5])
F_T : tensor([[[0.5000, 0.5000, 0.5000, 0.5000, 0.5000]]], grad_fn=<SigmoidBackward>)
c_t : tensor([[[ 0.1151, -0.1727, -0.1627, -0.1204,  0.0246]]],       grad_fn=<MulBackward0>)
"""
