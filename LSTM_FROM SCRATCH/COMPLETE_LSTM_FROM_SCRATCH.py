import torch as T
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

c_t_prev = (1, 5, 57)
class LSTM(nn.Module):
    def __init__(self, in_dims, hid_dims, out_dims):
        super().__init__()

        self.hid_dims = hid_dims
        self.in_dims = in_dims
        self.out_dims = out_dims

        # T.Tensor() INITIALIZES THE TENSOR OF GIVEN SHAPE WITH ZEROS, PARAMETER TURNS ON GRAD FOR IT
        # FORGET GATE
        self.U_f = Parameter(T.Tensor(self.in_dims, self.hid_dims))
        self.V_f = Parameter(T.Tensor(self.hid_dims, self.hid_dims))
        self.B_f = Parameter(T.Tensor(self.hid_dims))

        # LEARN GATE
        self.U_i = Parameter(T.Tensor(self.in_dims, self.hid_dims))
        self.V_i = Parameter(T.Tensor(self.hid_dims, self.hid_dims))
        self.B_i = Parameter(T.Tensor(self.hid_dims))
        self.U_l = Parameter(T.Tensor(self.in_dims, self.hid_dims))
        self.V_l = Parameter(T.Tensor(self.hid_dims, self.hid_dims))
        self.B_l = Parameter(T.Tensor(self.hid_dims))

        # USE GATE
        self.U_u = Parameter(T.Tensor(self.in_dims, self.hid_dims))
        self.V_u = Parameter(T.Tensor(self.hid_dims, self.hid_dims))
        self.B_u = Parameter(T.Tensor(self.hid_dims))

        self.init_weight()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.hid_dims)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inp, prev_states):
        
        batch_size, seq, _ = inp.shape
        if prev_states:
            h, c = prev_states
        else:
            h, c = (
                T.zeros(batch_size, self.hidden_size),
                T.zeros(batch_size, self.hidden_size)   
            )
        
        output_seq = []

        for i in range(seq):
            x_t = inp[:,i,:]

            # LEARN GATE
            N_t = self.tanh(x_t @ self.U_l + h @ self.V_l  + self.B_l)
            I_t = self.sigmoid(x_t @ self.U_i  + h @ self.V_i  + self.B_i)
            L_T = N_t * I_t

            # FORGET GATE 
            F_T = self.sigmoid(x_t @ self.U_f + h @ self.V_f  + self.B_f)
            c = c * F_T

            # REMEMBER GATE
            r = c + L_T

            # USE GATE
            O_T = self.sigmoid(x_t @ self.U_u + h @ self.V_u + self.B_u)
            U_T = self.tanh(r) * O_T

            # U_T IS THE OUTPUT OF EACH INDIVIDUAL BLOCK AND FINAL OUTPUT SEQUENCE IS CONCATENATION OF INDIVIDUAL OUTPUTS
            output_seq.append(U_T)

        output_seq = T.cat(output_seq)
        output_seq = output_seq.transpose(0, 1).contiguous() # TO CHANGE TO (BATCH, SEQ_LEN, HIDDEN_DIM)
                            # IMPORTANT : : : https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
        return output_seq, (U_T, r)


lstm = LSTM(10, 20, 1)
input = T.randn(1, 5, 10)
h0 = T.randn(1, 1, 20)
c0 = T.randn(1, 1, 20)
Output, (h, c) = lstm(input, (h0, c0))
print(
    f'OUR MODEL :\n\tINPUT : {input.shape}\n\tSTM : {h0.shape}\n\tLTM : {c0.shape}\n\tOUTPUT : {Output.shape}\n')
# INBUILT MODULE nn.LSTM()
rnn = nn.LSTM(10, 20, 1, batch_first = True)
Output, (hn, cn) = rnn(input, (h0, c0))
print(
    f'PYTORCH BUILT IN :\n\tINPUT : {input.shape}\n\tSTM : {h0.shape}\n\tLTM : {c0.shape}\n\tOUTPUT : {Output.shape}')

"""
                                                    OUTPUT
                                                    OUR MODEL :
                                                            INPUT : torch.Size([1, 5, 10])
                                                            STM : torch.Size([1, 1, 20])
                                                            LTM : torch.Size([1, 1, 20])
                                                            OUTPUT : torch.Size([1, 5, 20])

                                                    PYTORCH BUILT IN :
                                                            INPUT : torch.Size([1, 5, 10])
                                                            STM : torch.Size([1, 1, 20])
                                                            LTM : torch.Size([1, 1, 20])
                                                            OUTPUT : torch.Size([1, 5, 20])
"""





