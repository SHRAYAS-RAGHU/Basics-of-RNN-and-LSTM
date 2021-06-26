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

        # SINCE WE DO REPEATED MULTIPLICATION OF WEIGHTS WITH INPUT AND STM ITS BETTER TO DO MATRIX MULTIPLICAIOTN
        # T.Tensor() INITIALIZES THE TENSOR OF GIVEN SHAPE WITH ZEROS, PARAMETER TURNS ON GRAD FOR IT
        self.U = Parameter(T.Tensor(self.in_dims, 4 * self.hid_dims))
        self.V = Parameter(T.Tensor(self.hid_dims, 4 * self.hid_dims))
        self.B = Parameter(T.Tensor(4 * self.hid_dims))

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
            H_T, C_T = prev_states
        else:
            H_T, C_T = (
                T.zeros(batch_size, self.hidden_size),
                T.zeros(batch_size, self.hidden_size)   
            )
        
        output_seq = []

        for i in range(seq):
            x_t = inp[:,i,:]

            GATE_COMBINED_OUTPUT = x_t @ self.U + H_T @ self.V + self.B

            #LEARN
            N_T = self.tanh(GATE_COMBINED_OUTPUT[:, :, 0:self.hid_dims])
            I_T = self.sigmoid(GATE_COMBINED_OUTPUT[:, :, self.hid_dims:2*self.hid_dims])
            L_T = N_T * I_T
        
            #FORGET
            F_T = self.sigmoid(
                GATE_COMBINED_OUTPUT[:, :, 2*self.hid_dims:3*self.hid_dims])
            C_T *= F_T

            #REMEBER
            R_T = C_T + L_T

            #USE
            O_T = self.sigmoid(
                GATE_COMBINED_OUTPUT[:, :, 3*self.hid_dims:4*self.hid_dims])
            U_T = self.tanh(R_T) * O_T
            # U_T IS THE OUTPUT OF EACH INDIVIDUAL BLOCK AND FINAL OUTPUT SEQUENCE IS CONCATENATION OF INDIVIDUAL OUTPUTS
            output_seq.append(U_T)

        output_seq = T.cat(output_seq)
        output_seq = output_seq.transpose(0, 1).contiguous() # TO CHANGE TO (BATCH, SEQ_LEN, HIDDEN_DIM)
                            # IMPORTANT : : : https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
        return output_seq, (U_T, R_T)


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





