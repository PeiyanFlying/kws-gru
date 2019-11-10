import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math

import numpy as np

class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True ):
        super(GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        # self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        # self.rh2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.x2h_weight = Parameter(torch.Tensor(3*hidden_size, input_size))
        self.x2h_bias = Parameter(torch.Tensor(3*hidden_size))
        self.h2h_weight = Parameter(torch.Tensor(2*hidden_size, hidden_size))
        self.h2h_bias = Parameter(torch.Tensor(2*hidden_size))
        self.rh2h_weight = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.rh2h_bias = Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()


    def reset_parameters(self):
        # init linear weight
        init.kaiming_uniform_(self.x2h_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.h2h_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.rh2h_weight, a=math.sqrt(5))
        # init linear bias
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.x2h_weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.x2h_bias, -bound, bound)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.h2h_weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.h2h_bias, -bound, bound)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.rh2h_weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.rh2h_bias, -bound, bound)


    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x_mm = torch.mm(x, self.x2h_weight.t())
        gate_x = torch.add(self.x2h_bias, gate_x_mm)
        gate_h_mm = torch.mm(hidden, self.h2h_weight.t())
        gate_h = torch.add(self.h2h_bias, gate_h_mm)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i = gate_h.chunk(2, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        rh = resetgate * hidden
        candidate_mm = torch.mm(rh, self.rh2h_weight.t())
        candidate = torch.add(self.rh2h_bias,candidate_mm)
        newgate = torch.tanh(i_n + candidate)

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layersb
        self.layer_dim = layer_dim

        self.gruLayer = GRUCell(input_dim, hidden_dim, bias=True)

        self.fcLayer = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

        print("Use self-defined GRU model!!!!")

    def forward(self, x):

        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28

        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []

        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn = self.gruLayer(x[:, seq, :], hn)
            outs.append(hn)

        out = outs[-1].squeeze()

        out = self.fcLayer(out)
        out = self.softmax(out)
        # out.size() --> 100, 12
        return out


