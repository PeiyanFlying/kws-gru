import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import time
import tensorflow as tf
import argparse
import sys
sys.path.append("..")
from model import models
import input_data
import numpy as np
import copy

act_bits = 4

class sigmoidquantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        bits = act_bits
        n =  float(2**bits-1)

        input_tem = torch.abs(input)
        _max = torch.max(input_tem)

        if _max > 0:
            div = torch.floor(torch.log2(_max)) + 1
        else:
            div = 0
        q_number = bits - div

        if q_number < 0:
            q_number = torch.clamp(q_number, 0, 0)

        input = input.mul(2 ** q_number).trunc().div(2 ** q_number)  # 数学的角度进行quant//与activation的quantization一样
        threshold = 2 ** (bits - q_number) - 2 ** (-q_number)  # boundary    这个数学算法好nb呀。最大值定模，剩余做小数。  在合适的bit范围内；matrix的最大，不行就bit的最大。
        input = torch.clamp(input, -threshold, threshold)

        out = torch.round(input*(n/1.0)*0.2)*(1.0/n) + 0.5 #这一步本质上是二进制的数学运算
        out = torch.clamp(out,0.0,1.0)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 0.2*grad_output.clone()
        grad_input[input.ge(2.5)] = 0
        grad_input[input.le(-2.5)] = 0
        return grad_input

class Sigmoidquant(nn.Module):
    '''
    Quant the input activations
    '''
    def __init__(self, bits):
        super(Sigmoidquant, self).__init__()
        self.sigmoidquant = sigmoidquantization.apply
        assert bits >= 1, bits
        self.bits = bits

    def forward(self, input):
        out = self.sigmoidquant(input)
        #out = input
        return out


class Tanhquantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        bits = act_bits
        n =  float(2**bits-1)
        threshold = 1.0
        out = torch.round(input * (n / threshold)) * (threshold / n)
        out = torch.clamp(out, -1.0, 1.0)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1.0)] = 0
        grad_input[input.le(-1.0)] = 0
        return grad_input

class Tanhquant(nn.Module):
    '''
    Quant the input activations
    '''
    def __init__(self, bits):
        super(Tanhquant, self).__init__()
        self.tanhquant = Tanhquantization.apply
        assert bits >= 1, bits
        self.bits = bits

    def forward(self, input):
        out = self.tanhquant(input)
        #out = input
        return out

class act_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        bits = act_bits
        input_tem = torch.abs(input)
        _max = torch.max(input_tem)

        if _max > 0:
            div = torch.floor(torch.log2(_max)) + 1
        else:
            div = 0
        q_number = bits - div

        if q_number < 0:
            q_number = torch.clamp(q_number, 0, 0)

        out = input.mul(2 ** q_number).trunc().div(2 ** q_number)  # 数学的角度进行quant//与activation的quantization一样
        threshold = 2 ** (bits - q_number) - 2 ** (-q_number)  # boundary    这个数学算法好nb呀。最大值定模，剩余做小数。  在合适的bit范围内；matrix的最大，不行就bit的最大。
        out = torch.clamp(out, -threshold, threshold)
        ctx.save_for_backward(input, threshold)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, max_value = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(max_value)] = 0
        grad_input[input.le(-max_value)] = 0
        return grad_input

class Act_Quant(nn.Module):
    '''
    Quant the input activations
    '''
    def __init__(self, bits):
        super(Act_Quant, self).__init__()
        self.act_quantion = act_quant.apply
        assert bits >= 1, bits
        self.bits = bits


    def forward(self, input):
        out = self.act_quantion(input)
        #out = input
        return out



class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, save_act_value=False, save_act_dir=None):
        super(GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        # self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        # self.rh2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.x2h_weight = Parameter(torch.Tensor(3*hidden_size, input_size))
        self.x2h_bias = Parameter(torch.Tensor(3*hidden_size))
        self.h2h_weight = Parameter(torch.Tensor(2*hidden_size, hidden_size))
        self.h2h_bias = Parameter(torch.Tensor(2*hidden_size))
        self.rh2h_weight = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.rh2h_bias = Parameter(torch.Tensor(hidden_size))
        self.act0 = Act_Quant(bits=act_bits)
        self.act9 = Act_Quant(bits=act_bits)
        self.sig1 = Sigmoidquant(bits=act_bits)
        self.sig2 = Sigmoidquant(bits=act_bits)
        self.tanh = Tanhquant(bits=act_bits)
        self.reset_parameters()

        self.save_act_value = save_act_value
        self.save_act_dir = save_act_dir
        # counter to save activation value
        self.count = 0

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
        x = self.act0(x)

        gate_x_mm = torch.mm(x, self.x2h_weight.t())
        gate_x = torch.add(self.x2h_bias, gate_x_mm)
        gate_h_mm = torch.mm(hidden, self.h2h_weight.t())
        gate_h = torch.add(self.h2h_bias, gate_h_mm)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i = gate_h.chunk(2, 1)

        resetgate = self.sig1(i_r + h_r)
        inputgate = self.sig2(i_i + h_i)
        rh = resetgate * hidden
        candidate_mm = torch.mm(rh, self.rh2h_weight.t())
        candidate = torch.add(self.rh2h_bias,candidate_mm)
        newgate = self.tanh(i_n + candidate)

        if self.save_act_value == True:
            torch.save(i_r + h_r, self.save_act_dir + 'Before_resetgate_' + str(self.count))
            torch.save(i_i + h_i, self.save_act_dir + 'Before_inputgate_' + str(self.count))
            torch.save(i_n + candidate, self.save_act_dir + 'Before_newgate_' + str(self.count))


        hy = newgate + inputgate * (hidden - newgate)
        hy = self.act9(hy)

        if self.save_act_value == True:
            torch.save(x.data, self.save_act_dir + 'input_' + str(self.count))
            torch.save(gate_x_mm.data, self.save_act_dir + 'gate_x_mm_' + str(self.count))
            torch.save(gate_x.data, self.save_act_dir + 'gate_x_' + str(self.count))
            torch.save(gate_h_mm.data, self.save_act_dir + 'gate_h_mm_' + str(self.count))
            torch.save(gate_h.data, self.save_act_dir + 'gate_h_' + str(self.count))
            torch.save(candidate_mm.data, self.save_act_dir + 'candidate_mm_' + str(self.count))
            torch.save(candidate.data, self.save_act_dir + 'candidate_' + str(self.count))
            torch.save(resetgate.data, self.save_act_dir + 'resetgate_' + str(self.count))
            torch.save(inputgate.data, self.save_act_dir + 'inputgate_' + str(self.count))
            torch.save(newgate.data, self.save_act_dir + 'newgate_' + str(self.count))
            torch.save(hy.data, self.save_act_dir + 'hy_' + str(self.count))

        self.count = self.count + 1
        return hy


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, save_act_value=False, save_act_dir=None):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layersb
        self.layer_dim = layer_dim

        self.gruLayer = GRUCell(input_dim, hidden_dim, save_act_value, save_act_dir)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help="""\
        checkpoint file path
        """
        )
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.checkpoint=="":
        # print("A GRU model checkpoint is needed")
        exit("A GRU model checkpoint is needed")
    def save_checkpoint(model, filename='checkpoint.pth.tar'):
        """
        Save the training model
        """
        state = {}
        state['state_dict'] = model.state_dict()

        torch.save(state, filename)

    rnn = GRUModel(10, 100, 1, 12).cuda()
    rnn = rnn.train()
    input = torch.rand(100, 25, 10)
    input = torch.autograd.Variable(input).cuda()
    for name, w in rnn.named_parameters():
        print(name, w.shape)
    # output = rnn(input)
    # print(output.shape)
    print("=========================================")

    previous_name=FLAGS.checkpoint
    checkpoint_read = torch.load(previous_name)
    new_dict={}
    for k, v in checkpoint_read['state_dict'].items():
        if k=="gruLayer.x2h.weight":
            new_dict["gruLayer.x2h_weight"]=v
        elif k=="gruLayer.h2h.weight":
            new_dict["gruLayer.h2h_weight"]=v
        elif k=="gruLayer.x2h.bias":
            new_dict["gruLayer.x2h_bias"]=v
        elif k=="gruLayer.h2h.bias":
            new_dict["gruLayer.h2h_bias"]=v
        elif k=="gruLayer.rh2h.weight":
            new_dict["gruLayer.rh2h_weight"]=v
        elif k=="gruLayer.rh2h.bias":
            new_dict["gruLayer.rh2h_bias"]=v
        else:
            new_dict[k]=v
        print(k, v.shape)
    rnn.load_state_dict(new_dict)
    
    tmp_path = previous_name.split('/')
    tmp_path[-1] = 'new_'+tmp_path[-1]
    new_path = '/'.join(tmp_path)

    save_checkpoint(rnn, new_path)

