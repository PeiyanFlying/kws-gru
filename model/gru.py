import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math


def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!


global_bits = 5
global_alpha = None

# 这个肯定是一个多bit的quant
class sigmoidquantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        bits = global_bits
        n =  float(2**bits-1)
        threshold = 1.0
        out = input
        out = torch.round(input*(n/threshold)*0.4)*(threshold/n)
        out = torch.clamp(out,0.0,threshold)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
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

        bits = global_bits
        n =  float(2**bits-1)
        threshold = 1.0
        out = input
        out = torch.round(input*(n/threshold))*(threshold/n)
        out = torch.clamp(out,-1.0,1.0)
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


# 其实保证乘加运算前面以及BN后面跟着激活函数就好... 没必要激活函数夹着BN


class GRUModel(nn.Module): # 删除了options

    def __init__(self, input_dim, num_classes):
        super(GRUModel, self).__init__()

        # Reading parameters
        self.input_dim = input_dim
        self.gru_lay = list(map(int, options['gru_lay'].split(',')))
        self.gru_drop = list(map(float, options['gru_drop'].split(',')))
        self.gru_use_batchnorm = list(map(strtobool, options['gru_use_batchnorm'].split(',')))
        self.gru_use_laynorm = list(map(strtobool, options['gru_use_laynorm'].split(',')))
        self.gru_use_laynorm_inp = strtobool(options['gru_use_laynorm_inp'])
        self.gru_use_batchnorm_inp = strtobool(options['gru_use_batchnorm_inp'])
        self.gru_orthinit = strtobool(options['gru_orthinit'])
        self.gru_act = options['gru_act'].split(',')
        self.bidir = strtobool(options['gru_bidir'])
        self.use_cuda = strtobool(options['use_cuda'])


        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.wr = nn.ModuleList([])  # Reset Gate
        self.ur = nn.ModuleList([])  # Reset Gate

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm
        self.bn_wz = nn.ModuleList([])  # Batch Norm
        self.bn_wr = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations
        self.sig = nn.ModuleList([])  # Activations sig

        # 添加分类器
        self.fcLayer = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

        # Input layer normalization
        if self.gru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.gru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_gru_lay = len(self.gru_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_gru_lay):

            # Activations
            self.act.append(Tanhquant(bits=global_bits))
            self.sig.append(Sigmoidquant(bits=global_bits))

            add_bias = True

            if self.gru_use_laynorm[i] or self.gru_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))
            self.wr.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))
            self.ur.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))

            if self.gru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
                nn.init.orthogonal_(self.ur[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))
            self.bn_wr.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.gru_lay[i]))

            if self.bidir:
                current_input = 2 * self.gru_lay[i]
            else:
                current_input = self.gru_lay[i]

        self.out_dim = self.gru_lay[i] + self.bidir * self.gru_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.gru_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.gru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_gru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.gru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.gru_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.gru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.gru_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            wr_out = self.wr[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.gru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])

                wr_out_bn = self.bn_wr[i](wr_out.view(wr_out.shape[0] * wr_out.shape[1], wr_out.shape[2]))
                wr_out = wr_out_bn.view(wr_out.shape[0], wr_out.shape[1], wr_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # gru equation
                zt = self.sig(wz_out[k] + self.uz[i](ht))
                rt = self.sig(wr_out[k] + self.ur[i](ht))
                at = wh_out[k] + self.uh[i](rt * ht)
                hcand = self.act[i](at) * drop_mask
                ht = (zt * ht + (1 - zt) * hcand)

                if self.gru_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        out = x[-1].squeeze()

        out = self.fcLayer(out)
        out = self.softmax(out)

        return out

