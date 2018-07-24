import math

import torch

import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.linear1 = nn.Linear(in_features, out_features)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.linear1.weight.data.normal_(0, 0.005)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, feat_adj):
        # print('---------- ADJ ------------')
        # print(adj)
        # print('-------- FEAT_ADJ ---------')
        # print(feat_adj)
        new_input = torch.spmm(feat_adj.t(), input.t()).t()
        # weight = torch.spmm(feat_adj, self.weight)
        support = torch.mm(new_input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution_GI(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_GI, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.linear1 = nn.Linear(2 * in_features, in_features)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(500, in_features)
        self.sigmoid = nn.Sigmoid()
        self.gi_net = nn.Sequential(self.linear1, self.sigmoid)
        self.mask = torch.eye(in_features).unsqueeze(0).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.linear1.weight.data.normal_(0, 0.01)
        # self.linear2.weight.data.normal_(0, 0.01)

    def forward(self, input, adj, feat_adj):
        batch_size = input.size(0)
        input_tiled = input.unsqueeze(0).repeat(batch_size, 1, 1)
        gi_input = torch.cat((input_tiled, input_tiled.transpose(0, 1)), 2)
        gi = self.gi_net(gi_input) * (adj.to_dense().unsqueeze(2).repeat(1, 1, self.in_features))
        gi = gi.transpose(1, 2).reshape(batch_size * self.in_features, -1)
        # support = torch.mm(input, self.weight)
        
        input_gated = torch.mm(gi, input).reshape(batch_size, self.in_features, self.in_features)
        # mask = self.mask.repeat(batch_size, 1, 1)
        # support = (input_gated * mask).sum(1)
        # output = torch.spmm(adj, support)
        support = input_gated.sum(1)
        output = torch.mm(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution_raw(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_raw, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, feat_adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        # output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
