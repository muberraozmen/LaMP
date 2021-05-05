import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter

class XavierLinear(nn.Module):
    """
    Status:
        checked
    """

    def __init__(self, d_in, d_out, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class CompGCN(nn.Module):

    def __init__(self, in_features, out_features):
        super(CompGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        cuda_on = input.is_cuda

        A_in = torch.triu(adj, diagonal=1)
        A_out = A_in.transpose(2, 1)
        A_self = torch.eye(A_in.shape[-1], A_in.shape[-1]).unsqueeze(0).type(torch.bool)
        if cuda_on:
            A_in = A_in.cuda()
            A_out = A_out.cuda()
            A_self = A_self.cuda()

        support = torch.matmul(input, self.w)
        output_in = torch.matmul(A_in.float(), support.float())
        output_out = torch.matmul(A_out.float(), support.float())
        output_self = torch.matmul(A_self.float(), support.float())
        output = output_in + output_out + output_self

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MrMP(nn.Module):

    def __init__(self, adjs, d_word_vec=512, n_layers=2, phi_mode='sum'):
        super(MrMP, self).__init__()

        self.n_layers = n_layers
        self.phi_mode = phi_mode

        self.adjs = adjs
        self.reln_emb = torch.nn.Embedding(1, d_word_vec)

        self.layer_stack = nn.ModuleList()
        self.reln_emb_updates = nn.ModuleList()
        for _ in range(n_layers):
            self.layer_stack.append(CompGCN(d_word_vec, d_word_vec))  # for pulling
            self.layer_stack.append(CompGCN(d_word_vec, d_word_vec))  # for pushing
            self.reln_emb_updates.append(XavierLinear(d_word_vec, d_word_vec))

    def phi(self, label_emb, reln_emb, mode):
        if mode == 'mul':
            return label_emb * reln_emb
        elif mode == 'sum':
            return label_emb + reln_emb

    def forward(self, tgt):
        labels = tgt
        cuda_on = tgt.is_cuda

        r = torch.tensor([0])
        if cuda_on is True:
            r = r.cuda()

        for layer in range(self.n_layers):
            gcn_output = 0

            relations = self.reln_emb_updates[layer](self.reln_emb(r))

            pulling = relations.reshape(1, 1, -1)
            gcn_input = self.phi(labels, pulling, self.phi_mode)
            gcn_output += self.layer_stack[layer * 2](gcn_input, self.adjs[0])

            pushing = - pulling
            gcn_input = self.phi(labels, pushing, self.phi_mode)
            gcn_output += self.layer_stack[layer * 2 + 1](gcn_input, self.adjs[1])

            if layer != self.n_layers - 1:
                labels = torch.relu(gcn_output)
            else:
                labels = gcn_output

        # output = labels
        output = 2*(labels - labels.min())/(labels.max()-labels.min())-1

        return output

