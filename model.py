# -*- encoding: utf-8 -*-

import math
import torch

######################################## CCL ########################################
class CCL(torch.nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_hop, dropout_gcn, dropout_pre, use_bn_gcn, use_bn_pre, bias_gcn, bias_pre):
        super(CCL, self).__init__()

        self.num_class = dim_output
        self.num_hop = num_hop
        
        # propagation
        self.weight_gcn = torch.nn.Parameter(torch.FloatTensor(dim_input, dim_hidden))
        stdv_hidden = 1./math.sqrt(dim_hidden)
        self.weight_gcn.data.normal_(-stdv_hidden, stdv_hidden)
        if bias_gcn:
            self.bias_gcn = torch.nn.Parameter(torch.FloatTensor(dim_hidden))
            self.bias_gcn.data.normal_(-stdv_hidden, stdv_hidden)
        else:
            self.register_parameter('bias_gcn', None)  
        self.use_bn_gcn = use_bn_gcn
        self.bn_gcn = torch.nn.BatchNorm1d(dim_input)
        self.dropout_gcn = dropout_gcn

        # prediction
        self.prototypes = torch.nn.Parameter(torch.FloatTensor(dim_output, dim_hidden))
        stdv_output = 1./math.sqrt(dim_output)
        self.prototypes.data.normal_(-stdv_output, stdv_output)
        if bias_pre:
            self.bias_pre = torch.nn.Parameter(torch.FloatTensor(dim_output))
            self.bias_pre.data.normal_(-stdv_output, stdv_output)
        else:
            self.register_parameter('bias_pre', None)
        self.use_bn_pre = use_bn_pre
        self.bn_pre = torch.nn.BatchNorm1d(dim_hidden)
        self.dropout_pre = dropout_pre

    def forward(self, x, adj, t_p):
        # propagation
        x_ = x
        for hop in range(self.num_hop):
            x = torch.spmm(adj, x)
            x_ += x
        x_ = x_ / (self.num_hop+1)
        
        if self.use_bn_gcn:
            x_ = self.bn_gcn(x_)
        x_ = torch.nn.functional.dropout(x_, p=self.dropout_gcn, training=self.training)

        h = torch.mm(x_, self.weight_gcn)
        if self.bias_gcn is not None:
            h += self.bias_gcn

        h = torch.nn.functional.relu(h)

        # prediction
        if self.use_bn_pre:
            h_ = self.bn_pre(h)
        else:
            h_ = h
        h_ = torch.nn.functional.dropout(h_, p=self.dropout_pre, training=self.training)

        z = torch.mm(h_, self.prototypes.T)
        if self.bias_pre is not None:
            z = z + self.bias_pre
        z = torch.nn.functional.log_softmax(z/t_p, dim=1)
        return h, z