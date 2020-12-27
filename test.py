from os import read
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
from torch.nn.modules.activation import ReLU

from set2set import Set2Set

from encoders import SoftPoolingGcnEncoder

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            #print(y[0][0])
        return y

class GConvModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers, 
    pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, normalize=False, num_aggs=1, 
    args=None):

        super(GConvModule, self).__init__()

        add_self = not concat

        self.conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
        normalize_embedding=normalize, bias=True)

        self.conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=True) 
                 for i in range(num_layers-2)])

        self.conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=True)

        
        self.act = nn.ReLU()
        self.bn = bn
        self.num_aggs = num_aggs
        self.concat = concat

        if concat:
            pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            pred_input_dim = embedding_dim
        
        pred_input_dim = pred_input_dim * num_aggs

        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        
        self.pred_block = pred_model

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x, adj, embedding_mask=None):

        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]

        for i in range(len(self.conv_block)):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = self.conv_last(x,adj)
        x_all.append(x)

        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)

        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask

        ypred = self.pred_block(x_tensor)
        
        return ypred


if __name__=='__main__':
    x = torch.rand(20, 100, 10).cuda()
    adj = torch.rand(20, 100, 100).cuda()

    #net = GConvModule(10, 10, 5, 3, num_layers=5)
    net = SoftPoolingGcnEncoder(100, 10, 10, 5, 3, 5, 5, num_pooling=2, 
    assign_ratio=0.1, num_unpooling=2, unpool_ratio=0.1)
    net = net.cuda()

    a = net.forward(x, adj, range(20), True, True)
    print(a)

    '''
    from load_data import read_graphfile
    g = read_graphfile('./data', 'DND')
    print(len(g))
    print(g)
    '''

