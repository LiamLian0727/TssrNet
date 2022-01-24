import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):

        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.liner = nn.Parameter(torch.empty(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.liner.data,gain=1.414)

    def forward(self, node, edge):

        # (N,in_features) @ (in_features, out_features) -> (N, out_features)
        node_feature = torch.mm(node, self.W)
        N = node.shape[0]

        #up : (N, out_features) @ (out_features, 1) -> (N, 1)
        up = torch.matmul(node_feature, self.a[:self.out_features, :])

        # down : (N, out_features) @ (out_features, 1) -> (N, 1)
        down = torch.matmul(node_feature, self.a[self.out_features:, :])

        #attention: (N, N)
        attention = torch.mul(1 - torch.eye(N), self.leakyrelu(up + down.T))
        attention += -9e15 * torch.eye(N)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        #out_node: (N, 2*out_features) @ (2*out_features, out_features)->(N, out_features)
        out_node = torch.cat(torch.matmul(attention, node_feature),node_feature)
        out_node = torch.mm(out_node,self.liner)

        if self.concat:
            return F.elu(out_node)
        else:
            return out_node
