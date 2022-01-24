import torch
import torch.nn as nn
import torch.nn.functional as F

from network.model.relation_reasoning.GraphAttentionLayer import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout, alpha, num_heads=1):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(in_feature, hidden, dropout=dropout, alpha=alpha, concat=True) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hidden * num_heads, out_feature, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x))
        return F.log_softmax(x, dim=1)