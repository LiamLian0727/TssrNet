import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, graph, in_feature_dim, out_feature_dim):
        super(GATLayer, self).__init__()
        self.g = graph
        gain = nn.init.calculate_gain('relu')
        self.linear = nn.Linear(in_feature_dim, out_feature_dim, bias=False)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        self.attention = nn.Linear(2 * out_feature_dim, 1, bias=False)
        nn.init.xavier_normal_(self.attention.weight, gain=gain)

    def getAttention(self, edges):
        z_cat = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        attentions = self.attention(z_cat)
        return {'e': F.leaky_relu(attentions)}

    def doMessage(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def doReduce(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.linear(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.getAttention)
        self.g.update_all(self.doMessage, self.doReduce)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, graph, in_feature_dim, out_feature_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(graph, in_feature_dim, out_feature_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [att(h) for att in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_feature_dim, hidden_dim, out_feature_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_feature_dim, hidden_dim[0], num_heads[0], merge='cat')
        self.layer2 = MultiHeadGATLayer(g, hidden_dim[0] * num_heads[0], hidden_dim[1], num_heads[1], merge='cat')
        self.layer3 = MultiHeadGATLayer(g, hidden_dim[1] * num_heads[1], hidden_dim[2], num_heads[2], merge='cat')
        self.layer4 = MultiHeadGATLayer(g, hidden_dim[2] * num_heads[2], hidden_dim[3], num_heads[3], merge='cat')
        self.layer5 = MultiHeadGATLayer(g, hidden_dim[3] * num_heads[3], out_feature_dim, 1, merge='cat')

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        h = F.elu(h)
        h = self.layer3(h)
        h = F.elu(h)
        h = self.layer4(h)
        h = F.elu(h)
        h = self.layer5(h)
        return h
