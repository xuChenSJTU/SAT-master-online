import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN_layers import GCNLayer
from GAT_layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

        self.fc1 = nn.Linear(nhid, 200)
        self.fc2 = nn.Linear(200, nclass)

    def forward(self, x, adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        h1 = torch.cat([att(x, adj, is_fts_sparse=True) for att in self.attentions], dim=1)
        h1 = F.dropout(h1, self.dropout, training=self.training)
        h2 = self.out_att(h1, adj)

        h3 = F.elu(self.fc1(h2))
        h3 = F.dropout(h3, self.dropout, training=self.training)

        h4 = self.fc2(h3)

        return h4

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        """Dense version of GAT."""
        super(GCN, self).__init__()
        self.dropout = dropout
        self.GCNlayer1 = GCNLayer(nfeat, nhid, dropout=dropout)
        self.GCNlayer2 = GCNLayer(nhid, nhid, dropout=dropout)

        self.fc1 = nn.Linear(nhid, 200)
        self.fc2 = nn.Linear(200, nclass)

    def forward(self, x, sp_adj):
        h1 = self.GCNlayer1(x, sp_adj, is_sp_fts=True)
        h1 = F.dropout(h1, self.dropout, training=self.training)
        self.z = self.GCNlayer2(h1, sp_adj, is_sp_fts=False)

        h3 = F.elu(self.fc1(self.z))
        h3 = F.dropout(h3, self.dropout, training=self.training)

        h4 = self.fc2(h3)

        return h4

class GCN_eva(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, input_fts_sparse=True):
        """Dense version of GAT."""
        super(GCN_eva, self).__init__()
        self.dropout = dropout
        self.GCNlayer1 = GCNLayer(nfeat, nhid, dropout=dropout)
        self.GCNlayer2 = GCNLayer(nhid, nhid, dropout=dropout)
        self.input_fts_sparse = input_fts_sparse

        self.fc1 = nn.Linear(nhid, nclass)

    def forward(self, x, sp_adj):
        h1 = self.GCNlayer1(x, sp_adj, is_sp_fts=self.input_fts_sparse)
        h1 = F.dropout(h1, self.dropout, training=self.training)
        self.z = self.GCNlayer2(h1, sp_adj, is_sp_fts=False)

        h3 = F.log_softmax(self.fc1(self.z), dim=1)

        return h3


