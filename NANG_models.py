from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from layers import GCNLayer, GATLayer
from torchvision import datasets, transforms
from torchvision.utils import save_image

class LFI(nn.Module):
    def __init__(self, n_nodes, n_fts, n_hid, dropout, args):
        super(LFI, self).__init__()
        self.n_fts = n_fts
        self.n_hid = n_hid
        self.dropout = dropout


        # encoder for ae
        self.ae_fc1 = nn.Linear(n_fts, 200)
        self.ae_fc2 = nn.Linear(200, n_hid)

        # encoder for gae
        if args.enc_name == 'GCN':
            self.GCN1 = GCNLayer(n_nodes, 200, dropout=dropout)
            self.GCN2 = GCNLayer(200, n_hid, dropout=dropout)
        elif args.enc_name=='GAT':
            self.GCN1 = GATLayer(n_nodes, 200, dropout=dropout, alpha=args.alpha)
            self.GCN2 = GATLayer(200, n_hid, dropout=dropout, alpha=args.alpha)

        # specific decoder for fts
        self.G_ae_fc1 = nn.Linear(n_hid, 200)
        self.G_ae_fc2 = nn.Linear(200, n_fts)

        # specific decoder for gae
        self.G_gae_fc1 = nn.Linear(n_hid, n_hid)
        self.G_gae_fc2 = nn.Linear(n_hid, n_hid)

        self.disc = Discriminator(n_hid, n_hid, dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode_fts(self, z):
        # get specific fts
        fts1 = F.relu(self.G_ae_fc1(z))
        fts1 = F.dropout(fts1, self.dropout, training=self.training)
        fts2 = self.G_ae_fc2(fts1)
        return fts2

    def decode_adj(self, z):
        # get specific adj
        adj_z1 = F.relu(self.G_gae_fc1(z))
        adj_z1 = F.dropout(adj_z1, self.dropout, training=self.training)
        adj_z2 = self.G_gae_fc2(adj_z1)

        return torch.mm(adj_z2, torch.transpose(adj_z2, 0, 1))

    def forward(self, x, adj, diag_fts):
        # make inference for AE
        x = F.dropout(x, self.dropout, training=self.training)
        ae_h1 = F.relu(self.ae_fc1(x))
        ae_h1 = F.dropout(ae_h1, self.dropout, training=self.training)
        self.ae_z = self.ae_fc2(ae_h1)

        # make inference for GAE
        gae_h1 = self.GCN1(diag_fts, adj, is_sparse_input=True)
        gae_h1 = F.dropout(gae_h1, self.dropout, training=self.training)
        self.gae_z = self.GCN2(gae_h1, adj)

        ae_fts = self.decode_fts(self.ae_z)
        gae_fts = self.decode_fts(self.gae_z)
        ae_adj = self.decode_adj(self.ae_z)
        gae_adj = self.decode_adj(self.gae_z)

        return self.ae_z, ae_fts, ae_adj, self.gae_z, gae_fts, gae_adj

class Discriminator(nn.Module):
    def __init__(self, n_fts, n_hid, dropout):
        super(Discriminator, self).__init__()
        self.dropout = dropout

        self.fc1 = nn.Linear(n_fts, n_hid)
        self.fc2 = nn.Linear(n_hid, 1)

    def forward(self, x):
        # make mlp for discriminator
        h1 = self.fc1(x)
        h1 = F.dropout(F.relu(h1), self.dropout, training=self.training)
        h2 = self.fc2(h1)
        return h2