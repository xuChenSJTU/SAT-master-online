from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class VAE(nn.Module):
    def __init__(self, n_fts, n_hid, dropout):
        super(VAE, self).__init__()
        self.n_fts = n_fts
        self.n_hid = n_hid
        self.dropout = dropout

        self.fc1 = nn.Linear(n_fts, 200)
        self.fc21 = nn.Linear(200, n_hid)
        self.fc22 = nn.Linear(200, n_hid)
        self.fc3 = nn.Linear(n_hid, 200)
        self.fc4 = nn.Linear(200, n_fts)

    def encode(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.dropout(h3, self.dropout, training=self.training)
        return self.fc4(h3)

    def forward(self, x):
        self.mu, self.logvar = self.encode(x)
        self.z = self.reparameterize(self.mu, self.logvar)
        return self.decode(self.z), self.mu, self.logvar