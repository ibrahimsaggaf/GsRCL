'''
This code is modefied based on the impelemtation by 
https://github.com/ciortanmadalina/contrastive-sc
'''

import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, batchnorm, activation=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batchnorm = batchnorm
        self.activation = activation

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        if self.batchnorm:
            self.norm = nn.BatchNorm1d(self.out_dim)
        
        if self.activation:
            self.relu = nn.ReLU()


    def forward(self, x):
        x = self.linear(x)
        if self.batchnorm:
            x = self.norm(x)

        if self.activation:
            x = self.relu(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_dim, dim_enc, in_dim_proj, dim_proj, out_dim, batchnorm, 
                 dropout=None, frozen=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.dim_enc = dim_enc
        self.in_dim_proj = in_dim_proj
        self.dim_proj = dim_proj
        self.out_dim = out_dim
        self.batchnorm = batchnorm
        self.frozen = frozen
        
        layers = [
            LinearLayer(self.in_dim, self.dim_enc, batchnorm=self.batchnorm),
            LinearLayer(self.dim_enc, self.dim_enc, batchnorm=self.batchnorm),
            LinearLayer(self.dim_enc, self.dim_enc, batchnorm=self.batchnorm),
            LinearLayer(self.dim_enc, self.in_dim_proj, batchnorm=False, activation=False)
        ]
        if dropout is not None:
            layers.insert(0, nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)
        if not self.frozen:
            self.proj_head = nn.Sequential(
                LinearLayer(self.in_dim_proj, self.dim_proj, batchnorm=self.batchnorm),
                LinearLayer(self.dim_proj, self.out_dim, batchnorm=False, activation=False)
            )


    def forward(self, x):
        hiddens = self.encoder(x)
        if not self.frozen:
            proj = self.proj_head(hiddens)

            return proj

        return hiddens
