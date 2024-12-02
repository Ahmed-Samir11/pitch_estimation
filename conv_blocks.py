#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Peter Wu
#  Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)

"""Transformer model.

Modified from https://github.com/dgaddy/silent_speech.
"""

import logging
import numpy as np
import random
import torch
import torch.nn.functional as F

from torch import nn

class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)

class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Args:
            node_feats: Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats


class GCN(nn.Module):
    def __init__(self, in_channels = 2, hidden_dim = 16, out_channels = 1, dropout = 0.2):
        super(GCN, self).__init__()

        self.gc1 = GCNLayer(in_channels, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, out_channels)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

# Original conv blocks: no dropout, 2 hidden-hidden blocks.
class Original_GCN(nn.Module):
    def __init__(self, in_channels=8, hidden_dim = 1024, conv_dropout = 0.0, num_blocks = 2):
        super().__init__()

        self.gcn = GCN(in_channels=2, hidden_dim=32, out_channels=1, dropout=0.1)
        self.conv_blocks = nn.Sequential(
            ResBlock(in_channels, hidden_dim, 1),
            *[nn.Sequential(ResBlock(hidden_dim, hidden_dim, 1), 
                            nn.Dropout(conv_dropout)) for i in range(num_blocks)]
        )

    def forward(self, x):
        """
        Args:
            x: shape (batchsize, num_in_feats, seq_len).
        
        Return:
            out: shape (batchsize, num_out_feats, seq_len).
        """
        orig_x = x

        # Graph convolution part.
        # (B, C, T) 
        B, C, T = x.shape

        # (B, T, C) => (B*T, 2, C / 2) => (B * T, num points, 2)
        x = x.transpose(1, -1).reshape(B * T, 2, int(C / 2)).transpose(1, -1)
        distances = torch.cdist(x, x, p = 1)

        # (B * T, C / 2, 1)
        x = self.gcn(x, distances)

        # (B * T, C / 2, 1) => (B, C / 2, T)
        # Note that the number of channels here is halved.
        x = x.squeeze(-1).reshape(B, T, -1).transpose(1, -1)

        # Concatenate with original input.
        orig_x = torch.concatenate((orig_x, x), dim = 1)
        return self.conv_blocks(orig_x)


# Original conv blocks: no dropout, 2 hidden-hidden blocks.
class Original(nn.Module):
    def __init__(self, in_channels=8, hidden_dim = 1024, conv_dropout = 0.0, num_blocks = 2):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(in_channels, hidden_dim, 1),
            *[nn.Sequential(ResBlock(hidden_dim, hidden_dim, 1), 
                            nn.Dropout(conv_dropout)) for i in range(num_blocks)]
        )

    def forward(self, x):
        """
        Args:
            x: shape (batchsize, num_in_feats, seq_len).
        
        Return:
            out: shape (batchsize, num_out_feats, seq_len).
        """
        return self.conv_blocks(x)

# Original conv blocks: no dropout, 2 hidden-hidden blocks.
class Baseline(nn.Module):
    def __init__(self, in_channels=8, hidden_dim = 768, conv_dropout = 0.0, num_blocks = 2):
        super().__init__()

        self.conv_blocks = nn.Sequential(ResBlock(in_channels, hidden_dim, 1),
                                         ResBlock(hidden_dim, hidden_dim, 2),
                                         ResBlock(hidden_dim, hidden_dim, 2))

    def forward(self, x):
        """
        Args:
            x: shape (batchsize, num_in_feats, seq_len).
        
        Return:
            out: shape (batchsize, num_out_feats, seq_len).
        """
        return self.conv_blocks(x)
   
# Downsample by 8x. 
class Baseline_v2(nn.Module):
    def __init__(self, in_channels=8, hidden_dim = 768, conv_dropout = 0.0, num_blocks = 2):
        super().__init__()

        self.conv_blocks = nn.Sequential(ResBlock(in_channels, hidden_dim, 1),
                                         ResBlock(hidden_dim, hidden_dim, 2),
                                         ResBlock(hidden_dim, hidden_dim, 2), 
                                         ResBlock(hidden_dim, hidden_dim, 2))

    def forward(self, x):
        """
        Args:
            x: shape (batchsize, num_in_feats, seq_len).
        
        Return:
            out: shape (batchsize, num_out_feats, seq_len).
        """
        return self.conv_blocks(x) 
    
# Test convblocks for mri-2-f256.
class HandCraft_v1(nn.Module):
    def __init__(self, in_channels=8, hidden_dim = 1024, conv_dropout = 0.0, num_blocks = 2):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(in_channels, 512, 1),
            ResBlock(512, 512, 1),
            ResBlock(512, 1024, 1),
            *[nn.Sequential(ResBlock(hidden_dim, hidden_dim, 1), 
                            nn.Dropout(conv_dropout)) for i in range(num_blocks)]
        )

    def forward(self, x):
        """
        Args:
            x: shape (batchsize, num_in_feats, seq_len).
        
        Return:
            out: shape (batchsize, num_out_feats, seq_len).
        """
        return self.conv_blocks(x)
    

# Pooled conv blocks for EMG. 
# Original conv blocks: no dropout, 2 hidden-hidden blocks.
class Pool(nn.Module):
    def __init__(self, in_channels=8, hidden_dim = 1024, conv_dropout = 0.2, num_blocks = 4):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(in_channels, 256, 1),
            ResBlock(256, 512, 1),
            nn.AvgPool1d(2),
            ResBlock(512, hidden_dim, 1),
            nn.AvgPool1d(2),
            *[nn.Sequential(ResBlock(hidden_dim, hidden_dim, 1), 
                            nn.Dropout(conv_dropout)) for i in range(num_blocks)]
        )

    def forward(self, x):
        """
        Args:
            x: shape (batchsize, num_in_feats, seq_len).
        
        Return:
            out: shape (batchsize, num_out_feats, seq_len).
        """
        return self.conv_blocks(x)