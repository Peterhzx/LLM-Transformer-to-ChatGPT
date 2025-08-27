import math

import torch
from torch import nn

import torch.nn.functional as F


class BERT(nn.Module):
    def __init__(self, num_emb, emb_dim, pad_token_id, num_layer):
        super(BERT, self).__init__()
        self.embeddings = nn.Embedding(num_emb, emb_dim, pad_token_id)
        self.encoder = nn.ModuleList([EncoderLayer() for _ in range(num_layer)])


class EncoderLayer(nn.Module):


class MultiHeadAttention(nn.Module):


class ScaledDotProduct(nn.Module):


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta  = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
