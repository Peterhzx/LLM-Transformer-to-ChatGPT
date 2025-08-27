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
    # MultiHeadAttention only for encoder
    def __init__(self, num_heads, emb_dim):
        super(MultiHeadAttention, self).__init__()
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        head_dim = emb_dim // num_heads
        self.projection_q = nn.Parameter(torch.empty([num_heads, emb_dim, head_dim]))
        nn.init.xavier_uniform_(self.projection_q)
        self.projection_k = nn.Parameter(torch.empty([num_heads, emb_dim, head_dim]))
        nn.init.xavier_uniform_(self.projection_k)
        self.projection_v = nn.Parameter(torch.empty([num_heads, emb_dim, head_dim]))
        nn.init.xavier_uniform_(self.projection_v)
        self.scaled_dot_product = ScaledDotProduct()
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, q, k, v, key_padding_mask):
        q_proj = torch.einsum("bse,hed->bshd", q, self.projection_q)
        k_proj = torch.einsum("bse,hed->bshd", k, self.projection_k)
        v_proj = torch.einsum("bse,hed->bshd", v, self.projection_v)

        out = self.scaled_dot_product(q_proj, k_proj, v_proj, key_padding_mask)
        out = out.view(-1)
        out = self.linear(out)

        return out

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super(ScaledDotProduct, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v, key_padding_mask):  # positional mask in decoder is performed before softmax
        # q, k, v.shape [batch_size, seq_len, num_head, head_dim]
        scale = torch.sqrt(k.size(-1))
        k_t = k.permute(0, 2, 3, 1)
        q = q.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # key_padding_mask.shape [batch_size, seq_len]
        key_padding_mask = key_padding_mask == False
        key_padding_mask = key_padding_mask.unsqueeze(-1)
        score = (q @ k_t) / scale  # score.shape [batch_size, num_head, seq_len, seq_len]
        score = score.masked_fill(key_padding_mask, float('-inf'))  # True for mask
        score = self.softmax(score)
        out = score @ v
        out = out.permute(0, 2, 1, 3)
        # out.shape [batch_size, seq_len, num_head, head_dim]
        return out

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
