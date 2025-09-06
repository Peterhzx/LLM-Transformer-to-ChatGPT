import math

import torch
from torch import nn

import torch.nn.functional as F


class BERT(nn.Module):
    def __init__(self, num_tokens, embed_dim, pad_token_id, num_layers, num_heads, dropout, feedforward_dim=2048, max_possible_seq_len=2048, **kwargs):
        super(BERT, self).__init__()
        if kwargs:
            print(f"Ignored unused arguments: {', '.join(kwargs.keys())}")
        self.padding_idx = pad_token_id
        self.register_buffer('positional_encoding', self.build_positional_encoding(max_possible_seq_len, embed_dim))
        self.embeddings = nn.Embedding(num_tokens, embed_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([EncoderLayer(num_heads, embed_dim, feedforward_dim, dropout) for _ in range(num_layers)])
        self.nsp_linear = nn.Linear(embed_dim, 2)

    def forward(self, x, key_padding_mask=None):
        if key_padding_mask is None:
            key_padding_mask = (x == self.padding_idx)
        key_padding_mask = key_padding_mask.to(x.device)
        assert x.size(1) <= self.positional_encoding.size(0), \
            "encoder_input is longer than the precomputed positional encoding"
        enc_emb = self.embeddings(x) + self.positional_encoding[:x.size(1), :].unsqueeze(0).to(x.device)
        enc_emb = self.dropout(enc_emb)
        enc_output = enc_emb
        for i, layer in enumerate(self.encoder):
            enc_output = layer(enc_output, key_padding_mask)
        nsp_out = enc_output[:, 0, :]
        nsp_out = self.nsp_linear(nsp_out)
        out = enc_output @ self.embeddings.weight.T  # [batch_size, seq_len, num_emb]
        return out, nsp_out

    @staticmethod
    def build_positional_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, emb_dim, feedforward_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads, emb_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_1 = LayerNorm(emb_dim)
        self.feedforward_1 = nn.Linear(emb_dim, feedforward_dim)
        self.feedforward_2 = nn.Linear(feedforward_dim, emb_dim)
        self.norm_2 = LayerNorm(emb_dim)

    def forward(self, x, key_padding_mask):
        out = self.multi_head_attention(x, x, x, key_padding_mask)
        out += x
        _out = self.norm_1(out)
        out = self.feedforward_1(_out)
        out = F.gelu(out)
        out = self.dropout1(out)
        out = self.feedforward_2(out)
        out = self.dropout2(out)
        out += _out
        out = self.norm_2(out)
        return out


class MultiHeadAttention(nn.Module):
    # MultiHeadAttention only for encoder
    def __init__(self, num_heads, emb_dim, dropout):
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask):
        q_proj = torch.einsum("bse,hed->bshd", q, self.projection_q)
        k_proj = torch.einsum("bse,hed->bshd", k, self.projection_k)
        v_proj = torch.einsum("bse,hed->bshd", v, self.projection_v)

        out = self.scaled_dot_product(q_proj, k_proj, v_proj, key_padding_mask)
        batch_size, seq_len, num_heads, head_dim = out.size()
        out = out.reshape(batch_size, seq_len, num_heads * head_dim)
        out = self.linear(out)
        out = self.dropout(out)

        return out


class ScaledDotProduct(nn.Module):
    def __init__(self):
        super(ScaledDotProduct, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v, key_padding_mask):  # positional mask in decoder is performed before softmax
        # q, k, v.shape [batch_size, seq_len, num_head, head_dim]
        scale = math.sqrt(k.size(-1))
        k_t = k.permute(0, 2, 3, 1)
        q = q.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # key_padding_mask.shape [batch_size, seq_len]
        key_padding_mask = key_padding_mask[:, None, None, :]
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
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
