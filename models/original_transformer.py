import math

import torch
from torch import nn

import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, num_tokens, pad_token_id, feedforward_dim=2048, max_possible_seq_len=2048, **kwargs):
        super(Transformer, self).__init__()
        if kwargs:
            print(f"Ignored unused arguments: {', '.join(kwargs.keys())}")
        self.padding_idx = pad_token_id
        self.register_buffer('positional_encoding', self.build_positional_encoding(max_possible_seq_len, embed_dim))
        self.embeddings = nn.Embedding(num_tokens, embed_dim, padding_idx=pad_token_id)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embed_dim, num_heads, feedforward_dim) for _ in range(num_layers)])
        self.encoder_cache = None
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(embed_dim, num_heads, feedforward_dim) for _ in range(num_layers)])

    def forward(self, encoder_input, decoder_input, src_key_padding_mask=None, decoder_input_key_padding_mask=None,
                with_cache=False):  # with_cache should not be assigned True in the first inference step
        if src_key_padding_mask is None:
            src_key_padding_mask = (encoder_input == self.padding_idx)
        if decoder_input_key_padding_mask is None:
            decoder_input_key_padding_mask = (decoder_input == self.padding_idx)
        src_key_padding_mask = src_key_padding_mask.to(decoder_input.device)
        decoder_input_key_padding_mask = decoder_input_key_padding_mask.to(decoder_input.device)
        if with_cache and self.encoder_cache is not None:  # inference with cached encoder output
            assert decoder_input.size(1) <= self.positional_encoding.size(0), \
                "decoder_input is longer than the precomputed positional encoding"
            dec_emb = self.embeddings(decoder_input) + self.positional_encoding[:decoder_input.size(1), :].unsqueeze(
                0).to(decoder_input.device)
            self.encoder_cache = self.encoder_cache.to(decoder_input.device)
            dec_output = dec_emb
            for layer in self.decoder_layers:
                dec_output = layer(dec_output, self.encoder_cache, decoder_input_key_padding_mask, src_key_padding_mask)
            out = dec_output @ self.embeddings.weight.T
            return out
        else:
            self.encoder_cache = None
            assert encoder_input.size(1) <= self.positional_encoding.size(0), \
                "encoder_input is longer than the precomputed positional encoding"
            assert decoder_input.size(1) <= self.positional_encoding.size(0), \
                "decoder_input is longer than the precomputed positional encoding"
            enc_emb = self.embeddings(encoder_input) + self.positional_encoding[:encoder_input.size(1), :].unsqueeze(
                0).to(decoder_input.device)
            dec_emb = self.embeddings(decoder_input) + self.positional_encoding[:decoder_input.size(1), :].unsqueeze(
                0).to(decoder_input.device)
            enc_output = enc_emb
            for i, layer in enumerate(self.encoder_layers):
                enc_output = layer(enc_output, src_key_padding_mask)
                if torch.isnan(enc_output).any():
                    print(f"NaN in encoder layer {i}")
                    break
            if not self.training:
                self.encoder_cache = enc_output
            dec_output = dec_emb
            for i, layer in enumerate(self.decoder_layers):
                dec_output = layer(dec_output, enc_output, decoder_input_key_padding_mask, src_key_padding_mask)
                if torch.isnan(dec_output).any():
                    print(f"NaN in decoder layer {i}")
                    break
            out = dec_output @ self.embeddings.weight.T
            return out

    @staticmethod
    def build_positional_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim):
        super(EncoderLayer, self).__init__()
        self.multi_head = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.feedforward_1 = nn.Linear(embed_dim, feedforward_dim)
        self.feedforward_2 = nn.Linear(feedforward_dim, embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_key_padding_mask):
        if src_key_padding_mask is not None:
            fully_masked = src_key_padding_mask.all(dim=1)  # [batch]
            if fully_masked.any():
                for seq in src_key_padding_mask.tolist():
                    print(seq)
                print(f"Fully masked sequences: {fully_masked.sum().item()}")
        if torch.isnan(x).any():
            print("NaN in encoder input")
        out, _ = self.multi_head(x, x, x, key_padding_mask=src_key_padding_mask)
        if torch.isnan(out).any():
            print("NaN in encoder attention output")
            # Log magnitude statistics
            print(f"Max: {out.max().item()}, Min: {out.min().item()}")
        out = out + x
        MHA_out = self.norm_1(out)
        out = self.feedforward_1(MHA_out)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.feedforward_2(out)
        out = self.dropout2(out)
        out = out + MHA_out
        out = self.norm_2(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.feedforward_1 = nn.Linear(embed_dim, feedforward_dim)
        self.feedforward_2 = nn.Linear(feedforward_dim, embed_dim)
        self.norm_3 = nn.LayerNorm(embed_dim)

    def forward(self, decoder_input, encoder_output, decoder_input_key_padding_mask, src_key_padding_mask):
        if torch.isnan(decoder_input).any():
            print("NaN in decoder input")
        seq_len = decoder_input.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=decoder_input.device), diagonal=1)
        out, _ = self.self_attn(decoder_input, decoder_input, decoder_input, decoder_input_key_padding_mask,
                                attn_mask=attn_mask)
        if torch.isnan(out).any():
            print("NaN in decoder self attention output")
            # Log magnitude statistics
            print(f"Max: {out.max().item()}, Min: {out.min().item()}")
        out = out + decoder_input
        MMHA_out = self.norm_1(out)
        out, _ = self.cross_attn(MMHA_out, encoder_output, encoder_output, key_padding_mask=src_key_padding_mask)
        if torch.isnan(out).any():
            print("NaN in decoder cross attention output")
            # Log magnitude statistics
            print(f"Max: {out.max().item()}, Min: {out.min().item()}")
        out = out + MMHA_out
        MHA_out = self.norm_2(out)
        out = self.feedforward_1(MHA_out)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.feedforward_2(out)
        out = self.dropout2(out)
        out = out + MHA_out
        out = self.norm_3(out)
        return out
