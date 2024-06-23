#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import copy
import math
import torch
import torch.nn as nn


class PrepareForMultiHeadAttention(nn.Module):
    """生成Wq，Wk，Wv三个权重矩阵
    """

    def __init__(self, d_model, heads, d_k, bias):
        """
        Args:
            d_model (int): dim for model : 512
            heads (int): nums of attention head: 8
            d_k (int): dim for K
            bias (bool): bias for linear layer
        """
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x):
        # input_shape: [batch, seqlenth, d_model]
        head_shape = x.shape[:-1]
        x = self.linear(x)
        # reshape
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAtttention(nn.Module):
    """计算过程
    """

    def __init__(self, heads, d_model, dropout, bias=True):
        """
        Args:
            heads (int): 
            d_model (int):
            dropout (float): 
            bias (bool, optional):  Defaults to True.
        """
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None  # 主要用于画图或者输出debug

    def get_scores(self, query, key):
        # return [batch, seq_len, seq_len, heads]
        return torch.einsum('bihd,bjhd->bijh', query, key)

    def prepare_mask(self, mask, query_shape, key_shape):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        mask = mask.unsqueeze(-1)
        return mask

    def forward(self, query, key, value, mask=None):
        # 自注意力机制里，这里的query，key，value其实都是x
        batch_size, seq_len, _ = query.shape
        if mask:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        scores = self.get_scores(Q, K)
        scores *= self.scale
        if mask:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        x = torch.einsum('bijh,bjhd->bihd', attn, V)
        self.attn = attn.detach()
        x = x.reshape(batch_size, seq_len, -1)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class FeedForward(nn.Module):
    """FFN module
    """

    def __init__(self, d_model: int, hidden: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        """
        * `d_model` is the number of features in a token embedding
        * `hidden` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, hidden, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(hidden, d_model, bias=bias2)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, hidden, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # $f(x W_1 + b_1)$
        g = self.activation(self.layer1(x))
        # If gated, $f(x W_1 + b_1) \otimes (x V + b) $
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)
        # $(f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2$ or $f(x W_1 + b_1) W_2 + b_2$
        # depending on whether it is gated
        return self.layer2(x)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        """transformer layer

        Args:
            d_model (int): 
            self_attn (): multi-head-attention layer
            src_attn (): multi-head-attention layer
            feed_forward (): feed forwardd layer
            dropout (float): dropout prob
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm([d_model])

    def forward(self, x, mask):
        z = self.layernorm(x)
        self_attn = self.self_attn(z, z, z, mask)  # mha
        x = x + self.dropout(self_attn)  # add
        z = self.layernorm(x)  # norm
        ff = self.feed_forward(z)  # ff
        x = z + self.dropout(ff)  # add
        x = self.layernorm(x)  # norm
        return x


class Encoder(nn.Module):
    def __init__(self, layer, n_layers):
        """encoder layer

        Args:
            layer (): transformer layer
            n_layers (int): nums of layer: default 6
        """
        super().__init__()
        self.layers = self.clones(layer, n_layers)
        self.layernorm = nn.LayerNorm([layer.size])

    def clones(self, layer, N):
        return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

    def forward(self, x, mask):
        for l in self.layers:
            x = l(x, mask)
        return self.layernorm(x)


class TransformerModel(nn.Module):
    def __init__(self, dl_config):
        super().__init__()
        if dl_config.embedding_pretrained == 'random':
            self.embedding = nn.Embedding(dl_config.vocab_size, dl_config.embedding_size,
                                          padding_idx=dl_config.vocab_size - 1)
        else:
            self.embedding = nn.Embedding.from_pretrained(dl_config.embedding_matrix, freeze=False,
                                                          padding_idx=dl_config.vocab_size - 1)

        self.postion_embedding = PositionalEncoding(dl_config.d_model, dl_config.dropout)
        self.transformerlayer = TransformerLayer(d_model=dl_config.d_model,
                                                 self_attn=MultiHeadAtttention(dl_config.heads, dl_config.d_model,
                                                                               dl_config.dropout),
                                                 src_attn=None,
                                                 feed_forward=FeedForward(dl_config.d_model, dl_config.hidden,
                                                                          dl_config.dropout),
                                                 dropout=dl_config.dropout)

        self.encoder = Encoder(self.transformerlayer, dl_config.n_layers)
        self.fc1 = nn.Linear(dl_config.embedding_size, dl_config.nums_label)

    def forward(self, x):
        x = self.embedding(x)
        x = self.postion_embedding(x)
        x = self.encoder(x, mask=None)
        x = self.fc1(x)
        return x[:, -1, :]
