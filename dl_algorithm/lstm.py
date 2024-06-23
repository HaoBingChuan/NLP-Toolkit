#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Desc    : lstm for classifier
'''

import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, dl_config):
        super().__init__()
        if dl_config.embedding_pretrained == 'random':
            self.embedding = nn.Embedding(dl_config.vocab_size, dl_config.embedding_size,
                                          padding_idx=dl_config.vocab_size - 1)
        else:
            self.embedding = nn.Embedding.from_pretrained(dl_config.embedding_matrix, freeze=False,
                                                          padding_idx=dl_config.vocab_size - 1)
        self.lstm = nn.LSTM(dl_config.embedding_size, dl_config.hidden_size, batch_first=True, bidirectional=True,
                            dropout=dl_config.dropout)
        self.fc1 = nn.Linear(dl_config.hidden_size * 2, dl_config.hidden_size)
        self.fc2 = nn.Linear(dl_config.hidden_size, dl_config.nums_label)
        self.dropout = nn.Dropout(p=dl_config.dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embeding_size]
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x[:, -1, :]
