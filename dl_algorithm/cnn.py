#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, dl_config):
        super().__init__()
        if dl_config.embedding_pretrained == 'random':
            self.embedding = nn.Embedding(dl_config.vocab_size, dl_config.embedding_size,
                                          padding_idx=dl_config.vocab_size - 1)
        else:
            self.embedding = nn.Embedding.from_pretrained(dl_config.embedding_matrix, freeze=False,
                                                          padding_idx=dl_config.vocab_size - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, dl_config.nums_filters, (k, dl_config.embedding_size), stride=dl_config.stride,
                       padding=dl_config.pad_size) for k in dl_config.filter_size]
        )
        self.dropout = nn.Dropout(p=dl_config.dropout)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(dl_config.nums_filters * len(dl_config.filter_size), dl_config.nums_label)

    def conv_and_pool(self, x, conv):
        x = self.relu(conv(x))
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # 增加通道数为1
        x = [self.conv_and_pool(x, conv) for conv in self.convs]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
