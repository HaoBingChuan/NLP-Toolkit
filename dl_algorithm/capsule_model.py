#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Squash(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        # 防止除数为0
        self.epsilon = epsilon

    def forward(self, x):
        # x: [batch_size, nums_capsules, n_features]
        s2 = (x ** 2).sum(dim=-1, keepdim=True)
        return (s2 / (1 + s2)) * (x / torch.sqrt(s2 + self.epsilon))


class Router(nn.Module):
    def __init__(self, in_d, out_d, iterations=3):
        """
        Args:

            in_d (int): per capsues features, paper set 8
            out_d (int): 4*4=16 in paper
            iterations (int): Cij更新迭代的次数，论文里说3次就可以了
        """
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()

    def forward(self, nums_caps, out_caps, x):
        #    nums_caps (int): nums of capsules
        #    out_caps (int): unique labels
        # x: [batch_size, nums_capsules, n_features]
        # [1152,10,8,16]*[64,1152,8] -> [64,1152,10,16]

        # init Wij
        # [1152,10,8,16]
        self.w = nn.Parameter(torch.randn(nums_caps, out_caps, self.in_d, self.out_d))

        u_hat = torch.einsum('ijnm,bin->bijm', self.w, x)

        # init bij --> zero [batch, nums_capsules, out_caps] [64, 1152,10]
        b = x.new_zeros(x.shape[0], nums_caps, out_caps)
        v = None

        for i in range(self.iterations):
            c = self.softmax(b)  # [64, 1152, 16]
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            v = self.squash(s)
            a = torch.einsum('bjm,bijm->bij', v, u_hat)
            b = b + a
        return v


class MarginLoss(nn.Module):
    def __init__(self, lambda_=0.5, m1=0.9, m2=0.1):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.lambda_ = lambda_

    def forward(self, v, labels):
        # v: [batch_size, out_caps, out_d] 64,10,16 there is a capsule for each label
        # labels : [batch_size]
        n_labels = v.shape[1]
        v_norm = torch.sqrt(v)  # [batch_size, out_caps]
        labels = torch.eye(n_labels, device=labels.device)[labels]  # [batch_size, out_caps]
        loss = labels * F.relu(self.m1 - v_norm) + self.lambda_ * (1.0 - labels) * F.relu(v_norm - self.m2)
        return loss.sum(dim=-1).mean()


class CapsuleModel(nn.Module):
    def __init__(self, dl_config):
        super().__init__()
        if dl_config.embedding_pretrained == 'random':
            self.embedding = nn.Embedding(dl_config.vocab_size, dl_config.embedding_size,
                                          padding_idx=dl_config.vocab_size - 1)
        else:
            self.embedding = nn.Embedding.from_pretrained(dl_config.embedding_matrix, freeze=False,
                                                          padding_idx=dl_config.vocab_size - 1)
        self.in_d = dl_config.in_d
        self.out_d = dl_config.out_d
        self.nums_label = dl_config.nums_label
        self.reshape_num = dl_config.reshape_num
        self.conv1 = nn.Conv2d(1, 256, (2, dl_config.embedding_size), stride=1, padding=dl_config.pad_size)
        self.conv2 = nn.Conv2d(256, self.reshape_num * self.in_d, (2, 1), stride=2, padding=dl_config.pad_size)
        self.squash = Squash()
        self.digit_capsules = Router(self.in_d, self.out_d, dl_config.iter)

    def forward(self, data):
        x = self.embedding(data)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)

        caps = x.view(x.shape[0], self.in_d, self.reshape_num * x.shape[-1] * x.shape[-2]).permute(0, 2, 1)
        caps = self.digit_capsules(caps.shape[1], self.nums_label, caps)

        # pre = (caps ** 2).sum(-1).argmax(-1)
        return (caps ** 2).sum(-1)
