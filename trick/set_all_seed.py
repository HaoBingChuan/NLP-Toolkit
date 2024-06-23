#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Description    : 固定所有的随机数种子
'''

import os
import torch
import numpy as np
import random
from torch.backends import cudnn


# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
