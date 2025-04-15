import torch
import sys
sys.path.append('/home/wu_tian_ci/GAFL')

from myutils.config import *
import argparse

from torch import nn as nn
# encoder_layer = nn.TransformerEncoderLayer(
#     d_model=5,
#     nhead=1,
#     dim_feedforward=5,
#     dropout=0.0,
#     activation="gelu",
#     batch_first=True
# )
# tfm = nn.TransformerEncoder(
#     encoder_layer=encoder_layer, num_layers=1
# )
# a=torch.randn(1,5,5)
# print(a)
# mask=torch.tensor([[0.,float('-inf'),0.,0.,0.]])
# mask2=torch.tensor([[False,True,False,False,False]])
# ans=tfm(a,
# src_key_padding_mask=mask)
# ans2=tfm(a)
# print(ans)
# print(ans2)
# ans3=tfm(a,src_key_padding_mask=mask2)
# print(ans3)

import math
def prob_at_least_one(N, m, k):
    return 1 - (math.comb(N - m, k) / math.comb(N, k))

# 示例：总共157个物体，想要的物体有5个，抽16个
N = 157
m = 1
k = 16

p = prob_at_least_one(N, m, k)
print(f"拿到至少一个想要的概率: {p:.4f}")