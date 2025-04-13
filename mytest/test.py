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
a=~torch.tensor([0.,1.,0.,0.],dtype=torch.bool)
print(a)
a=torch.tensor([[False,True],[True,True]])
print(a)
print(a.shape)
mask=torch.all(a==True,dim=-1)
a[mask]=False
print(a)