import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('/home/wu_tian_ci/GAFL')
from myutils.common import *
import scipy.sparse as sp
from torch import nn
import numpy as np
from myutils.common import *
import einops
from myutils.dynamic import *
from myutils.config import *
from torch_geometric.data import Data,Batch
from gpnn.gpnnmodel import GPNN
from .common import TwoLayer

class OneStream(nn.Module):
    def __init__(self,config,branch_type='TS'):
        super().__init__()
        self.branch_type=branch_type
        self.encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.transformer.heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )

        self.encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.transformer.heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.temporal = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer1, num_layers=config.tse.time.layer
        )
        self.spatial = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer2, num_layers=config.tse.spatial.layer
        )

        self.temporal_mlp=TwoLayer(config.dims,config.dropout,eps=config.eps)
        self.spatial_mlp=TwoLayer(config.dims,config.dropout,eps=config.eps)
        self.f=config.frames
        self.a=config.actors
        self.d=config.dims
    # batch,frame,actors,dims
    # come in/out  batch  frame, actors, dims 
    def forward(self,X,t_embed,s_embed):
        if self.branch_type=='TS':
            X=einops.rearrange(X,'b f a d -> (b a) f d')
            t_x=self.temporal_mlp(self.temporal(X))+X
            t_x=einops.rearrange(t_x,' (b a) f d -> (b f ) a d',a=self.a,f=self.f)
            s_x=self.spatial_mlp(self.spatial(t_x))+t_x
            x=einops.rearrange(s_x,'(b f) a d -> b f a d',f=self.f)
        elif self.branch_type=='ST':
            X=einops.rearrange(X,'b f a d -> (b f) a d')
            s_x=self.spatial_mlp(self.spatial(X))+X
            s_x=einops.rearrange(s_x,' (b f) a d -> (b a) f d',a=self.a,f=self.f)
            t_x=self.temporal_mlp(self.temporal(s_x))+s_x
            x=einops.rearrange(t_x,' (b a) f d -> b f a d',a=self.a,f=self.f)

        return x


class TSE(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.tsb=OneStream(config,'TS')
        self.stb=OneStream(config,'ST')
        self.fusion=TwoLayer(config.dims,config.dropout,eps=config.eps)
        
    # come in/out  batch  frame, actors, dims 
    def forward(self,X):
        st_x=self.stb(X)
        ts_x=self.tsb(X)
        return self.fusion(st_x+ts_x)


