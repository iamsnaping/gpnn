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
from gpnn2.gpnn2 import GPNN as GPNN2
from gpnn2.gpnn2 import GPNN2 as GPNN3
from gpnn2.gpnn2 import GPNNText as GPNNText
from gpnn2.gpnn2 import GPNN3 as GPNN4
from gpnn2.gpnn2 import GPNN4 as GPNN5
from myutils.gpfplus import (SimplePrompt,GPFPlus,SinglePrompt)

def self_count(model,x,node):
    x=torch.randn(1,16,768)
    node=torch.randn(16,768)
    model(x,node)




# batch,frame,actors,dim
class OneBranch(nn.Module):
    def __init__(self,config,branch_type='TS'):
        super().__init__()
        self.branch_type=branch_type
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.transformer.heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=config.transformer.layers
        )
        self.dgc=DynamicGCN(config)
        self.f=config.frames
        self.a=config.actors
        self.d=config.dims
    # batch,frame,actors,dims
    # come in/out graph-like data -> batch*frame*actors, dims
    def forward(self,X,edge_index):
        if self.branch_type=='TS':
            X=einops.rearrange(X,'(b f) a d -> (b a) f d',f=self.f,a=self.a)
            X=self.encoder(X)
            X=einops.rearrange(X,'(b a) f d -> (b f) a d',a=self.a)
            X,edge_index=self.dgc(X,edge_index)
        else:
            X,edge_index=self.dgc(X,edge_index)
            X=einops.rearrange(X,'(b f) a d -> (b a) f d',f=self.f,a=self.a)
            X=self.encoder(X)
            X=einops.rearrange(X,'(b a) f d -> (b f) a d',a=self.a,f=self.f)
        return X,edge_index

class OneBranchGCN(nn.Module):
    def __init__(self,config,branch_type='TS'):
        super().__init__()
        self.branch_type=branch_type
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.transformer.heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=config.transformer.layers
        )
        self.dgc=TinyGCN(768,768)
        self.f=config.frames
        self.a=config.actors
        self.d=config.dims
        self.s_mlp=TwoLayer(config.dims,config.dropout,config.eps)
        self.t_mlp=TwoLayer(config.dims,config.dropout,config.eps)
    # batch,frame,actors,dims
    # come in/out graph-like data -> batch*frame actors, dims 
    def forward(self,X,edge_index):
        if self.branch_type=='TS':
            X=einops.rearrange(X,'(b f) a d -> (b a) f d',f=self.f,a=self.a)
            X=self.t_mlp(self.encoder(X))
            X=einops.rearrange(X,'(b a) f d -> (b f a) d',a=self.a,f=self.f)
            X,edge_index=self.dgc(X,edge_index)
            X=einops.rearrange(X,'(b f a) d -> (b f) a d',a=self.a,f=self.f)
            X=self.s_mlp(X)
        else:
            X=einops.rearrange(X,'(b f) a d -> (b f a) d',f=self.f,a=self.a)
            X,edge_index=self.dgc(X,edge_index)
            X=self.s_mlp(X)
            X=einops.rearrange(X,'(b f a) d -> (b a) f d',f=self.f,a=self.a)
            X=self.t_mlp(self.encoder(X))
            X=einops.rearrange(X,'(b a) f d -> (b f) a d',a=self.a,f=self.f)
        return X,edge_index

class Fusion(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dgc=DynamicGCN(config)
    def forward(self,TS,ST):
        TS_x,TS_edge=TS
        ST_x,ST_edge=ST
        fusion=TS_x*0.5+ST_x*0.5
        edges=torch.unique(torch.cat([TS_edge,ST_edge],dim=1),dim=1)
        return self.dgc(fusion,edges)

class OneBranch_T(nn.Module):
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
            encoder_layer=self.encoder_layer1, num_layers=config.transformer.layers
        )
        self.spatial = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer2, num_layers=config.transformer.layers
        )
        
        self.f=config.frames
        self.a=config.actors+1
        self.d=config.dims
    # batch,frame,actors,dims
    # come in/out  batch  frame, actors, dims 
    def forward(self,X,t_embed,s_embed):
        if self.branch_type=='TS':
            X=einops.rearrange(X,'b f a d -> (b a) f d',f=self.f)
            X=self.temporal(X+t_embed)
            X=einops.rearrange(X,'(b a) f d -> (b f) a d',a=self.a)
            X=self.spatial(X+s_embed)
            X=einops.rearrange(X,'(b f) a d -> b f a d',f=self.f)
        else:
            X=einops.rearrange(X,'b f a d -> (b f) a d')
            X=self.spatial(X+s_embed)
            X=einops.rearrange(X,'(b f) a d -> (b a) f d',f=self.f)
            X=self.temporal(X+t_embed)
            X=einops.rearrange(X,'(b a ) f d -> b f a d',a=self.a)
        return X

class Branches_T(nn.Module):
    def __init__(self, config,branch_type,nums):
        super().__init__()
        self.list=nn.ModuleList()
        for i in range(nums):
            self.list.append(OneBranch_T(config,branch_type))
        self.t_embed = nn.Parameter(torch.zeros(1,16,1, 768))
        self.s_embed=nn.Parameter(torch.zeros(1,1,11, 768))
        nn.init.xavier_uniform_(self.t_embed)
        nn.init.xavier_uniform_(self.s_embed)
    

    def forward(self,x):
        B,Frame,Nums,Dims=x.shape
        t_embed=self.t_embed.repeat(B,1,Nums,1)
        s_embed=self.s_embed.repeat(B,Frame,1,1)
        t_embed=einops.rearrange(t_embed,'b f a d -> (b a) f d')
        s_embed=einops.rearrange(s_embed,'b f a d -> (b f) a d')
        for layer in self.list:
            x=layer(x,t_embed,s_embed)
        return x

class TSE_Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ts=Branches_T(config,'TS',1)
        self.st=Branches_T(config,'ST',1)

    def forward(self,X):
        x1=self.ts(X)
        x2=self.st(X)
        fusion=x1*0.5+x2*0.5
        return fusion

class MixLayer(nn.Module):
    def __init__(self, config,mlp_flag=False):
        super().__init__()
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.transformer.heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.temporal = nn.TransformerEncoder(
            encoder_layer=temporal_encoder_layer, num_layers=config.mix.tfm_layer
        )

        spatial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.transformer.heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.spatial = nn.TransformerEncoder(
            encoder_layer=spatial_encoder_layer, num_layers=config.mix.tfm_layer
        )
        if mlp_flag:
            self.temporal_mlp=TwoLayer(config.dims,config.dropout,config.eps)
            self.spatial_mlp=TwoLayer(config.dims,config.dropout,config.eps)

        self.frames=config.frames
        self.actors=config.actors+1
        self.mlp_flag=mlp_flag


    # temporal batch*node frame dims -> spatial
    # spatial batch* frame node dims -> temporal
    # mask batch frame node
    def forward(self,temporal_x,spaital_x,mask):

        spaital_x=einops.rearrange(spaital_x,'(b f) n d ->  (b n) f d',f=self.frames,n=self.actors) 

        temporal_x=einops.rearrange(temporal_x,'(b n) f d -> (b f) n d',f=self.frames,n=self.actors)
        if mask is not None:
            tem_mask=einops.rearrange(mask,'b f n -> (b f) n',f=self.frames,n=self.actors)
            spa_mask=einops.rearrange(mask,'b f n ->  (b n) f ',f=self.frames,n=self.actors) 
            spa_mask[torch.all(spa_mask==True,dim=-1)]=False
        
        if self.mlp_flag:
            if mask is None:
                out_s=self.spatial(self.spatial_mlp(temporal_x))
                out_t=self.temporal(self.temporal_mlp(spaital_x))
            else:
                out_s=self.spatial(self.spatial_mlp(temporal_x),
                                src_key_padding_mask=tem_mask)
                out_t=self.temporal(self.temporal_mlp(spaital_x),
                                src_key_padding_mask=spa_mask)      
        else:
            if mask is None:
                out_s=self.spatial(temporal_x)
                out_t=self.temporal(spaital_x)
            else:
                # breakpoint()
                out_s=self.spatial(temporal_x,
                        src_key_padding_mask=tem_mask)
                out_t=self.temporal(spaital_x,
                        src_key_padding_mask=spa_mask)
        # breakpoint()
        return out_t,out_s

class MixBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # st
        self.layer1=MixLayer(config,False)
        # ts
        self.layer2=MixLayer(config,True)

        self.frames=config.frames
        self.actors=config.actors+1
        self.mode=config.mix.mode
        self.tmr_mlp=MLP(config.dims,config.dims,config.dropout,config.eps)
        self.spt_mlp=MLP(config.dims,config.dims,config.dropout,config.eps)


    # out temporal spatial
    # temporal batch*node frame dim
    # spatial batch*frame node dim
    def forward(self,temporal_in,spatial_in,mask):

        layer1_t,layer1_s=self.layer1(temporal_in,spatial_in,mask)

        if self.mode=='mix':
            layer2_t=self.tmr_mlp(einops.rearrange(layer1_s,'(b f) n d -> (b n) f d',f=self.frames,n=self.actors)+layer1_t+temporal_in)
            layer2_s=self.spt_mlp(einops.rearrange(layer1_t,'(b n) f d -> (b f) n d',f=self.frames,n=self.actors)+layer1_s+spatial_in)
        else:
            layer2_t=temporal_in+layer1_t
            layer2_s=spatial_in+layer1_s

        layer3_t,layer3_s=self.layer2(layer2_t,layer2_s,mask)
        return layer3_t,layer3_s

class MixTSE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer=config.mix.layer
        self.tse=nn.ModuleList() 
        for i in range(self.layer):
            self.tse.append(MixBlock(config))
        self.mlp=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.GELU())

        
    # batch frame node dim
    # mask batch frame node
    def forward(self,X,mask=None):
        ba,fr,nod,dim=X.shape
        temporal=einops.rearrange(X,'b f n d -> (b n) f d')
        spatial=einops.rearrange(X,'b f n d -> (b f) n d')
        if mask is not None:
            padding_=torch.zeros((ba,fr,1),dtype=torch.bool).to(mask.device)
            mask=torch.cat([padding_,mask],dim=-1)
        for layer in self.tse:
            temporal,spatial=layer(temporal,spatial,mask)
        temporal=einops.rearrange(temporal,'(b n) f d -> b f n d',b=ba,f=fr,n=nod)
        spatial=einops.rearrange(spatial,'(b f) n d -> b f n d',b=ba,f=fr,n=nod)
        return self.mlp(temporal+spatial)
        # return self.merging(torch.cat([temporal,spatial],dim=-1))

class TemporalSpatialEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ts_branch1=OneBranch(config,'TS')
        self.ts_branch2=OneBranch(config,'TS')
        self.st_branch1=OneBranch(config,"ST")
        self.st_branch2=OneBranch(config,"ST")
        self.fusion=Fusion(config)
    
    def forward(self,X,edge_index):
        ts_data,ts_edge_index=self.ts_branch1(X,edge_index)
        st_data,st_edge_index=self.st_branch1(X,edge_index)
        ts_data=self.ts_branch2(ts_data,ts_edge_index)
        st_data=self.st_branch2(st_data,st_edge_index)
        x,edge=self.fusion(ts_data,st_data)
        return x,edge

class TSGPNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ts_branch1=OneBranchGCN(config,'TS')
        self.ts_branch2=OneBranchGCN(config,'TS')
        self.st_branch1=OneBranchGCN(config,"ST")
        self.st_branch2=OneBranchGCN(config,"ST")
        self.fusion=Fusion(config)
    
    def forward(self,X,edge_index):
        ts_data,ts_edge_index=self.ts_branch1(X,edge_index)
        st_data,st_edge_index=self.st_branch1(X,edge_index)
        ts_data=self.ts_branch2(ts_data,ts_edge_index)
        st_data=self.st_branch2(st_data,st_edge_index)
        x=ts_data[0]*0.5+st_data[0]*0.5
        edges=torch.unique(torch.cat([ts_data[1],st_data[1]],dim=1),dim=1)
        return x,edges
    
class TSGCN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ts_branch1=OneBranchGCN(config,'TS')
        self.ts_branch2=OneBranchGCN(config,'TS')
        self.st_branch1=OneBranchGCN(config,"ST")
        self.st_branch2=OneBranchGCN(config,"ST")
        self.fusion=Fusion(config)
    
    def forward(self,X,edge_index):
        ts_data,ts_edge_index=self.ts_branch1(X,edge_index)
        st_data,st_edge_index=self.st_branch1(X,edge_index)
        ts_data=self.ts_branch2(ts_data,ts_edge_index)
        st_data=self.st_branch2(st_data,st_edge_index)
        x=ts_data[0]*0.5+st_data[0]*0.5
        return x

class CommonDGC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dgc=DynamicGCN(config)
    
    def forward(self,X,edge):
        return self.dgc(X,edge)
    
class PrivateDGC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dgc=DynamicGCN(config)
    
    def forward(self,X,edge):
        return self.dgc(X,edge)   

class GeneralGPNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gpnn=GPNN2(config)
        self.fintune=False
    

    def switch_fine(self, flag=None):
        if flag == None:
            self.fintune= True if self.fintune==False else False
        else:
            self.fintune=flag

    def forward(self,edge_features,node_features,finetune_tokens=None):
        if self.fintune:
            node_features=finetune_tokens+node_features
            return 
        else:
            edge_features,node_features=self.gpnn(edge_features,node_features)

        return node_features

class AtomActionSingle(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layer1=MLP(config.dims,config.atom.dims,config.dropout,config.eps)
        self.layer2=MLP(config.atom.dims,config.atom.singular,config.dropout,config.eps)
    
    def forward(self,X):
        data1=self.layer1(X)
        data2=self.layer2(data1)
        return data1,data2

class AtomAction(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.atoms=nn.ModuleList()
        self.heads=config.atom.heads
        for i in range(self.heads):
            self.atoms.append(AtomActionSingle(config))
    

    def forward(self,X):
        data1_list=[]
        data2_list=[]
        for layer in self.atoms:
            data1,data2=layer(X)
            data1_list.append(data1)
            data2_list.append(data2)
        
        return torch.stack(data1_list,dim=-2),torch.stack(data2_list,dim=-2)

class CLSHead(nn.Module):
    def __init__(self,dims,eps,dropout,cls):
        super().__init__()
        # self.ffn=FFN(dims,eps,dims*4,dropout)
        self.ffn=MLP(dims,dims,dropout,eps)
        self.cls_head=nn.Linear(dims,cls)
    def get_last_layer(self):
        return self.ffn.get_last_layer()

    def forward(self,X):
        return self.cls_head(self.ffn(X))

class ReconstructNetwork(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.layer=MLPs(config.dims,config.dropout,config.eps,config.recs.layer)
    
    def forward(self,X):
        return self.layer(X)
        # return self.layer2(self.layer1(X))

class CLIPAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pj=nn.Sequential(nn.Dropout(config.dropout),
                              nn.Linear(config.adapter.pj.indims,config.adapter.pj.outdims),nn.ReLU(inplace=True))
        self.adapter=nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True)
        )


        # self.adapter=nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(768, 768),
        #     nn.ReLU(inplace=True)
        # )
    
    # batch frames nums dims
    def forward(self,x):
        # pj_f=self.pj(x)
        # fea=self.adapter(pj_f)

        fea=self.pj(x+self.adapter(x))
        return fea

class VisualModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.norm=nn.Sequential(nn.LayerNorm(config.dims),nn.GELU())
        self.sq=nn.Sequential()
        for i in range(config.lan.layers):
            if i != config.lan.layers - 1:
                self.sq.add_module('ffn'+str(i),FFN(in_dim=config.dims,dropout=config.dropout,eps=config.eps))
            else:
                self.sq.add_module('ffn'+str(i),FFN(in_dim=config.dims,dropout=config.dropout,eps=config.eps,norm_=False))
    def forward(self,X):
        return self.norm(X+self.sq(X))

class ProjectionHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.li=nn.Sequential(nn.Linear(512,768),nn.LayerNorm(768),nn.GELU())
        self.ffn=VisualModel(config)

    def forward(self,X,bbx):
        X=self.li(X)+bbx
        return self.ffn(X)

class MyModel(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
        self.tse=TemporalSpatialEncoder(config)
        self.cdgc=CommonDGC(config)
        self.pdgc=PrivateDGC(config)
        self.ge=GraphEmbedding(config)
        self.atom=AtomAction(config)
        self.cls_head=CLSHead(config)

        
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,512),nn.LayerNorm(512),nn.GELU())
        self.pretrain=pretrain
        
        self.pj=ProjectionHead(config)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        nn.init.xavier_uniform_(self.pos)
        
    # 
    @torch.no_grad()
    def pre_process(self,X,mask):
        mask1=mask.unsqueeze(-1).float()
        mask2=mask.unsqueeze(-2).float()

        mask3=mask1@mask2

        B,Frame,Nums,Dims=X.shape
        data_list=[]
        x=einops.rearrange(X,'b f n d -> (b f) n d')
        mask3=einops.rearrange(mask3,'b f n d -> (b f) n d')
        for i in range(B*Frame):
            
            feature_map=x[i]
            mask_list=mask3[i]
            edge_index, edge_attr = dense_to_sparse(mask_list)

            # 构建 PyTorch Geometric 的数据对象
            graph = Data(x=feature_map, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(graph)
        batch = Batch.from_data_list(data_list)
        return batch.x,batch.edge_index,mask3


    def forward(self,X,mask,bbx_list):
        B,Frame,Nums,Dims=X.shape
        bbx=self.bbx_linear(bbx_list)

        pos=self.pos.repeat(B,1,Nums,1)
        X=self.pj(X+bbx)+pos
        X,edge_index,mask3=self.pre_process(X,mask)
        X=einops.rearrange(X,'(b f a) d -> (b f) a d',b=B,f=Frame)
        X,edge_index=self.tse(X,edge_index)
        # X=einops.rearrange(X,'(b f) a d -> (b f a) d',f=Frame,a=Nums)

        common=self.cdgc(X,edge_index)

        # print(common[0].shape)
        cls_token=self.ge(einops.rearrange(common[0],'(b f) a d -> b f a d',a=self.config.actors,f=self.config.frames))

        if not self.pretrain:
            private=self.pdgc(X,edge_index)
            atom=self.atom(cls_token)
        ans=self.cls_head(cls_token)
        if not self.pretrain:
            return private,common,atom,ans

        return ans


class MyModel_T(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
        self.tse=TSE_Transformer(config)
        self.ge=GraphEmbedding(config)
        
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,512),nn.LayerNorm(512),nn.GELU())
        self.pretrain=pretrain
        
        self.pj=ProjectionHead(config)
        # front embedding
        

    def forward(self,X,mask,bbx_list):
        B,Frame,Nums,Dims=X.shape
        bbx=self.bbx_linear(bbx_list)
        
        X=self.pj(X+bbx)
        X=self.tse(X)


        cls_token=self.ge(X)
        ans=self.cls_head(cls_token)
        
        return ans


class MyModel_CLS(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
        self.tse=TemporalSpatialEncoder(config)
        self.cdgc=CommonDGC(config)
        self.pdgc=PrivateDGC(config)
        self.ge=GraphEmbedding(config)
        self.atom=AtomAction(config)
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,512),nn.LayerNorm(512),nn.GELU())
        self.pretrain=pretrain
        
        self.pj=ProjectionHead(config)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1, 16,1, 768))
        nn.init.xavier_uniform_(self.pos)
        self.cls_embed=nn.Embedding(36,512)
        
    # 
    @torch.no_grad()
    def pre_process(self,X,mask):
        mask1=mask.unsqueeze(-1).float()
        mask2=mask.unsqueeze(-2).float()
        mask3=mask2@mask1
        B,Frame,Nums,Dims=X.shape
        data_list=[]
        x=einops.rearrange(X,'b f n d -> (b f) n d')
        mask3=einops.rearrange(mask3,'b f n d -> (b f) n d')
        for i in range(B*Frame):
            
            feature_map=x[i]
            mask_list=mask3[i]
            edge_index, edge_attr = dense_to_sparse(mask_list)

            # 构建 PyTorch Geometric 的数据对象
            graph = Data(x=feature_map, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(graph)
        batch = Batch.from_data_list(data_list)
        return batch.x,batch.edge_index,mask3


    def forward(self,X,mask,bbx_list):
        X=self.cls_embed(X).squeeze(-2)
        mask_4d=mask.unsqueeze(-1).expand(-1, -1, -1, 512)
        X=X*mask_4d
        B,Frame,Nums,Dims=X.shape
        bbx=self.bbx_linear(bbx_list)
        pos=self.pos.repeat(B,1,Nums,1)
        X=self.pj(X+bbx)+pos
        X,edge_index,mask3=self.pre_process(X,mask)
        X,edge_index=self.tse(X,edge_index)
        common=self.cdgc(X,edge_index)

        cls_token=self.ge(einops.rearrange(common[0],'(b f) a d -> b f a d',f=self.config.frames))
        if not self.pretrain:
            private=self.pdgc(X,edge_index)
            atom=self.atom(cls_token)
        ans=self.cls_head(cls_token)
        if not self.pretrain:
            return private,common,atom,ans
        
        return ans


class PureT(nn.Module):

    def __init__(self, config,pretrain):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.transformer.heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=8
        )

        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,512),nn.LayerNorm(512),nn.GELU())
        self.pretrain=pretrain
        
        self.pj=ProjectionHead(config)

        self.cls_token=nn.Parameter(torch.zeros(1,768))
        nn.init.xavier_uniform_(self.cls_token)
        self.t_embed = nn.Parameter(torch.zeros(1,16,1, 768))
        self.s_embed=nn.Parameter(torch.zeros(1,1,10, 768))
        nn.init.xavier_uniform_(self.t_embed)
        nn.init.xavier_uniform_(self.s_embed)

    def forward(self,X,mask,bbx_list):
        B,Frame,Nums,Dims=X.shape
        bbx=self.bbx_linear(bbx_list)
        
        X=self.pj(X+bbx)
        t_embed=self.t_embed.repeat(B,1,Nums,1)
        s_embed=self.s_embed.repeat(B,Frame,1,1)
        X=X+t_embed+s_embed
        X=einops.rearrange(X,'b f a d -> b (f a) d')
        cls_token=self.cls_token.repeat(B,1).unsqueeze(1)
        X=torch.cat([cls_token,X],dim=1)
        X=self.transformer(X)
        ans_token=X[:,0,:]
        ans=self.cls_head(ans_token)

        
        return ans

class MyViT(nn.Module):
    def __init__(self,config):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dims,
            nhead=config.transformer.heads,
            dim_feedforward=config.dims*4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )

        # Temporal Transformer
        self.vit = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=8
        )

        self.cls=nn.Embedding(1,config.dims)

    def forward(self,x):
        cls_token=self.cls.weight.unsqueeze(0).repeat(x.shape[0],1,1)
        x=torch.cat([cls_token,x],dim=-2)
        ans= self.vit(x)
        return ans[:,0,:]

class SViT(nn.Module):
    def __init__(self,config):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dims,
            nhead=config.transformer.heads,
            dim_feedforward=config.dims*4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )

        # Temporal Transformer
        self.vit = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=4
        )

    # batch_size,..,dims
    def forward(self,x):
        ans= self.vit(x)
        return ans

class TextT(nn.Module):

    def __init__(self, config,pretrain):
        super().__init__()
        self.vit=MyViT(config)
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,512),nn.LayerNorm(512),nn.GELU())
        self.pretrain=pretrain
        
        self.pj=ProjectionHead(config)

        self.cls_token=nn.Parameter(torch.zeros(1,768))
        nn.init.xavier_uniform_(self.cls_token)
        self.t_embed = nn.Parameter(torch.zeros(1,4,1, 768))
        self.s_embed=nn.Parameter(torch.zeros(1,1,10, 768))
        nn.init.xavier_uniform_(self.t_embed)
        nn.init.xavier_uniform_(self.s_embed)

        self.cls_embed=nn.Embedding(36,512)
        
    def forward(self,X,mask,bbx_list):
        B,Frame,Nums,Dims=X.shape
        bbx=self.bbx_linear(bbx_list)
        X=self.cls_embed(X).squeeze(-2)
        mask_4d=mask.unsqueeze(-1).expand(-1, -1, -1, 512)
        X=X*mask_4d

        # X=self.pj(X+bbx)
        X=self.pj(X)
        t_embed=self.t_embed.repeat(B,1,Nums,1)
        s_embed=self.s_embed.repeat(B,Frame,1,1)
        X=X+t_embed+s_embed
        X=einops.rearrange(X,'b f a d -> b (f a) d')
        ans_token=self.vit(X)
        
        ans=self.cls_head(ans_token)

        
        return ans

class MyModelGCN(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
        self.tse=TSGCN(config)

        self.ge=GraphEmbedding(config)
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain

        self.rel_mlp=MLPCLS(config.dims*2,config.cls.rel,config.dropout,config.eps)
        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)       

        self.pj=CLIPAdapter(config)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        # self.vm=VisualModel(config)
        nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
        
    # 
    @torch.no_grad()
    def pre_process(self,X,mask):
        mask1=mask.unsqueeze(-1).float()
        mask2=mask.unsqueeze(-2).float()

        mask3=mask1@mask2

        B,Frame,Nums,Dims=X.shape
        data_list=[]
        x=einops.rearrange(X,'b f n d -> (b f) n d')
        mask3=einops.rearrange(mask3,'b f n d -> (b f) n d')
        for i in range(B*Frame):
            
            feature_map=x[i]
            mask_list=mask3[i]
            edge_index, edge_attr = dense_to_sparse(mask_list)

            # 构建 PyTorch Geometric 的数据对象
            graph = Data(x=feature_map, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(graph)
        batch = Batch.from_data_list(data_list)
        return batch.x,batch.edge_index,mask3

    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,X,mask,bbx_list,rel_label,cls_label):
        B,Frame,Nums,Dims=X.shape
        Nums=Nums-1
        bbx=self.bbx_linear(bbx_list)
        
        # time pos
        pos=self.pos.repeat(B,1,Nums,1)
        all_feature=self.pj(X,bbx)

        human_obj_feature=all_feature[:,:,1:,:]



        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-1,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        rel_fea=torch.cat([human_feature,obj_feature],dim=-1)
        rel_fea=rel_fea.reshape(-1,self.config.dims*2)
        # print(mask.shape)
        flat_rel_mask=mask[:,:,1:].reshape(-1,1)
        flag_rel_mask_tensor=torch.tensor(flat_rel_mask,dtype=torch.bool,device=flat_rel_mask.device).squeeze()
        rel=self.rel_mlp(rel_fea[flag_rel_mask_tensor])
        rel_label_dims=rel_label.shape[-1]
        rel_label=rel_label[:,:,1:,:].reshape(-1,rel_label_dims)[flag_rel_mask_tensor]

        flat_mask=mask.reshape(-1,1)
        flat_mask_tensor=torch.tensor(flat_mask,dtype=torch.bool,device=flat_mask.device).squeeze()
        cls=self.obj_mlp(human_obj_feature.reshape(-1,self.config.dims)[flat_mask_tensor])
        cls_label_dims=cls_label.shape[-1]
        cls_label=cls_label.reshape(-1,cls_label_dims)[flat_mask_tensor]


        X=human_obj_feature+pos
        X,edge_index,mask3=self.pre_process(X,mask)
        X=einops.rearrange(X,'(b f a) d -> (b f) a d',b=B,f=Frame)
        X=self.tse(X,edge_index)
        # X=einops.rearrange(X,'(b f) a d -> (b f a) d',f=Frame,a=Nums)


        # print(common[0].shape)
        cls_token=self.ge(einops.rearrange(X,'(b f) a d -> b f a d',a=self.config.actors,f=self.config.frames))

        ans=self.cls_head(cls_token)

        return ans,rel,cls,rel_label,cls_label

class VideoEmb(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        vl:
        out1: 96
        # frame * 96
        out2: 1536
        '''
        self.video_level=Flatter(768,96,1536,0.1,1e-5)
    
    def forward(self,X):
        return self.video_level(X)


class MymodelGPNN(nn.Module):

    def __init__(self, config,params,pretrain=False):
        super().__init__()
        self.tse=TSGPNN(config)
        self.gpnn=GPNN(params)
        self.ge=VideoEmb()
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain

        self.rel_mlp=MLPCLS(config.dims*2,config.cls.rel,config.dropout,config.eps)
        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)       

        self.pj=CLIPAdapter(config)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        # self.vm=VisualModel(config)
        nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
        
    # 
    @torch.no_grad()
    def pre_process(self,X,mask):
        mask1=mask.unsqueeze(-1).float()
        mask2=mask.unsqueeze(-2).float()

        mask3=mask1@mask2

        B,Frame,Nums,Dims=X.shape
        data_list=[]
        x=einops.rearrange(X,'b f n d -> (b f) n d')
        mask3=einops.rearrange(mask3,'b f n d -> (b f) n d')
        for i in range(B*Frame):
            
            feature_map=x[i]
            mask_list=mask3[i]
            edge_index, edge_attr = dense_to_sparse(mask_list)

            # 构建 PyTorch Geometric 的数据对象
            graph = Data(x=feature_map, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(graph)
        batch = Batch.from_data_list(data_list)
        return batch,mask3

    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,X,mask,bbx_list,rel_label,cls_label):
        B,Frame,Nums,Dims=X.shape
        Nums=Nums-1
        bbx=self.bbx_linear(bbx_list)
        
        # time pos
        pos=self.pos.repeat(B,1,Nums,1)
        all_feature=self.pj(X,bbx)
        human_obj_feature=all_feature[:,:,1:,:]
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-1,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        rel_fea=torch.cat([human_feature,obj_feature],dim=-1)
        rel_fea=rel_fea.reshape(-1,self.config.dims*2)
        # print(mask.shape)
        flat_rel_mask=mask[:,:,1:].reshape(-1,1)
        flag_rel_mask_tensor=torch.tensor(flat_rel_mask,dtype=torch.bool,device=flat_rel_mask.device).squeeze()
        rel=self.rel_mlp(rel_fea[flag_rel_mask_tensor])
        rel_label_dims=rel_label.shape[-1]
        rel_label=rel_label[:,:,1:,:].reshape(-1,rel_label_dims)[flag_rel_mask_tensor]

        flat_mask=mask.reshape(-1,1)
        flat_mask_tensor=torch.tensor(flat_mask,dtype=torch.bool,device=flat_mask.device).squeeze()
        cls=self.obj_mlp(human_obj_feature.reshape(-1,self.config.dims)[flat_mask_tensor])
        cls_label_dims=cls_label.shape[-1]
        cls_label=cls_label.reshape(-1,cls_label_dims)[flat_mask_tensor]


        X=human_obj_feature+pos
        batch_data,mask3=self.pre_process(X,mask)
        X=batch_data.x
        edge_index=batch_data.edge_index
        X=einops.rearrange(X,'(b f a) d -> (b f) a d',b=B,f=Frame)
        X,edges=self.tse(X,edge_index)
        # print('x.shape',X.shape)
        batch_data.x=einops.rearrange(X,'(b f ) a d -> (b f a ) d',b=B,f=Frame)
        batch_data.edge_index=edge_index
        # print(batch_data.edge_attr)
        # encode_values = dict(zip(['x', 'num_pooling_layers'], [h, len(assignments)]))

        encode_value=self.gpnn(batch_data)
        # X=einops.rearrange(X,'(b f) a d -> (b f a) d',f=Frame,a=Nums)
        X=encode_value['x']
        breakpoint()

        # print(common[0].shape)
        # cls_token=self.ge(einops.rearrange(X,'(b f) a d -> b f a d',a=self.config.actors,f=self.config.frames))
        cls_token=self.ge(einops.rearrange(X,'(b f) d -> b f d',b=B,f=Frame))
        ans=self.cls_head(cls_token)

        return ans,rel,cls,rel_label,cls_label

class MymodelGPNN2(nn.Module):

    def __init__(self, config,params,pretrain=False):
        super().__init__()
        # self.tse=TSE_Transformer(config)
        self.tse=MixTSE(config)
        self.gpnn=GPNN2(config)
        self.ge=VideoEmb()
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain

        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)
        
        self.rel_mlp_2=MLPCLS(config.dims*2,config.dims,config.dropout,config.eps)

        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)       

        self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
                    dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        
        self.pj=ProjectionHead(config.dims,config.eps,config.dims*4,config.dropout)

        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        # self.vm=VisualModel(config)
        nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
        
    # 


    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,X,mask,bbx_list,rel_label,cls_label):
        B,Frame,Nums,Dims=X.shape
        Nums=Nums-1
        bbx=self.bbx_linear(bbx_list)
        
        # time pos
        pos=self.pos.repeat(B,1,Nums,1)
        all_feature=self.tse(self.adapter(X)+bbx)

        # always train 
        all_feature=self.pj(all_feature)

        human_obj_feature=all_feature[:,:,1:,:]

        edge_feature_1=human_obj_feature.unsqueeze(-2).repeat(1,1,1,10,1)
        edge_feature_2=edge_feature_1.transpose(-2,-3)
        edge_feature=self.rel_mlp_2(torch.cat([edge_feature_1,edge_feature_2],dim=-1))
    
        rel_fea_2=edge_feature[:,:,0,1:,:].reshape(-1,self.config.dims)


        flat_rel_mask=mask[:,:,1:].reshape(-1,1)
        flag_rel_mask_tensor=torch.tensor(flat_rel_mask,dtype=torch.bool,device=flat_rel_mask.device).squeeze()

        rel=self.rel_mlp(rel_fea_2[flag_rel_mask_tensor])
        rel_label_dims=rel_label.shape[-1]
        rel_label=rel_label[:,:,1:,:].reshape(-1,rel_label_dims)[flag_rel_mask_tensor]

        flat_mask=mask.reshape(-1,1)
        flat_mask_tensor=torch.tensor(flat_mask,dtype=torch.bool,device=flat_mask.device).squeeze()
        cls=self.obj_mlp(human_obj_feature.reshape(-1,self.config.dims)[flat_mask_tensor])
        cls_label_dims=cls_label.shape[-1]
        cls_label=cls_label.reshape(-1,cls_label_dims)[flat_mask_tensor]


        X=human_obj_feature+pos
        node_features=self.gpnn(edge_feature,X)

        cls_token=node_features[:,:,0,:]
        cls_token=self.ge(cls_token)
        ans=self.cls_head(cls_token)

        return ans,rel,cls,rel_label,cls_label


class MymodelGPNN3(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
        # self.tse=TSE_Transformer(config)
        self.tse=MixTSE(config)
        self.gpnn=GPNN3(config)
        self.ge=DynamicFlatter(config.dims)
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain
        
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*3,768),nn.GELU(),
                                    nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-5),nn.GELU())

        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)       

        self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
                    dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        # self.vm=VisualModel(config)
        nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
        
    # 


    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,X,mask,bbx_list,rel_label,cls_label):
        B,Frame,Nums,Dims=X.shape
        Nums=Nums-1
        bbx=self.bbx_linear(bbx_list)
        
        # time pos
        pos=self.pos.repeat(B,1,Nums,1)
        all_feature=self.tse(self.adapter(X)+bbx)

        # always train 
        all_feature=self.pj(all_feature)

        human_obj_feature=all_feature[:,:,1:,:]

        global_feature=all_feature[:,:,0,:].unsqueeze(-2).unsqueeze(-2).repeat(1,1,10,10,1)

        edge_feature_1=human_obj_feature.unsqueeze(-2).repeat(1,1,1,10,1)
        edge_feature_2=edge_feature_1.transpose(-2,-3)

        edge_feature=torch.cat([global_feature,edge_feature_1,edge_feature_2],dim=-1)
    
        rel_fea_2=self.edge_fun(edge_feature[:,:,0,1:,:]).reshape(-1,self.config.dims)


        flat_rel_mask=mask[:,:,1:].reshape(-1,1)
        flag_rel_mask_tensor=torch.tensor(flat_rel_mask,dtype=torch.bool,device=flat_rel_mask.device).squeeze()

        rel=self.rel_mlp(rel_fea_2[flag_rel_mask_tensor])
        rel_label_dims=rel_label.shape[-1]
        rel_label=rel_label[:,:,1:,:].reshape(-1,rel_label_dims)[flag_rel_mask_tensor]

        flat_mask=mask.reshape(-1,1)
        flat_mask_tensor=torch.tensor(flat_mask,dtype=torch.bool,device=flat_mask.device).squeeze()
        cls=self.obj_mlp(human_obj_feature.reshape(-1,self.config.dims)[flat_mask_tensor])
        cls_label_dims=cls_label.shape[-1]
        cls_label=cls_label.reshape(-1,cls_label_dims)[flat_mask_tensor]


        X=human_obj_feature+pos
        node_features=self.gpnn(X,global_feature)

        # cls_token=node_features[:,:,0,:]
        # cls_token=self.ge(cls_token)
        cls_token=self.ge(node_features)
        # cls_token_global=self.ge()
        ans=self.cls_head(cls_token)

        return ans,rel,cls,rel_label,cls_label

# construct
class MymodelGPNN4(nn.Module):

    def __init__(self, config,params,pretrain=False):
        super().__init__()
        # self.tse=TSE_Transformer(config)
        self.tse=MixTSE(config)
        self.common=GPNN3(config)
        self.private=GPNN3(config)
        self.ge=VideoEmb()
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768,eps=config.eps),nn.GELU())
        self.pretrain=pretrain
        
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*3,768),nn.GELU(),
                                    nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-5),nn.GELU())

        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)       

        self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
                    dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)

        self.rcsn=ReconstructNetwork(config)

        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        # self.vm=VisualModel(config)
        nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
        
    # 


    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,X,mask,bbx_list,rel_label,cls_label):
        B,Frame,Nums,Dims=X.shape
        Nums=Nums-1
        bbx=self.bbx_linear(bbx_list)
        
        # time pos
        pos=self.pos.repeat(B,1,Nums,1)
        all_feature=self.tse(self.adapter(X)+bbx)

        # always train 
        all_feature=self.pj(all_feature)

        human_obj_feature=all_feature[:,:,1:,:]

        global_feature=all_feature[:,:,0,:].unsqueeze(-2).unsqueeze(-2).repeat(1,1,10,10,1)

        edge_feature_1=human_obj_feature.unsqueeze(-2).repeat(1,1,1,10,1)
        edge_feature_2=edge_feature_1.transpose(-2,-3)

        edge_feature=torch.cat([global_feature,edge_feature_1,edge_feature_2],dim=-1)
    
        rel_fea_2=self.edge_fun(edge_feature[:,:,0,1:,:]).reshape(-1,self.config.dims)


        flat_rel_mask=mask[:,:,1:].reshape(-1,1)
        flag_rel_mask_tensor=torch.tensor(flat_rel_mask,dtype=torch.bool,device=flat_rel_mask.device).squeeze()

        rel=self.rel_mlp(rel_fea_2[flag_rel_mask_tensor])
        rel_label_dims=rel_label.shape[-1]
        rel_label=rel_label[:,:,1:,:].reshape(-1,rel_label_dims)[flag_rel_mask_tensor]

        flat_mask=mask.reshape(-1,1)
        flat_mask_tensor=torch.tensor(flat_mask,dtype=torch.bool,device=flat_mask.device).squeeze()
        cls=self.obj_mlp(human_obj_feature.reshape(-1,self.config.dims)[flat_mask_tensor])
        cls_label_dims=cls_label.shape[-1]
        cls_label=cls_label.reshape(-1,cls_label_dims)[flat_mask_tensor]


        X=human_obj_feature+pos
        common_features=self.common(X,global_feature)
        private_features=self.private(X,global_feature)

        cls_token1=common_features[:,:,0,:]
        cls_token2=private_features[:,:,0,:]+cls_token1
        cls_tokens=torch.cat([cls_token1,cls_token2],dim=0)
        cls_tokens=self.ge(cls_tokens)
        # cls_token_global=self.ge()
        ans=self.cls_head(cls_tokens)
        ans_common=ans[:B,:]
        ans_all=ans[B:,:]
        target_features=human_obj_feature
        reconstruct_features=self.rcsn(common_features+private_features)

        return ans_common,common_features,private_features,reconstruct_features,target_features,\
                ans_all,rel,cls,rel_label,cls_label

# no mask
class MymodelGPNN5(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
        # self.tse=TSE_Transformer(config)
        self.tse=MixTSE(config)
        self.gpnn=GPNN3(config)
        self.ge=DynamicFlatter(config.dims)
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain
        
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*3,768),nn.GELU(),
                                    nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-5),nn.GELU())

        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)       

        self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
                    dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        # self.vm=VisualModel(config)
        nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
        
    # 


    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,X,mask,bbx_list):
        B,Frame,Nums,Dims=X.shape
        Nums=Nums-1
        bbx=self.bbx_linear(bbx_list)
        
        # time pos
        pos=self.pos.repeat(B,1,Nums,1)
        all_feature=self.tse(self.adapter(X)+bbx)

        # always train 
        all_feature=self.pj(all_feature)

        human_obj_feature=all_feature[:,:,1:,:]

        global_feature=all_feature[:,:,0,:].unsqueeze(-2).unsqueeze(-2).repeat(1,1,10,10,1)

        edge_feature_1=human_obj_feature.unsqueeze(-2).repeat(1,1,1,10,1)
        edge_feature_2=edge_feature_1.transpose(-2,-3)

        edge_feature=torch.cat([global_feature,edge_feature_1,edge_feature_2],dim=-1)
    
        rel_fea_2=self.edge_fun(edge_feature[:,:,0,1:,:])


        # flat_rel_mask=mask[:,:,1:].reshape(-1,1)
        # flag_rel_mask_tensor=torch.tensor(flat_rel_mask,dtype=torch.bool,device=flat_rel_mask.device).squeeze()

        rel=self.rel_mlp(rel_fea_2)
        # rel_label_dims=rel_label.shape[-1]
        # rel_label=rel_label[:,:,1:,:].reshape(-1,rel_label_dims)[flag_rel_mask_tensor]

        # flat_mask=mask.reshape(-1,1)
        # flat_mask_tensor=torch.tensor(flat_mask,dtype=torch.bool,device=flat_mask.device).squeeze()
        cls=self.obj_mlp(human_obj_feature)
        # cls_label_dims=cls_label.shape[-1]
        # cls_label=cls_label.reshape(-1,cls_label_dims)[flat_mask_tensor]


        X=human_obj_feature+pos
        node_features=self.gpnn(X,global_feature)

        # cls_token=node_features[:,:,0,:]
        # cls_token=self.ge(cls_token)
        cls_token=self.ge(node_features)
        # cls_token_global=self.ge()
        ans=self.cls_head(cls_token)

        return ans,rel,cls


class MymodelGPNNText(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
        # self.tse=TSE_Transformer(config)
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        # self.cls_embed=nn.Linear(38,768,bias=False)
        # self.rel_embed=nn.Embedding(30,768)
        # self.cls_mlp=nn.Sequential(nn.Linear(38,768),nn.GELU(),nn.Dropout(config.dropout),FFN(config.dims,config.eps,config.dims*4,config.dropout))
        self.rel_mlp=nn.Sequential(nn.Linear(30,768),nn.GELU(),nn.Dropout(config.dropout),FFN(config.dims,config.eps,config.dims*4,config.dropout))

        self.tse=MixTSE(config)
        self.gpnn=GPNNText(config)
        self.ge=DynamicFlatter(config.dims)
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.fusion=nn.Sequential(nn.Linear(768,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain
        
        # self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        # self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*3,768),nn.GELU(),
        #                             nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-5),nn.GELU())

        # self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)

        # self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
        #             dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        
        # self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        # self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        # self.vm=VisualModel(config)
        # nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
        
    # 


    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,cls,bbx_list):
        B,Frame,Nums,=cls.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        # breakpoint()
        # rel_feature=self.rel_mlp(rel)
        # breakpoint()
        # rel_ans=self.rel_mlp(rel_feature)
        # breakpoint()
        cls_feature=self.fusion(cls_feature+bbx)
        
        # time pos
        node_features=self.gpnn(cls_feature)

        # cls_token=node_features[:,:,0,:]
        # cls_token=self.ge(cls_token)
        cls_token=self.ge(node_features)
        # cls_token_global=self.ge()
        ans=self.cls_head(cls_token)

        return ans

class MymodelGPNN6(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
        # self.tse=TSE_Transformer(config)
        self.tse=MixTSE(config)
        self.gpnn=GPNN5(config)
        self.ge=DynamicFlatter(config.dims)
        self.cls_head=CLSHead(config)
        self.global_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain
        
        self.gle=VideoEmb()

        self.nodes_fusion=MLP(config.dims*2,config.dims,config.dropout,config.eps)

        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)       

        self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
                    dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        # self.vm=VisualModel(config)
        nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
        
    # 


    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,X,mask,bbx_list):
        B,Frame,Nums,Dims=X.shape
        Nums=Nums-1
        bbx=self.bbx_linear(bbx_list)
        
        # time pos
        pos=self.pos.repeat(B,1,Nums,1)
        all_feature=self.tse(self.adapter(X)+bbx)

        # always train 

        human_obj_feature=all_feature[:,:,1:,:]

        global_feature=all_feature[:,:,0,:].unsqueeze(-2)


        # print('global',global_feature.shape)
        global_ans=self.global_head(self.gle(global_feature.squeeze()))

        global_feature=global_feature.repeat(1,1,10,1)

        nodes_cat=self.nodes_fusion(torch.cat([global_feature,human_obj_feature],dim=-1))
    
        nodes_cat=self.pj(nodes_cat)

        cls=self.obj_mlp(nodes_cat)



        X=nodes_cat+pos
        node_features=self.gpnn(X)

        cls_token=self.ge(node_features)
        
        ans=self.cls_head(cls_token)

        return ans,global_ans,cls

# rel cls
class MymodelGPNNText2(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
        # self.tse=TSE_Transformer(config)
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        self.rel_embed=nn.Linear(30,768)
        # self.cls_embed=nn.Linear(38,768,bias=False)
        # self.rel_embed=nn.Embedding(30,768)
        # self.cls_mlp=nn.Sequential(nn.Linear(38,768),nn.GELU(),nn.Dropout(config.dropout),FFN(config.dims,config.eps,config.dims*4,config.dropout))
        # self.rel_mlp=nn.Sequential(nn.Linear(30,768),nn.GELU(),nn.Dropout(config.dropout),FFN(config.dims,config.eps,config.dims*4,config.dropout))

        self.tse=MixTSE(config)
        self.gpnn=GPNN5(config)
        self.ge=DynamicFlatter(config.dims)
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.fusion=nn.Sequential(nn.Linear(768,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain
        
        # self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        # self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*3,768),nn.GELU(),
        #                             nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-5),nn.GELU())

        # self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)

        # self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
        #             dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        
        # self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        # self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        # self.vm=VisualModel(config)
        # nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
        
    # 


    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,cls,rel,bbx_list):
        B,Frame,Nums=cls.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        # breakpoint()
        # rel_feature=self.rel_mlp(rel)
        # breakpoint()
        # rel_ans=self.rel_mlp(rel_feature)
        # breakpoint()
        cls_feature=self.fusion(cls_feature+bbx)
        human_feature=cls_feature[:,:,0,:].unsqueeze(-2)
        obj_feature=cls_feature[:,:,1:,:]
        # time pos
        human_feature,obj_feature=self.gpnn(human_feature,obj_feature,rel_feature)
        node_features=torch.cat([human_feature,obj_feature],dim=-2)

        # cls_token=node_features[:,:,0,:]
        # cls_token=self.ge(cls_token)
        cls_token=self.ge(node_features)
        # cls_token_global=self.ge()
        ans=self.cls_head(cls_token)

        return ans


class MymodelGPNN7(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
   
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        self.rel_embed=nn.Linear(30,768)


        self.tse=MixTSE(config)
        self.gpnn=GPNN5(config)
        self.ge=DynamicFlatter(config.dims)
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.fusion=nn.Sequential(nn.Linear(768,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain
        
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*3,768),nn.GELU(),
                                    nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-5),nn.GELU())


        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)

        self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
                    dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        # self.vm=VisualModel(config)
        # nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
        
    # 


    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,frames,cls,rel,bbx_list):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        frames_feature=self.tse(self.fusion(self.adapter(frames)+bbx)+pos)


        human_obj_feature=frames_feature[:,:,1:,:]
        cls_ans=self.obj_mlp(human_obj_feature)
        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_feature[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))+rel_feature

        # human_feature=cls_feature[:,:,0,:].unsqueeze(-2)
        # obj_feature=cls_feature[:,:,1:,:]
        
        human_feature,obj_feature=self.gpnn(human_feature,obj_feature,edge_feature)
        node_features=torch.cat([human_feature,obj_feature],dim=-2)

        # cls_token=node_features[:,:,0,:]
        # cls_token=self.ge(cls_token)
        cls_token=self.ge(node_features)
        # cls_token_global=self.ge()
        ans=self.cls_head(cls_token)

        return ans,cls_ans,rel_ans

# construct
class GPNNSeperate(nn.Module):

    def __init__(self, config,pretrain=False):
        super().__init__()
   
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        self.rel_embed=nn.Linear(30,768)


        self.tse=MixTSE(config)
        self.private=GPNN5(config)
        self.common=GPNN5(config)
        self.pffn=MLPs(config.dims,config.dropout,config.eps)
        self.cffn=MLPs(config.dims,config.dropout,config.eps)

        self.ge=DynamicFlatter(config.dims)
        self.cls_head=CLSHead(config)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.fusion=nn.Sequential(nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain
        
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*3,768),nn.GELU(),
                                    nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-5),nn.GELU())


        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)

        self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
                    dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        
        
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1, 768))
        self.rcsn=ReconstructNetwork(config)
        # self.vm=VisualModel(config)
        # nn.init.xavier_uniform_(self.pos)
        # self.box_fusion_linear=MLP(config.dims,config.dims,config.dropout,config.eps)
    
    def init_tokens(self,config,weight_init):
        self.prompt=GPFPlus(config)
        self.prompt.reset_parameters(weight_init)

    # freeze every parameters 
    def fine_tune(self,config,weight_init):
        self.init_tokens(config,weight_init)
        for param in self.parameters():
            param.requires_grad = False
        self.prompt=None
        for param in self.pj.parameters():
            param.requires_grad=True

    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,frames,cls,rel,bbx_list,task_id):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        frames_feature=self.pj(self.tse(self.fusion(self.adapter(frames)+bbx)+pos))
        # frames_feature=self.tse(self.fusion(self.adapter(frames)+bbx)+pos)
        

        # total features for a consist edge cls
        human_obj_feature=frames_feature[:,:,1:,:]
        cls_ans=self.obj_mlp(human_obj_feature)
        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_feature[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))+rel_feature

        # human_feature=cls_feature[:,:,0,:].unsqueeze(-2)
        # obj_feature=cls_feature[:,:,1:,:]
        

        # common features
        c_human_obj_feature=self.cffn(human_obj_feature)
        c_human_feature=c_human_obj_feature[:,:,0,:].unsqueeze(-2)
        c_obj_feature=c_human_obj_feature[:,:,1:,:]
        c_human_feature,c_obj_feature=self.common(c_human_feature,c_obj_feature,edge_feature)
        c_node_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)

        # private features
        p_human_obj_feature=self.pffn(human_obj_feature)
        P_human_feature=p_human_obj_feature[:,:,0,:].unsqueeze(-2)
        p_obj_feature=p_human_obj_feature[:,:,1:,:]
        p_human_feature,p_obj_feature=self.common(P_human_feature,p_obj_feature,edge_feature)
        p_node_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        rec=p_human_obj_feature+c_human_obj_feature
        t_node_features=p_node_features+c_node_features
        node_features=torch.cat([t_node_features,c_node_features],dim=0)
        construct_features=self.rcsn(rec)
        # cls_token=node_features[:,:,0,:]
        # cls_token=self.ge(cls_token)
        cls_token=self.ge(node_features)
        # cls_token_global=self.ge()
        t_ans=self.cls_head(cls_token)
        ans=t_ans[:B,:]
        c_ans=t_ans[B:,:]
        return ans,c_ans,c_human_obj_feature,p_human_obj_feature,construct_features,human_obj_feature,cls_ans,rel_ans

class GPNNMix(nn.Module):
    # false linear true embedding
    def __init__(self, config,flag=False):
        super().__init__()
   
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        self.rel_embed=nn.Linear(30,768)


        self.tse=MixTSE(config)
        self.gpnn=GPNN5(config)
        self.gpfp=GPFPlus(config,flag)
        self.pffn=MLPs(config.dims,config.dropout,config.eps)
        self.cffn=MLPs(config.dims,config.dropout,config.eps)

        self.ge=DynamicFlatter(config.dims)
        self.cls_head=CLSHead(config.dims,config.eps,config.dropout,config.cls.ag)
        self.p_head=CLSHead(config.dims,config.eps,config.dropout,config.cls.ag+1)
        self.c_head=CLSHead(config.dims,config.eps,config.dropout,config.cls.ag+1)

        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.fusion=nn.Sequential(nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768),nn.GELU())
        self.flag=flag
        
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*3,768),nn.GELU(),
                                    nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-5),nn.GELU())


        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)

        # self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
        #             dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        self.adapter=CLIPAdapter(config)
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1,768))
        self.rcsn=ReconstructNetwork(config)

        self.t_pj=TwoLayer(config.dims,config.dropout,config.eps)




    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,frames,cls,rel,bbx_list,task_id):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        pre_feature=self.tse(self.fusion(adapter_feature+bbx)+pos)
        target_feature=pre_feature[:,:,1:,:]
        # frames_features=
        # projection head
        frames_features=self.pj(pre_feature)
        target_feature=frames_features[:,:,1:,:]
        frames_feature=self.gpfp(frames_features,task_id)
        # frames_feature=self.tse(self.fusion(self.adapter(frames)+bbx)+pos)
        

        # total features for a consist edge cls
        human_obj_feature=frames_feature[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_feature[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))+rel_feature

        # human_feature=cls_feature[:,:,0,:].unsqueeze(-2)
        # obj_feature=cls_feature[:,:,1:,:]
        

        # common features
        c_human_obj_feature=self.cffn(human_obj_feature)
        c_human_feature=c_human_obj_feature[:,:,0,:].unsqueeze(-2)
        c_obj_feature=c_human_obj_feature[:,:,1:,:]
        
        
        # private features
        p_human_obj_feature=self.pffn(human_obj_feature)
        P_human_feature=p_human_obj_feature[:,:,0,:].unsqueeze(-2)
        p_obj_feature=p_human_obj_feature[:,:,1:,:]

        # concat
        t_human_features=torch.cat([c_human_feature,P_human_feature],dim=0)
        t_obj_features=torch.cat([c_obj_feature,p_obj_feature],dim=0)
        t_edge_features=torch.cat([edge_feature,edge_feature],dim=0)
        t_human_feature,t_obj_feature=self.gpnn(t_human_features,t_obj_features,t_edge_features)


        t_features=torch.cat([t_human_feature,t_obj_feature],dim=-2)

        p_node_features=t_features[B:,:,:,:]
        c_node_features=t_features[:B,:,:,:]

        rec=self.t_pj(p_human_obj_feature+c_human_obj_feature)
        t_node_features=p_node_features+c_node_features
        node_features=torch.cat([t_node_features,c_node_features,p_node_features],dim=0)
        construct_features=self.rcsn(rec)
        # cls_token=node_features[:,:,0,:]
        # cls_token=self.ge(cls_token)
        cls_token=self.ge(node_features)
        # cls_token_global=self.ge()
        t_ans=self.cls_head(cls_token[:B,:])
        c_ans=self.c_head(cls_token[B:2*B,:])
        p_ans=self.p_head(cls_token[2*B:,:])
        return t_ans,c_ans,p_ans,c_human_obj_feature,p_human_obj_feature,construct_features,target_feature,cls_ans,rel_ans

class GPNNMix2(nn.Module):
    # false linear true embedding
    def __init__(self, config,flag=False):
        super().__init__()
   
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        self.rel_embed=nn.Linear(30,768)


        self.tse=MixTSE(config)
        self.cgpnn=GPNN5(config)
        self.pgpnn=GPNN5(config)
        self.pgpfp=GPFPlus(config,flag)
        self.cgpfp=GPFPlus(config,flag)
        # self.pffn=MLPs(config.dims,config.dropout,config.eps)
        # self.cffn=MLPs(config.dims,config.dropout,config.eps)

        self.ge_t=DynamicFlatter(config.dims)
        self.ge_c=DynamicFlatter(config.dims)
        self.ge_p=DynamicFlatter(config.dims)

        self.ge_o=DynamicFlatter(config.dims)

        self.cls_head=CLSHead(config.dims,config.eps,config.dropout,config.cls.ag)
        self.o_head=CLSHead(config.dims,config.eps,config.dropout,config.cls.ag)
        self.p_head=CLSHead(config.dims,config.eps,config.dropout,config.cls.ag+1)
        self.c_head=CLSHead(config.dims,config.eps,config.dropout,config.cls.ag+1)

        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.ReLU(inplace=True))
        self.fusion=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.ReLU())
        self.flag=flag
        
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*3,config.dims),nn.ReLU(),
                                    nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.ReLU())


        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)

        # self.adapter=MLP(in_dim=config.adapter.pj.indims,out_dim=config.adapter.pj.outdims,
        #             dropout=config.adapter.pj.dropout,eps=config.adapter.pj.eps)
        self.adapter=CLIPAdapter(config)
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1,768))
        self.rcsn=ReconstructNetwork(config)
        self.total_pj=MLP(config.dims,config.dims,config.dropout,config.eps)

        self.gf=GateFusion(config)
        # self.total_pj=nn.Sequential(nn.Linear(config.dims,config.dims),nn.GELU(),
        #                             nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())




    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,frames,cls,rel,bbx_list,task_id):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        pre_feature=self.tse(self.fusion(adapter_feature+bbx)+pos)
        target_feature=pre_feature[:,:,1:,:]
        o_cls=self.o_head(self.ge_o(target_feature))
        # frames_features=
        # projection head
        frames_features=self.pj(pre_feature)
        target_feature=frames_features[:,:,1:,:]



        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))+rel_feature

        # human_feature=cls_feature[:,:,0,:].unsqueeze(-2)
        # obj_feature=cls_feature[:,:,1:,:]
        

        # common features
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        obj_feature=human_obj_feature[:,:,1:,:]
        
        

        # concat
        p_human_feature,p_obj_feature=self.pgpnn(human_feature,obj_feature,edge_feature,self.pgpfp,task_id)

        c_human_feature,c_obj_feature=self.cgpnn(human_feature,obj_feature,edge_feature,self.cgpfp,task_id)




        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        rec=self.gf(p_features,c_features)
        t_node_features=self.total_pj(rec)
        # node_features=torch.cat([t_node_features,c_features,p_features],dim=0)
        construct_features=self.rcsn(rec)
        # cls_token=node_features[:,:,0,:]
        # cls_token=self.ge(cls_token)
        # cls_token=self.ge(node_features)
        # cls_token_global=self.ge()
        t_ans=self.cls_head(self.ge_t(t_node_features))
        c_ans=self.c_head(self.ge_c(c_features))
        p_ans=self.p_head(self.ge_p(p_features))
        # t_ans=self.cls_head(cls_token[:B,:])
        # c_ans=self.c_head(cls_token[B:2*B,:])
        # p_ans=self.p_head(cls_token[2*B:,:])
        return t_ans,c_ans,p_ans,c_features,p_features,construct_features,target_feature,cls_ans,rel_ans,o_cls


class Head(nn.Module):
    def __init__(self, dims,eps,dropout,out_cls):
        super().__init__()
        self.dy=DynamicFlatter(dims,dropout,eps)
        self.head=CLSHead(dims,eps,dropout,out_cls)

    def set_visual(self,flag):
        self.dy.set_visual(flag)
    
    def clear_visual(self):
        self.dy.clear_visual()

    def get_visual(self):
        return self.dy.get_visual()

    def forward(self,X):
        return self.head(self.dy(X))

    def get_last_layer(self):
        return self.head.get_last_layer()

class GPNNMix3(nn.Module):
    # false linear true embedding
    # stage_1 train_stage private/common/middle
    # stage_2 train_stage total/private/common/middle
    # stage_3 inference only
    # stage_4 retrain middle,total

    def __init__(self, config,flag=False,train_stage=1):
        super().__init__()
        self.stage=train_stage
        self.flag=flag
        self.config=config
        print('train_stage',self.stage)
        if self.stage==1:
            self.model_init1(config)
        elif self.stage in [2,3]:
            self.model_init1(config)
            self.model_init2(config)
        elif self.stage==4:
            self.model_init1(config)
            self.model_init2(config)
            self.freeze()
        else:
            raise NotImplementedError
            
    # stage 1/3 supervise common private
    def model_init1(self,config):
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        self.rel_embed=nn.Linear(30,768)
        self.tse=MixTSE(config)
        self.cgpnn=GPNN5(config,config.gpnn.layer.common)
        self.pgpnn=GPNN5(config,config.gpnn.layer.private)
        if config.prompt.type==1:
            print('pgfp')
            self.pgpfp=GPFPlus(config,self.flag)
            self.cgpfp=GPFPlus(config,self.flag)
        elif config.prompt.type==0:
            self.pgpfp=SimplePrompt(config,self.flag)
            self.cgpfp=SimplePrompt(config,self.flag)
        else:
            raise NotImplementedError

        # self.cmffn=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        # self.mffn=FFN(config.dims,config.eps,config.dims*4,config.dropout)

        # self.mffn=TwoLayer(config.dims,config.dropout,config.eps)
        self.mffn=FFN(config.dims,config.eps,config.dims*4,config.dropout)

        self.mffn2=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        # self.mffn2=TwoLayer(config.dims,config.dropout,config.eps)
        # for total feature
        self.mffn3=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        # self.mffn3=TwoLayer(config.dims,config.dropout,config.eps)
        # self.cm_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)
        self.m_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)
        # self.m_head2=Head(config.dims,config.eps,config.dropout,config.cls.ag)
        self.p_head=Head(config.dims,config.eps,config.dropout,config.cls.ag+1)
        self.c_head=Head(config.dims,config.eps,config.dropout,config.cls.ag+1)
        self.bbx_linear=nn.Sequential(nn.Linear(4,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.fusion=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*3,config.dims),nn.GELU(),
                                    nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)
        self.adapter=CLIPAdapter(config)
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1,768))

        self.recs=ReconstructNetwork(config)

    # stage 2/3
    def model_init2(self,config):
        self.total_pj=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.gf=GateFusion(config)
        self.cls_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)

    def get_weight(self,p_loss,c_loss):
        p_grad = torch.autograd.grad(p_loss, self.p_head.get_last_layer(), retain_graph=True)[0]
        c_grad = torch.autograd.grad(c_loss, self.c_head.get_last_layer(), retain_graph=True)[0]

        d_weight = torch.norm(c_grad) / (torch.norm(p_grad) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        train_modules={
            4: [self.total_pj,self.cls_head,self.m_head],
            2: [self.mffn, self.m_head, self.gf, self.cls_head, self.total_pj]
        }

        for module in train_modules.get(self.stage,[]):
            for param in module.parameters():
                param.requires_grad=True



    # private common middle only
    def forward1(self,frames,cls,rel,bbx_list,task_id):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))+rel_feature

        
        # common features

        
        
        
        p_f=self.mffn(self.pgpfp(human_obj_feature,task_id)+self.cgpfp(human_obj_feature,task_id))
        # c_f=self.mffn2(self.pgpfp(human_obj_feature,task_id)+self.cgpfp(human_obj_feature,task_id))
        # c_f=self.cmffn(self.pgpfp(human_obj_feature,task_id)+self.cgpfp(human_obj_feature,task_id))
        # cm_ans=self.cm_head(c_f)
        pm_ans=self.m_head(p_f)
        # cm_ans=self.m_head2(c_f)
        # breakpoint()
        p_human_feature,p_obj_feature=self.pgpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        c_human_feature,c_obj_feature=self.cgpnn(human_feature,obj_feature,edge_feature,self.cgpfp,task_id)
        # c_human_feature,c_obj_feature=self.pgpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        # m_ans -> private common
        return c_ans,p_ans,cls_ans,rel_ans,pm_ans,c_features,p_features


    def forward2(self,frames,cls,rel,bbx_list,task_id):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=edge_feature+rel_feature

        # common features
        # human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        # obj_feature=human_obj_feature[:,:,1:,:]
        with torch.no_grad():
            p_noise=self.pgpfp.get_prompt(human_obj_feature,task_id)
            c_noise=self.cgpfp.get_prompt(human_obj_feature,task_id)
        human_obj_feature=self.mffn(human_obj_feature+p_noise+c_noise)
        pcm_ans=self.m_head(human_obj_feature)
        

        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        # c_f=self.cmffn(self.pgpfp(human_obj_feature,task_id)+self.cgpfp(human_obj_feature,task_id))
        # cm_ans=self.cm_head(c_f)
        

        # breakpoint()
        p_human_feature,p_obj_feature=self.pgpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        # c_human_feature,c_obj_feature=self.cgpnn(human_feature,obj_feature,edge_feature,self.cgpfp,task_id)
        c_human_feature,c_obj_feature=self.pgpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id)

        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        recs=self.recs(t_node)
        

        return c_ans,p_ans,cls_ans,rel_ans,pcm_ans,c_features,p_features,recs,human_obj_feature,t_ans

    # inference only
    def forward3(self,frames,cls,rel,bbx_list,task_id):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)

        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))+rel_feature

        
        # common features
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        obj_feature=human_obj_feature[:,:,1:,:]
        
        
        
        p_f=self.mffn(self.pgpfp(human_obj_feature,task_id)+self.cgpfp(human_obj_feature,task_id))
        # c_f=self.mffn2(self.pgpfp(human_obj_feature,task_id)+self.cgpfp(human_obj_feature,task_id))
        # c_f=self.cmffn(self.pgpfp(human_obj_feature,task_id)+self.cgpfp(human_obj_feature,task_id))
        # cm_ans=self.cm_head(c_f)
        pm_ans=self.m_head(p_f)
        # cm_ans=self.m_head2(c_f)
        # breakpoint()
        p_human_feature,p_obj_feature=self.pgpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        c_human_feature,c_obj_feature=self.cgpnn(human_feature,obj_feature,edge_feature,self.cgpfp,task_id)
        # c_human_feature,c_obj_feature=self.pgpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        t_ans=self.cls_head(self.total_pj(self.gf(p_features,c_features)))
        # m_ans -> private common
        return c_ans,p_ans,pm_ans,t_ans

    def forward4(self,frames,cls,rel,bbx_list,task_id):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)

        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=edge_feature+rel_feature

        # common features
        # human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        # obj_feature=human_obj_feature[:,:,1:,:]
        with torch.no_grad():
            p_noise=self.pgpfp.get_prompt(human_obj_feature,task_id)
            c_noise=self.cgpfp.get_prompt(human_obj_feature,task_id)
        human_obj_feature=self.mffn(human_obj_feature+p_noise+c_noise)
        pcm_ans=self.m_head(human_obj_feature)
        

        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        # c_f=self.cmffn(self.pgpfp(human_obj_feature,task_id)+self.cgpfp(human_obj_feature,task_id))
        # cm_ans=self.cm_head(c_f)
        

        # breakpoint()
        p_human_feature,p_obj_feature=self.pgpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        # c_human_feature,c_obj_feature=self.cgpnn(human_feature,obj_feature,edge_feature,self.cgpfp,task_id)
        c_human_feature,c_obj_feature=self.pgpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id)

        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))

        return t_ans,pcm_ans
    
    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,frames,cls,rel,bbx_list,task_id):
        if self.stage==1:
            return self.forward1(frames,cls,rel,bbx_list,task_id)
        elif self.stage==2:
            return self.forward2(frames,cls,rel,bbx_list,task_id)
        elif self.stage==3:
            return self.forward3(frames,cls,rel,bbx_list,task_id)
        elif self.stage==4:
            return self.forward4(frames,cls,rel,bbx_list,task_id)
        else:
            raise NotImplementedError

class GPNNMix3_Test(nn.Module):
    # false linear true embedding
    # stage_1 train_stage private/common/middle
    # stage_2 train_stage total/private/common/middle
    # stage_3 inference only
    # stage_4 retrain middle,total

    def __init__(self, config,flag=False,train_stage=1):
        super().__init__()
        self.stage=train_stage
        self.flag=flag
        self.config=config
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        self.rel_embed=nn.Linear(30,768)
        self.tse=MixTSE(config)
        self.cgpnn=GPNN5(config,config.gpnn.layer.common)
        self.pgpnn=GPNN5(config,4)
        if config.prompt.type==1:
            print('pgfp')
            self.pgpfp=GPFPlus(config,self.flag)
            self.cgpfp=GPFPlus(config,self.flag)
        elif config.prompt.type==0:
            self.pgpfp=SimplePrompt(config,self.flag)
            self.cgpfp=SimplePrompt(config,self.flag)
        else:
            raise NotImplementedError



        self.mffn=FFN(config.dims,config.eps,config.dims*4,config.dropout)

        self.m_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)

        self.p_head=Head(config.dims,config.eps,config.dropout,config.cls.ag+1)
        self.c_head=Head(config.dims,config.eps,config.dropout,config.cls.ag+1)
        self.bbx_linear=nn.Sequential(nn.Linear(4,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.fusion=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*3,config.dims),nn.GELU(),
                                    nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)
        self.adapter=CLIPAdapter(config)
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1,768))
        self.total_pj=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.gf=GateFusion(config)
        self.cls_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)


    # total only
    def forward(self,frames,cls,rel,bbx_list,task_id):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)

        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=edge_feature+rel_feature

        
        # common features
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        obj_feature=human_obj_feature[:,:,1:,:]
        
        
        
        p_f=self.mffn(self.pgpfp(human_obj_feature,task_id)+self.cgpfp(human_obj_feature,task_id))

        pm_ans=self.m_head(p_f)

        p_human_feature,p_obj_feature=self.pgpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        c_human_feature,c_obj_feature=self.cgpnn(human_feature,obj_feature,edge_feature,self.cgpfp,task_id)
        # c_human_feature,c_obj_feature=self.pgpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        t_ans=self.cls_head(self.total_pj(self.gf(p_features,c_features)))
        # m_ans -> private common
        return c_ans,p_ans,pm_ans,t_ans


# new no middle

class GPNNMix4(nn.Module):
    # false linear true embedding
    # stage_1 train_stage private/common/total
    # stage_2 train_stage continue total middle
    # stage_3 train_stage backbone only -> middle
    # stage_4 inference -> middle total commonm private
    # stage_5 train_stage backbone only -> middle continue
    # stage_6 single branch
    # stage_7 dual branch
    # stage_8 single branch continue
    # stage_9 dual branch continue
    # stage_10 single branch inference
    # stage_11 dual branch inference
    # stage_12 groudtruth test total out
    # stage_13 no prompt test total out
    # init_1 backbone 
    # init_2 backbone+private+common+total+middle

    def __init__(self, config,flag=False,train_stage=1,pre=False,lt=0):
        super().__init__()
        self.stage=train_stage
        self.flag=flag
        self.config=config
        self.pre=pre
        # loss type
        # 0: all loss;1: reconstruction.2.separation.3 no loss
        self.lt=lt
        print('train_stage',self.stage)
        if self.stage in [3,5]:
            self.model_init1(config)
            self.model_init3(config)
        elif self.stage in [2,4,1,6,7,8,12,13]:
            self.model_init1(config)
            self.model_init2(config)
        elif self.stage in [9]:
            self.model_init1(config)
            self.model_init2(config)
        elif self.stage in [10,11]:
            self.model_init1(config)
            self.model_init2(config)
            self.model_init3(config)
            # self.model_init3(config)
        else:
            raise NotImplementedError
        self.freeze()
    
    def set_visual(self,flag=True):
        self.gpnn.set_visual(flag)
        self.p_head.set_visual(flag)
        self.c_head.set_visual(flag)
        self.cls_head.set_visual(flag)

    def get_visual(self):
        return self.gpnn.visual(),self.p_head.get_visual(),self.c_head.get_visual(),self.cls_head.get_visual()
    def clear_visual(self):
        self.gpnn.clear_visual(),self.p_head.clear_visual(),self.c_head.clear_visual(),self.cls_head.clear_visual()
    
    def model_init1(self,config):
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        self.rel_embed=nn.Linear(30,768)
        self.tse=MixTSE(config)
        # self.feature_mergin=nn.Sequential(nn.Linear(config.dims*2,config.dims),nn.LayerNorm(config.dims),nn.GELU())


        self.mffn=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        
        
        self.bbx_linear=nn.Sequential(nn.Linear(4,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.fusion=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*3,config.dims),nn.GELU(),
                                    nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)
        self.adapter=CLIPAdapter(config)
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        self.pos = nn.Parameter(torch.zeros(1,config.frames,1,config.dims))
        # self.mergin_feature=nn.Sequential(nn.Linear(config.dims*2,config.dims),nn.LayerNorm(config.dims),nn.GELU())

    # stage 2/3
    def model_init2(self,config):
        if config.prompt.type==1:
            print('pgfp')
            self.pgpfp=GPFPlus(config,self.flag)
            self.cgpfp=GPFPlus(config,self.flag)
        elif config.prompt.type==0:
            self.pgpfp=SimplePrompt(config,self.flag)
            self.cgpfp=SimplePrompt(config,self.flag)
        else:
            raise NotImplementedError
        self.gpnn=GPNN5(config,config.gpnn.layer.one)
        # self.m_head2=Head(config.dims,config.eps,config.dropout,config.cls.ag)

        self.mffn2=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.mffn3=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        if self.stage not in [9]:
            self.p_head=Head(config.dims,config.eps,config.dropout,config.cls.ag+1)
            self.c_head=Head(config.dims,config.eps,config.dropout,config.cls.ag+1)
        if self.lt<2:
            self.recs=ReconstructNetwork(config)
        self.total_pj=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.gf=GateFusion(config)
        self.cls_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)

    # stage 2/3
    def model_init3(self,config):
        self.m_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)


    def get_weight(self,p_loss,c_loss):
        if self.stage in [6]:
            p_grad = torch.autograd.grad(p_loss, self.c_head.get_last_layer(), retain_graph=True)[0]
        else:
            p_grad = torch.autograd.grad(p_loss, self.p_head.get_last_layer(), retain_graph=True)[0]
        c_grad = torch.autograd.grad(c_loss, self.c_head.get_last_layer(), retain_graph=True)[0]

        d_weight = torch.norm(c_grad) / (torch.norm(p_grad) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
    
    def freeze(self):
        if self.stage in [1,3,4,6,7,10,11,12,13]:
            return
        for param in self.parameters():
            param.requires_grad = False
        # 2 total/middle
        # 3 middle backbone only
        train_modules={
            2: ['total_pj','cls_head'],
            8: ['total_pj','cls_head'],
            9: ['total_pj','cls_head'],
            5: ['m_head'],
            #2: [self.mffn, self.m_head, self.gf, self.cls_head, self.total_pj]
        }
        # modules=[m for m in getattr(self,module_name)]
        modules=[getattr(self,module_name) for module_name in train_modules.get(self.stage,[])]
        for module in modules:
            for param in module.parameters():
                param.requires_grad=True

    # private common total
    def forward1(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        

        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        recs=self.recs(t_node)
        

        return c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans
    # total middle
    def forward2(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)

        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        m_ans=self.m_head(human_obj_feature)
        

        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))

        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        # c_human_feature,c_obj_feature=self.cgpnn(human_feature,obj_feature,edge_feature,self.cgpfp,task_id)
        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id)

        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        

        return t_ans,m_ans

    # backbone only
    def forward3(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)

        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]

        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)

        human_obj_feature=self.mffn(human_obj_feature)
        m_ans=self.m_head(human_obj_feature)
        
        return m_ans,cls_ans,rel_ans
    # inference -> middle/common/private/total
    def forward4(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        # m_ans=self.m_head(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)
        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)
        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)
        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        # t_node=self.mergin_feature(torch.cat([p_features,c_features],dim=-1))
        # t_node=self.feature_mergin(torch.cat([p_features,c_features],dim=-1))
        t_ans=self.cls_head(self.total_pj(t_node))
        

        return c_ans,p_ans,t_ans
    # backbone only continue
    def forward5(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()

        cls_feature=self.cls_embed(cls)

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)

        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]+cls_feature
        # supevised by adapter feature
        human_obj_feature=human_obj_feature
        human_obj_feature=self.mffn(human_obj_feature)
        m_ans=self.m_head(human_obj_feature)
        return m_ans
 
  # single branch
    def forward6(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        task_id=torch.cat([task_id,(~task_id.bool()).float()],dim=0)
        nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)
        

        # p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        # c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,task_id))
        edge_feature=torch.cat([edge_feature,edge_feature],dim=0)
        
        pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        # pc
        pc_feature=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)
        p_features=pc_feature[B:,:,:,:]

        c_features=pc_feature[:B,:,:,:]


        pc_ans=self.c_head(pc_feature)
        p_ans=pc_ans[B:,:]
        c_ans=pc_ans[:B,:]
        # p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        recs=self.recs(t_node)
        

        return c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans
    # dual branch
    def forward7(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)
        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)
        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)
        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        # t_node=self.mergin_feature(torch.cat([p_features,c_features],dim=-1))
        # t_node=self.feature_mergin(torch.cat([p_features,c_features],dim=-1))
        t_ans=self.cls_head(self.total_pj(t_node))
        if self.lt<2:
            recs=self.recs(t_node)
        else:
            recs=[]
        return c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans
    # total middle
  # single branch
    def forward8(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        with torch.no_grad():
            # nums =node+1
            B,Frame,Nums,dims=frames.shape
            # breakpoint()
            Nums=Nums
            # bbx_list=bbx_list[:,:,1:,:]
            bbx=self.bbx_linear(bbx_list)
            # breakpoint()
            cls_feature=self.cls_embed(cls)
            rel_feature=self.rel_embed(rel)
            

            pos=self.pos.repeat(B,1,Nums,1)
            adapter_feature=self.adapter(frames)
            # supervised
            # pre_feature=
    
            # frames_features=
            # projection head
            frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
            # breakpoint()

            # total features for a consist edge cls
            human_obj_feature=frames_features[:,:,1:,:]
            # supevised by adapter feature
            # scene graph
            human_obj_feature=human_obj_feature+cls_feature
            #no scenegraph
            # human_obj_feature=human_obj_feature
            human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
            human_features=human_feature.repeat(1,1,Nums-2,1)
            obj_feature=human_obj_feature[:,:,1:,:]
            global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
            edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
    
            # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

            # scene graph
            edge_feature=edge_feature+rel_feature

            human_obj_feature=self.mffn(human_obj_feature)
            task_id=torch.cat([task_id,(~task_id.bool()).float()],dim=0)
            nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)
            

            # p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
            # c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
            pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,task_id))
            edge_feature=torch.cat([edge_feature,edge_feature],dim=0)
            
            pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


            # pc
            pc_feature=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)
            p_features=pc_feature[B:,:,:,:]

            c_features=pc_feature[:B,:,:,:]

            # p_ans=self.p_head(p_features)
            # rec=
            # t_node_features=
            t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        m_ans=self.m_head(human_obj_feature)

        

        return t_ans,m_ans
    # dual branch
    def forward9(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
       
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        # pre_feature=

        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        # m_ans=self.m_head(human_obj_feature)
        

        return t_ans
    # total middle
  # single branch
    @torch.no_grad()
    def forward10(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        task_id=torch.cat([task_id,(~task_id.bool()).float()],dim=0)
        # batch frame node dims
        nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)
        

        # p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        # c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,task_id))
        edge_feature=torch.cat([edge_feature,edge_feature],dim=0)
        #obj: batch*2 frame node-1 dims / human:batch*2 frame 1 dims


        pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        # pc
        pc_feature=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)
        p_features=pc_feature[B:,:,:,:]

        c_features=pc_feature[:B,:,:,:]
        pc_ans=self.c_head(pc_feature)
        
        p_ans=pc_ans[B:,:]
        c_ans=pc_ans[:B,:]
        # p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))

        m_ans=self.m_head(human_obj_feature)
        

        return c_ans,p_ans,t_ans,m_ans
    # dual branch
    @torch.no_grad()
    def forward11(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))

        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        # scene graph
        human_obj_feature=human_obj_feature+cls_feature

        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
 
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)
        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)
        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)

        t_node=self.gf(p_features,c_features)
        # t_node=self.mergin_feature(torch.cat([p_features,c_features],dim=-1))
        # t_node=self.feature_mergin(torch.cat([p_features,c_features],dim=-1))
        t_ans=self.cls_head(self.total_pj(t_node))
        m_ans=self.m_head(human_feature)
        return t_ans,m_ans
    # ground truth test
    def forward12(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        # common id task id
        pc_id=torch.cat(task_id,dim=0)
        nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)

        pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,pc_id))
        
        pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],
                                                  torch.cat([edge_feature,edge_feature],dim=0),self.cgpfp,pc_id,mask,tfm_mask)



        pc_features=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)



        pc_ans=self.c_head(pc_features)
        # rec=
        # t_node_features=
        t_node=self.gf(pc_features[:B,:,:,:],pc_features[B:,:,:,:])
        t_ans=self.cls_head(self.total_pj(t_node))
        recs=self.recs(t_node)
        
        return pc_ans[:B:,:],pc_ans[B:,:],cls_ans,rel_ans,pc_features[:B,:,:,:],pc_features[B:,:,:,:],recs,human_obj_feature,t_ans
    # no prompt test
    def forward13(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        pc_feature=self.mffn2(human_obj_feature)
        
        p_human_feature,p_obj_feature=self.gpnn(pc_feature[:,:,0,:].unsqueeze(-2),pc_feature[:,:,1:,:],edge_feature,None,task_id,mask,tfm_mask)



        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)


        # rec=
        # t_node_features=
        t_ans=self.cls_head(self.total_pj(p_features))
        return cls_ans,rel_ans,t_ans
    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        forwards=[   
                 self.forward1,
                 self.forward2,
                 self.forward3,
                 self.forward4,
                 self.forward5,
                 self.forward6,
                 self.forward7,
                 self.forward8,
                 self.forward9,
                 self.forward10,
                 self.forward11,
                 self.forward12,
                 self.forward13]
        return forwards[self.stage-1](frames,cls,rel,bbx_list,task_id,mask,tfm_mask)

#cad 120
class GPNNMix5(nn.Module):

    def __init__(self, config,flag=False,train_stage=1,pre=False,lt=0):
        super().__init__()
        self.stage=train_stage
        self.flag=flag
        self.config=config
        self.pre=pre
        # loss type
        # 0: all loss;1: reconstruction.2.separation.3 no loss
        self.lt=lt
        print('train_stage',self.stage)
        if self.stage in [3,5]:
            self.model_init1(config)
            self.model_init3(config)
        elif self.stage in [2,1,4,6,7,8,12,13]:
            self.model_init1(config)
            self.model_init2(config)
        elif self.stage in [9]:
            self.model_init1(config)
            self.model_init2(config)
        elif self.stage in [10,11]:
            self.model_init1(config)
            self.model_init2(config)
            self.model_init3(config)
        else:
            raise NotImplementedError
        self.freeze()
    
    def set_visual(self,flag=True):
        self.gpnn.set_visual(flag)
        self.p_head.set_visual(flag)
        self.c_head.set_visual(flag)
        self.cls_head.set_visual(flag)

    def get_visual(self):
        return self.gpnn.visual(),self.p_head.get_visual(),self.c_head.get_visual(),self.cls_head.get_visual()
    
    def model_init1(self,config):
        self.cls_embed=nn.Embedding(13,768,padding_idx=0)
        # self.rel_embed=nn.Linear(4,768)
        self.tse=MixTSE(config)
        # self.recs=ReconstructNetwork(config)
        self.prompt=SinglePrompt(config)
        self.mffn=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        
        
        self.bbx_linear=nn.Sequential(nn.Linear(4,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.fusion=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())

        # self.rel=nn.Sequential(nn.Flatten(start_dim=-2,end_dim=-1),nn.Linear(49,1),nn.GELU())
        # self.rel2=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(1024,768),nn.LayerNorm(768),nn.GELU())
        # self.obj=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(2048,768),nn.LayerNorm(768),nn.GELU())
        self.rel=nn.Sequential(nn.Flatten(start_dim=-2,end_dim=-1),nn.Linear(49,1),nn.GELU())
        self.rel2=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(1024,768),nn.LayerNorm(768),nn.GELU())
        self.obj=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(2048,768),nn.LayerNorm(768),nn.GELU())
        # self.edge_f=nn.Sequential(nn.Linear(768,768),nn.LayerNorm(768),nn.GELU())
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        self.pos = nn.Parameter(torch.zeros(1,config.frames,1,config.dims))
        # self.mergin_feature=nn.Sequential(nn.Linear(config.dims*2,config.dims),nn.LayerNorm(config.dims),nn.GELU())

    # stage 2/3
    def model_init2(self,config):
        # if config.prompt.type==1:
        #     print('pgfp')
        #     self.pgpfp=GPFPlus(config,self.flag)
        #     self.cgpfp=GPFPlus(config,self.flag)
        # elif config.prompt.type==0:
        #     self.pgpfp=SimplePrompt(config,self.flag)
        #     self.cgpfp=SimplePrompt(config,self.flag)
        # else:
        #     raise NotImplementedError
        self.gpnn=GPNN5(config,config.gpnn.layer.one)
        # self.m_head2=Head(config.dims,config.eps,config.dropout,config.cls.ag)

        # self.mffn2=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        # self.mffn3=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        # if self.stage not in [9]:
        #     self.p_head=Head(config.dims,config.eps,config.dropout,config.cad.cls)
        #     self.c_head=Head(config.dims,config.eps,config.dropout,config.cad.cls)
        # if self.lt<2:
        #     self.recs=ReconstructNetwork(config)
        self.total_pj=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        # self.gf=GateFusion(config)
        self.cls_head=Head(config.dims,config.eps,config.dropout,config.cad.cls)

    # stage 2/3
    def model_init3(self,config):
        self.m_head=Head(config.dims,config.eps,config.dropout,config.cad.cls)


    def get_weight(self,p_loss,c_loss):
        if self.stage in [6]:
            p_grad = torch.autograd.grad(p_loss, self.c_head.get_last_layer(), retain_graph=True)[0]
        else:
            p_grad = torch.autograd.grad(p_loss, self.p_head.get_last_layer(), retain_graph=True)[0]
        c_grad = torch.autograd.grad(c_loss, self.c_head.get_last_layer(), retain_graph=True)[0]

        d_weight = torch.norm(c_grad) / (torch.norm(p_grad) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
    
    def freeze(self):
        if self.stage in [1,3,4,6,7,10,11,12,13]:
            return
        for param in self.parameters():
            param.requires_grad = False
        # 2 total/middle
        # 3 middle backbone only
        train_modules={
            2: ['total_pj','cls_head'],
            8: ['total_pj','cls_head'],
            9: ['total_pj','cls_head'],
            5: ['m_head'],
            #2: [self.mffn, self.m_head, self.gf, self.cls_head, self.total_pj]
        }
        # modules=[m for m in getattr(self,module_name)]
        modules=[getattr(self,module_name) for module_name in train_modules.get(self.stage,[])]
        for module in modules:
            for param in module.parameters():
                param.requires_grad=True
    # dual branch
    def forward7(self,frames,cls,rel,bbx_list):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        # obj/
        # bbx_list=[batch,frame,node+rela,4]
        # max objects 11
        bbx=self.bbx_linear(bbx_list)
        bbx1=bbx[:,:,:11,:]
        bbx2=bbx[:,:,11:,:]
        # breakpoint()
        cls_feature=self.cls_embed(cls).squeeze(-2)
        # rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.obj(frames)
        # supervised
        # pre_feature=
 
        # frames_features=
        # projection head
        human_obj_feature=self.pj(self.tse(self.fusion(adapter_feature+bbx1+cls_feature))+pos)
        # breakpoint()
        edge_feature=self.rel2(self.rel(rel).squeeze(-1))
        edge_feature=edge_feature+bbx2

        # total features for a consist edge cls
        # human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        # scene graph
        # breakpoint()
        #no scenegraph
        # human_obj_feature=human_obj_feature
        # scene graph
        edge_feature=edge_feature+bbx2

        human_obj_feature=self.mffn(self.prompt(human_obj_feature,[]))

        
        # task_id2=(~task_id.bool()).float()
        # p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        # c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        # p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        # c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)
        c_human_feature,c_obj_feature=self.gpnn(human_obj_feature[:,:,0,:].unsqueeze(-2),human_obj_feature[:,:,1:,:],edge_feature,self.prompt,[])
        # p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)
        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)
        # c_ans=self.c_head(c_features)
        # p_ans=self.p_head(p_features)
        # t_node=self.gf(p_features,c_features)
        # t_ans=self.cls_head(self.total_pj(t_node))
        # if self.lt<2:
        #     recs=self.recs(t_node)
        # else:
        #     recs=[]
        # recs=self.recs(c_features)
        t_ans=self.cls_head(self.total_pj(c_features))
        return t_ans
    # total middle
  # single branch
   # dual branch
    def forward9(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
       
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        # pre_feature=

        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        # m_ans=self.m_head(human_obj_feature)
        

        return t_ans
    # total middle
    # single branch
    # dual branch
    @torch.no_grad()
    def forward11(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))

        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        # scene graph
        human_obj_feature=human_obj_feature+cls_feature

        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
 
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)
        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)
        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)

        t_node=self.gf(p_features,c_features)
        # t_node=self.mergin_feature(torch.cat([p_features,c_features],dim=-1))
        # t_node=self.feature_mergin(torch.cat([p_features,c_features],dim=-1))
        t_ans=self.cls_head(self.total_pj(t_node))
        m_ans=self.m_head(human_feature)
        return t_ans,m_ans

    
    # ground truth test
    def forward12(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        # common id task id
        pc_id=torch.cat(task_id,dim=0)
        nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)

        pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,pc_id))
        
        pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],
                                                  torch.cat([edge_feature,edge_feature],dim=0),self.cgpfp,pc_id,mask,tfm_mask)



        pc_features=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)



        pc_ans=self.c_head(pc_features)
        # rec=
        # t_node_features=
        t_node=self.gf(pc_features[:B,:,:,:],pc_features[B:,:,:,:])
        t_ans=self.cls_head(self.total_pj(t_node))
        recs=self.recs(t_node)
        
        return pc_ans[:B:,:],pc_ans[B:,:],cls_ans,rel_ans,pc_features[:B,:,:,:],pc_features[B:,:,:,:],recs,human_obj_feature,t_ans
    
    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,frames,cls,rel,bbx_list):
        forwards=[  '',
                    '',
                    '',
                    '',
                    '',
                    '',
                 self.forward7,
                 '',
                 self.forward9,
                 self.forward11,
                 self.forward12,
                 ]
        return forwards[self.stage-1](frames,cls,rel,bbx_list)

class GPNNMix5_1(nn.Module):
    # false linear true embedding
    # stage_1 train_stage private/common/total
    # stage_2 train_stage continue total middle
    # stage_3 train_stage backbone only -> middle
    # stage_4 inference -> middle total commonm private
    # stage_5 train_stage backbone only -> middle continue
    # stage_6 single branch
    # stage_7 dual branch
    # stage_8 single branch continue
    # stage_9 dual branch continue
    # stage_10 single branch inference
    # stage_11 dual branch inference
    # stage_12 groudtruth test total out
    # stage_13 no prompt test total out
    # init_1 backbone 
    # init_2 backbone+private+common+total+middle

    def __init__(self, config,flag=False,train_stage=1,pre=False,lt=0):
        super().__init__()
        self.stage=train_stage
        self.flag=flag
        self.config=config
        self.pre=pre
        # loss type
        # 0: all loss;1: reconstruction.2.separation.3 no loss
        self.lt=lt
        print('train_stage',self.stage)
        if self.stage in [3,5]:
            self.model_init1(config)
            self.model_init3(config)
        elif self.stage in [2,1,4,6,7,8,12,13]:
            self.model_init1(config)
            self.model_init2(config)
        elif self.stage in [9]:
            self.model_init1(config)
            self.model_init2(config)
        elif self.stage in [10,11]:
            self.model_init1(config)
            self.model_init2(config)
            self.model_init3(config)
            # self.model_init3(config)
        else:
            raise NotImplementedError
        self.freeze()
    
    def set_visual(self,flag=True):
        self.gpnn.set_visual(flag)
        self.p_head.set_visual(flag)
        self.c_head.set_visual(flag)
        self.cls_head.set_visual(flag)

    def get_visual(self):
        return self.gpnn.visual(),self.p_head.get_visual(),self.c_head.get_visual(),self.cls_head.get_visual()
    
    def model_init1(self,config):
        self.cls_embed=nn.Embedding(13,768,padding_idx=0)
        # self.obj=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(2048,768),nn.LayerNorm(768),nn.GELU())
        self.rel=nn.Sequential(nn.Flatten(start_dim=-2,end_dim=-1),nn.Linear(49,1),nn.GELU())
        self.rel2=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(1024,768),nn.LayerNorm(768),nn.GELU())
        self.obj=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(2048,768),nn.LayerNorm(768),nn.GELU())
        self.tse=MixTSE(config)
        # self.feature_mergin=nn.Sequential(nn.Linear(config.dims*2,config.dims),nn.LayerNorm(config.dims),nn.GELU())


        self.mffn=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        
        
        self.bbx_linear=nn.Sequential(nn.Linear(4,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.fusion=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())

        self.edge_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*3,config.dims),nn.GELU(),
                                    nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        self.pos = nn.Parameter(torch.zeros(1,config.frames,1,config.dims))
        # self.mergin_feature=nn.Sequential(nn.Linear(config.dims*2,config.dims),nn.LayerNorm(config.dims),nn.GELU())

    # stage 2/3
    def model_init2(self,config):
        if config.prompt.type==1:
            print('pgfp')
            self.pgpfp=SinglePrompt(config,self.flag)
            self.cgpfp=SinglePrompt(config,self.flag)
        elif config.prompt.type==0:
            self.pgpfp=SimplePrompt(config,self.flag)
            self.cgpfp=SimplePrompt(config,self.flag)
        else:
            raise NotImplementedError
        self.gpnn=GPNN5(config,config.gpnn.layer.one)
        # self.m_head2=Head(config.dims,config.eps,config.dropout,config.cls.ag)

        self.mffn2=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.mffn3=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.p_head=Head(config.dims,config.eps,config.dropout,config.cad.cls)
        self.c_head=Head(config.dims,config.eps,config.dropout,config.cad.cls)

        self.recs=ReconstructNetwork(config)
        self.total_pj=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.gf=GateFusion(config)
        self.cls_head=Head(config.dims,config.eps,config.dropout,config.cad.cls)

    # stage 2/3
    def model_init3(self,config):
        self.m_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)


    def get_weight(self,p_loss,c_loss):
        if self.stage in [6]:
            p_grad = torch.autograd.grad(p_loss, self.c_head.get_last_layer(), retain_graph=True)[0]
        else:
            p_grad = torch.autograd.grad(p_loss, self.p_head.get_last_layer(), retain_graph=True)[0]
        c_grad = torch.autograd.grad(c_loss, self.c_head.get_last_layer(), retain_graph=True)[0]

        d_weight = torch.norm(c_grad) / (torch.norm(p_grad) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight
    
    def freeze(self):
        if self.stage in [1,3,4,6,7,10,11,12,13]:
            return
        for param in self.parameters():
            param.requires_grad = False
        # 2 total/middle
        # 3 middle backbone only
        train_modules={
            2: ['total_pj','cls_head'],
            8: ['total_pj','cls_head'],
            9: ['total_pj','cls_head'],
            5: ['m_head'],
            #2: [self.mffn, self.m_head, self.gf, self.cls_head, self.total_pj]
        }
        # modules=[m for m in getattr(self,module_name)]
        modules=[getattr(self,module_name) for module_name in train_modules.get(self.stage,[])]
        for module in modules:
            for param in module.parameters():
                param.requires_grad=True

    # private common total
    def forward1(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        

        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        recs=self.recs(t_node)
        

        return c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans
    # total middle
    def forward2(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)

        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        m_ans=self.m_head(human_obj_feature)
        

        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))

        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        # c_human_feature,c_obj_feature=self.cgpnn(human_feature,obj_feature,edge_feature,self.cgpfp,task_id)
        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id)

        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        

        return t_ans,m_ans

    # backbone only
    def forward3(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)

        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]

        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        human_obj_feature=human_obj_feature+cls_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)

        human_obj_feature=self.mffn(human_obj_feature)
        m_ans=self.m_head(human_obj_feature)
        
        return m_ans,cls_ans,rel_ans
    # inference -> middle/common/private/total
    def forward4(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        m_ans=self.m_head(human_obj_feature)

        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))

        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id)

        # c_human_feature,c_obj_feature=self.cgpnn(human_feature,obj_feature,edge_feature,self.cgpfp,task_id)
        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id)

        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        return c_ans,p_ans,t_ans,m_ans
    # backbone only continue
    def forward5(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()

        cls_feature=self.cls_embed(cls)

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)

        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]+cls_feature
        # supevised by adapter feature
        human_obj_feature=human_obj_feature
        human_obj_feature=self.mffn(human_obj_feature)
        m_ans=self.m_head(human_obj_feature)
        return m_ans
 
  # single branch
    def forward6(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        task_id=torch.cat([task_id,(~task_id.bool()).float()],dim=0)
        nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)
        

        # p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        # c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,task_id))
        edge_feature=torch.cat([edge_feature,edge_feature],dim=0)
        
        pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        # pc
        pc_feature=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)
        p_features=pc_feature[B:,:,:,:]

        c_features=pc_feature[:B,:,:,:]


        pc_ans=self.c_head(pc_feature)
        p_ans=pc_ans[B:,:]
        c_ans=pc_ans[:B,:]
        # p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        recs=self.recs(t_node)
        

        return c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans
    # dual branch
    def forward7(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        B,Frame,Nums,dims=frames.shape


        bbx=self.bbx_linear(bbx_list)
        bbx1=bbx[:,:,:11,:]
        bbx2=bbx[:,:,11:,:]
        # breakpoint()
        cls_feature=self.cls_embed(cls).squeeze(-2)
        # rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.obj(frames)
        # supervised
        # pre_feature=
 
        # frames_features=
        # projection head
        human_obj_feature=self.pj(self.tse(self.fusion(adapter_feature+bbx1+cls_feature))+pos)
        # breakpoint()
        edge_feature=self.rel2(self.rel(rel).squeeze(-1))
        edge_feature=edge_feature+bbx2

        
        human_obj_feature=self.mffn(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)
        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)
        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)
        c_ans=self.c_head(c_features)
        p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        # t_node=self.mergin_feature(torch.cat([p_features,c_features],dim=-1))
        # t_node=self.feature_mergin(torch.cat([p_features,c_features],dim=-1))
        t_ans=self.cls_head(self.total_pj(t_node))
        if self.lt<2:
            recs=self.recs(t_node)
        else:
            recs=[]
        return c_ans,p_ans,c_features,p_features,recs,human_obj_feature,t_ans
    # total middle
  # single branch
    def forward8(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        with torch.no_grad():
            # nums =node+1
            B,Frame,Nums,dims=frames.shape
            # breakpoint()
            Nums=Nums
            # bbx_list=bbx_list[:,:,1:,:]
            bbx=self.bbx_linear(bbx_list)
            # breakpoint()
            cls_feature=self.cls_embed(cls)
            rel_feature=self.rel_embed(rel)
            

            pos=self.pos.repeat(B,1,Nums,1)
            adapter_feature=self.adapter(frames)
            # supervised
            # pre_feature=
    
            # frames_features=
            # projection head
            frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
            # breakpoint()

            # total features for a consist edge cls
            human_obj_feature=frames_features[:,:,1:,:]
            # supevised by adapter feature
            # scene graph
            human_obj_feature=human_obj_feature+cls_feature
            #no scenegraph
            # human_obj_feature=human_obj_feature
            human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
            human_features=human_feature.repeat(1,1,Nums-2,1)
            obj_feature=human_obj_feature[:,:,1:,:]
            global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
            edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
    
            # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

            # scene graph
            edge_feature=edge_feature+rel_feature

            human_obj_feature=self.mffn(human_obj_feature)
            task_id=torch.cat([task_id,(~task_id.bool()).float()],dim=0)
            nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)
            

            # p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
            # c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
            pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,task_id))
            edge_feature=torch.cat([edge_feature,edge_feature],dim=0)
            
            pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


            # pc
            pc_feature=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)
            p_features=pc_feature[B:,:,:,:]

            c_features=pc_feature[:B,:,:,:]

            # p_ans=self.p_head(p_features)
            # rec=
            # t_node_features=
            t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        m_ans=self.m_head(human_obj_feature)

        

        return t_ans,m_ans
    # dual branch
    def forward9(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
       
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        # pre_feature=

        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)

        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)


        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))
        # m_ans=self.m_head(human_obj_feature)
        

        return t_ans
    # total middle
  # single branch
    @torch.no_grad()
    def forward10(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))

        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        task_id=torch.cat([task_id,(~task_id.bool()).float()],dim=0)
        # batch frame node dims
        nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)
        

        # p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id))
        # c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,task_id))
        edge_feature=torch.cat([edge_feature,edge_feature],dim=0)
        #obj: batch*2 frame node-1 dims / human:batch*2 frame 1 dims


        pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)


        # pc
        pc_feature=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)
        p_features=pc_feature[B:,:,:,:]

        c_features=pc_feature[:B,:,:,:]
        pc_ans=self.c_head(pc_feature)
        
        p_ans=pc_ans[B:,:]
        c_ans=pc_ans[:B,:]
        # p_ans=self.p_head(p_features)
        # rec=
        # t_node_features=
        t_node=self.gf(p_features,c_features)
        t_ans=self.cls_head(self.total_pj(t_node))

        m_ans=self.m_head(human_obj_feature)
        

        return c_ans,p_ans,t_ans,m_ans
    # dual branch
    @torch.no_grad()
    def forward11(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))

        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature

        # scene graph
        human_obj_feature=human_obj_feature+cls_feature

        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
 
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)

        
        task_id2=(~task_id.bool()).float()
        p_f=self.mffn2(self.pgpfp(human_obj_feature,task_id2))
        c_f=self.mffn3(self.cgpfp(human_obj_feature,task_id))
        
        p_human_feature,p_obj_feature=self.gpnn(p_f[:,:,0,:].unsqueeze(-2),p_f[:,:,1:,:],edge_feature,self.pgpfp,task_id2,mask,tfm_mask)

        c_human_feature,c_obj_feature=self.gpnn(c_f[:,:,0,:].unsqueeze(-2),c_f[:,:,1:,:],edge_feature,self.cgpfp,task_id,mask,tfm_mask)
        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)
        c_features=torch.cat([c_human_feature,c_obj_feature],dim=-2)

        t_node=self.gf(p_features,c_features)
        # t_node=self.mergin_feature(torch.cat([p_features,c_features],dim=-1))
        # t_node=self.feature_mergin(torch.cat([p_features,c_features],dim=-1))
        t_ans=self.cls_head(self.total_pj(t_node))
        m_ans=self.m_head(human_feature)
        return t_ans,m_ans

    
    # ground truth test
    def forward12(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        # common id task id
        pc_id=torch.cat(task_id,dim=0)
        nhuman_obj_feature=torch.cat([human_obj_feature,human_obj_feature],dim=0)

        pc_f=self.mffn2(self.cgpfp(nhuman_obj_feature,pc_id))
        
        pc_human_feature,pc_obj_feature=self.gpnn(pc_f[:,:,0,:].unsqueeze(-2),pc_f[:,:,1:,:],
                                                  torch.cat([edge_feature,edge_feature],dim=0),self.cgpfp,pc_id,mask,tfm_mask)



        pc_features=torch.cat([pc_human_feature,pc_obj_feature],dim=-2)



        pc_ans=self.c_head(pc_features)
        # rec=
        # t_node_features=
        t_node=self.gf(pc_features[:B,:,:,:],pc_features[B:,:,:,:])
        t_ans=self.cls_head(self.total_pj(t_node))
        recs=self.recs(t_node)
        
        return pc_ans[:B:,:],pc_ans[B:,:],cls_ans,rel_ans,pc_features[:B,:,:,:],pc_features[B:,:,:,:],recs,human_obj_feature,t_ans
    
    # no prompt test
    def forward13(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls)
        rel_feature=self.rel_embed(rel)
        

        pos=self.pos.repeat(B,1,Nums,1)
        adapter_feature=self.adapter(frames)
        # supervised
        adapter_humam=adapter_feature[:,:,1:,:]
        # pre_feature=
 
        # frames_features=
        # projection head
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos,tfm_mask))
        # breakpoint()

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        cls_ans=self.obj_mlp(adapter_humam)
        # scene graph
        human_obj_feature=human_obj_feature+cls_feature
        #no scenegraph
        # human_obj_feature=human_obj_feature
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        # print('shap',human_features.shape,global_feature.shape,obj_feature.shape)

        # scene graph
        edge_feature=edge_feature+rel_feature

        human_obj_feature=self.mffn(human_obj_feature)
        pc_feature=self.mffn2(human_obj_feature)
        
        p_human_feature,p_obj_feature=self.gpnn(pc_feature[:,:,0,:].unsqueeze(-2),pc_feature[:,:,1:,:],edge_feature,None,task_id,mask,tfm_mask)



        p_features=torch.cat([p_human_feature,p_obj_feature],dim=-2)


        # rec=
        # t_node_features=
        t_ans=self.cls_head(self.total_pj(p_features))
        return cls_ans,rel_ans,t_ans
   
    # add [0.,0.,1.,1.] to the first line of every batch 
    def forward(self,frames,cls,rel,bbx_list,task_id,mask=None,tfm_mask=None):
        forwards=[   
                 self.forward1,
                 self.forward2,
                 self.forward3,
                 self.forward4,
                 self.forward5,
                 self.forward6,
                 self.forward7,
                 self.forward8,
                 self.forward9,
                 self.forward10,
                 self.forward11,
                 self.forward12,
                 self.forward13]
        return forwards[self.stage-1](frames,cls,rel,bbx_list,task_id,mask,tfm_mask)



class PureMix(nn.Module):
    # stage 1/3 supervise common private
    def __init__(self,config):
        super().__init__()
        self.tse=MixTSE(config)
        self.cls_embed=nn.Embedding(38,768,padding_idx=0)

        self.mffn=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        # self.m_head=CLSHead(config.dims,config.eps,config.dropout,config.cls.ag)
        # self.m_dy=DynamicFlatterWithGate(config,config.dims,config.dropout,config.eps)
        self.m_head=Head(config.dims,config.eps,config.dropout,config.cls.ag)
        self.pgpfp=GPFPlus(config,self.flag)
        self.cgpfp=GPFPlus(config,self.flag)
        self.bbx_linear=nn.Sequential(nn.Linear(4,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.fusion=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.rel_mlp=MLPCLS(config.dims,config.cls.rel,config.dropout,config.eps)

        self.edge_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*3,config.dims),nn.GELU(),
                                    nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())
        self.obj_mlp=MLPCLS(config.dims,config.cls.obj,config.dropout,config.eps)
        self.adapter=CLIPAdapter(config)
        self.pj=ProjectionHeadFT(config.dims,config.eps,config.dims*4,config.dropout)
        # front embedding
        self.pos = nn.Parameter(torch.zeros(1,16,1,768))

    def forward(self,frames,cls_ids,bbx_list):
        # nums =node+1
        B,Frame,Nums,dims=frames.shape
        # breakpoint()
        Nums=Nums
        # bbx_list=bbx_list[:,:,1:,:]
        bbx=self.bbx_linear(bbx_list)
        # breakpoint()
        cls_feature=self.cls_embed(cls_ids)

        pos=self.pos.repeat(B,1,Nums,1)
        # frames=frames
        adapter_feature=self.adapter(frames)
        # supervised
        # adapter_cls=adapter_feature[:,:,1:,:]
        # adapter_humam=adapter_cls[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        # adapter_objs=adapter_cls[:,:,1:,:]
        # adapter_global=adapter_feature[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)

        # edge_feature=self.edge_fun(torch.cat([adapter_cls[:,:,0,:].unsqueeze(-2),adapter_global,adapter_objs],dim=-1))
        # edge_feature=self.edge_fun(torch.cat([adapter_humam,adapter_objs],dim=-1))
        # pre_feature=
 
        # frames_features=
        # projection head
        # breakpoint()
        frames_features=self.pj(self.tse(self.fusion(adapter_feature+bbx)+pos))

        # total features for a consist edge cls
        human_obj_feature=frames_features[:,:,1:,:]
        # supevised by adapter feature
        # cls_ans=self.obj_mlp(adapter_cls)
        cls_ans=self.obj_mlp(human_obj_feature)
        human_feature=human_obj_feature[:,:,0,:].unsqueeze(-2)
        human_features=human_feature.repeat(1,1,Nums-2,1)
        obj_feature=human_obj_feature[:,:,1:,:]
        global_feature=frames_features[:,:,0,:].unsqueeze(-2).repeat(1,1,Nums-2,1)
        edge_feature=self.edge_fun(torch.cat([human_features,global_feature,obj_feature],dim=-1))
        rel_ans=self.rel_mlp(edge_feature)
        m_ans=self.m_head(self.mffn(human_obj_feature+cls_feature))
        # breakpoint()


        return m_ans,rel_ans,cls_ans


class TextT2(nn.Module):

    def __init__(self,config,pretrain):
        super().__init__()
        self.vit=MyViT(config)
        self.cls_head=CLSHead(config)
        self.SVit=SViT(config)
        self.reflect=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.config=config
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.pretrain=pretrain
        self.cls_token=nn.Parameter(torch.zeros(1,768))
        nn.init.xavier_uniform_(self.cls_token)
        self.t_embed = nn.Parameter(torch.zeros(1,16,768))
        self.s_embed = nn.Parameter(torch.zeros(1,1,10,768))
        nn.init.xavier_uniform_(self.t_embed)
        nn.init.xavier_uniform_(self.s_embed)

        self.cls_embed=nn.Embedding(38,768,padding_idx=0)
        
    def forward(self,X,bbx_list):

        B,F,A=X.shape
        X=self.cls_embed(X)
        # breakpoint()
        bbx_feature=self.bbx_linear(bbx_list)
        X=self.reflect(X+bbx_feature)
        s_embed=self.s_embed.repeat(B,F,1,1)
        X=X+s_embed
        X=einops.rearrange(X,'b f a d -> (b f) a d')
        frame_token=self.SVit(X)
        frame_token=einops.rearrange(frame_token,'(b f) a d -> b (f a) d',f=F,b=B)
        t_embed=self.t_embed.repeat(B,A,1)
        frame_token=frame_token+t_embed
        ans_token=self.vit(frame_token)
        ans=self.cls_head(ans_token)
        return ans


class CategoryBoxEmbeddings(nn.Module):
    def __init__(self):
        super(CategoryBoxEmbeddings, self).__init__()
        self.category_embeddings = nn.Embedding(embedding_dim=768,num_embeddings=38,padding_idx=0)
        self.box_embedding = nn.Linear(4, 768)
        self.score_embeddings = nn.Linear(1, 768)
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, catego,bbx) -> torch.Tensor:
        category_embeddings = self.category_embeddings(catego)

        boxes_embeddings = self.box_embedding(bbx)
        embeddings = category_embeddings + boxes_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.category_box_embeddings = CategoryBoxEmbeddings()
        self.cls_token=nn.Parameter(torch.zeros(1,1,768))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=768 * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=4
        )

    def forward(self, catego,bbx):

        cb_embeddings = self.category_box_embeddings(catego,bbx)

        num_frames, num_boxes, hidden_size = cb_embeddings.size()[1:]
        # [Batch size x Num. frames, Num. boxes, Hidden size]
        cb_embeddings = cb_embeddings.flatten(0, 1)
        B_tmp=cb_embeddings.size()[0]
        cls_token=self.cls_token.repeat(B_tmp,1,1)
        cb_embeddings=torch.concat([cls_token,cb_embeddings],dim=-2)
        # [Num. boxes, Batch size x Num. frames, Hidden size]
        cb_embeddings = cb_embeddings.transpose(0, 1)
        # [Batch size x Num. frames, Num. boxes]
        layout_embeddings = self.transformer(
            src=cb_embeddings,
        )
        # [Batch size x Num. frames, Num. boxes, Hidden size]
        layout_embeddings = layout_embeddings.transpose(0, 1)
        # [Batch size, Num. frames, Num. boxes, Hidden_size]
        layout_embeddings = layout_embeddings.view(
            -1, num_frames, num_boxes+1, hidden_size
        )
        # [Batch size, Num. frames, Hidden size]
        layout_embeddings = layout_embeddings[:, :, 0, :]

        return layout_embeddings


class STImage(nn.Module):
    def __init__(self,config):
        super(STImage, self).__init__()
        self.visual=ProjectionHead(config)
        self.cls_token=nn.Parameter(torch.zeros(1,1,768))
        self.bbx_linear=nn.Sequential(nn.Linear(4,768),nn.LayerNorm(768),nn.GELU())
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=768 * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=4
        )

    def forward(self, catego,bbx):
        bbx=self.bbx_linear(bbx)
        cb_embeddings = self.visual(catego,bbx)
        num_frames, num_boxes, hidden_size = cb_embeddings.size()[1:]

        cb_embeddings = cb_embeddings.flatten(0, 1)
        B_tmp=cb_embeddings.size()[0]
        cls_token=self.cls_token.repeat(B_tmp,1,1)
        cb_embeddings=torch.concat([cls_token,cb_embeddings],dim=-2)
        cb_embeddings = cb_embeddings.transpose(0, 1)
 
        layout_embeddings = self.transformer(
            src=cb_embeddings,
        )
        # [Batch size x Num. frames, Num. boxes, Hidden size]
        layout_embeddings = layout_embeddings.transpose(0, 1)
        # [Batch size, Num. frames, Num. boxes, Hidden_size]
        layout_embeddings = layout_embeddings.view(
            -1, num_frames, num_boxes+1, hidden_size
        )
        # [Batch size, Num. frames, Hidden size]
        layout_embeddings = layout_embeddings[:, :, 0, :]

        return layout_embeddings


class FramesEmbeddings(nn.Module):
    def __init__(self):
        super(FramesEmbeddings, self).__init__()
        self.layout_embedding = SpatialTransformer()
        self.position_embeddings = nn.Embedding(
            16, 768
        )
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer(
            "position_ids", torch.arange(16).expand((1, -1))
        )
        

    def forward(self,catego,bbx):
        # [Batch size, Num. frames, Hidden size]

        layouts_embeddings = self.layout_embedding(catego,bbx)
        # Frame type and position embeddings
       
        num_frames = 16
        position_embeddings = self.position_embeddings(
            self.position_ids[:, :num_frames]
        )
        # Preparing everything together
        embeddings = layouts_embeddings + position_embeddings 
        embeddings = self.dropout(self.layer_norm(embeddings))

        return embeddings


class FEImage(nn.Module):
    def __init__(self,config):
        super(FEImage, self).__init__()
        self.layout_embedding = STImage(config)
        self.position_embeddings = nn.Embedding(
            16, 768
        )
        self.layer_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer(
            "position_ids", torch.arange(16).expand((1, -1))
        )
        

    def forward(self,catego,bbx):
        # [Batch size, Num. frames, Hidden size]

        layouts_embeddings = self.layout_embedding(catego,bbx)
        # Frame type and position embeddings
       
        num_frames = 16
        position_embeddings = self.position_embeddings(
            self.position_ids[:, :num_frames]
        )
        # Preparing everything together
        embeddings = layouts_embeddings + position_embeddings 
        embeddings = self.dropout(self.layer_norm(embeddings))

        return embeddings

class StltBackbone(nn.Module):
    def __init__(self):
        super(StltBackbone, self).__init__()
        self.frames_embeddings = FramesEmbeddings()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=768 * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        # Temporal Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=8
        )
        self.cls_token=nn.Embedding(1,768)

    def forward(self, catego,bbx):
        # [Batch size, Num. frames, Hidden size]
        frames_embeddings = self.frames_embeddings(catego,bbx)
        # [Num. frames, Batch size, Hidden size]
        # [Num. frames, Num. frames]
        X,Y,Z=frames_embeddings.shape
        cls_token=self.cls_token.weight.unsqueeze(0).repeat(X,1,1)
        # frames_embeddings=torch.cat([cls_token,frames_embeddings],dim=1)
        # [Num. frames, Batch size, Hidden size]
        transformer_output = self.transformer(
            src=frames_embeddings
        )
        return transformer_output

class StltBImage(nn.Module):
    def __init__(self,config):
        super(StltBImage, self).__init__()
        self.frames_embeddings = FEImage(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=768 * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        # Temporal Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=8
        )
        self.cls_token=nn.Embedding(1,768)

    def forward(self, catego,bbx):
        # [Batch size, Num. frames, Hidden size]
        frames_embeddings = self.frames_embeddings(catego,bbx)
        # [Num. frames, Batch size, Hidden size]
        # [Num. frames, Num. frames]
        X,Y,Z=frames_embeddings.shape
        cls_token=self.cls_token.weight.unsqueeze(0).repeat(X,1,1)
        frames_embeddings=torch.cat([cls_token,frames_embeddings],dim=1)
        # [Num. frames, Batch size, Hidden size]
        transformer_output = self.transformer(
            src=frames_embeddings
        )
        return transformer_output

class Stlt(nn.Module):
    def __init__(self,config):
        super(Stlt, self).__init__()
        self.backbone = StltBackbone()
        self.prediction_head = CLSHead(config)
        


    def forward(self,catgo,bbx):
        # [Num. frames, Batch size, Hidden size]
        stlt_output = self.backbone(catgo,bbx)
        # [Batch size, Hidden size]
        stlt_output = stlt_output[:,-1,:]
        logits = self.prediction_head(stlt_output)
        return logits
        
class StltImage(nn.Module):
    def __init__(self,config):
        super(StltImage, self).__init__()
        self.backbone = StltBImage(config)
        self.prediction_head = CLSHead(config)
        


    def forward(self,catgo,bbx):
        # [Num. frames, Batch size, Hidden size]
        stlt_output = self.backbone(catgo,bbx)
        # [Batch size, Hidden size]
        stlt_output = stlt_output[:,0,:]
        logits = self.prediction_head(stlt_output)
        return logits
     

# class FrameTrans(nn.Module):

#     def __init__():
#         super().__init__()

if __name__=='__main__':
    config=load_config()
    device='cuda:0'
    a=torch.randn(5,16,10,768)
    data=process_data(a)
    X,edge=data.x.to(device),data.edge_index.to(device)
    model=MyModel(config)
    model.to(device)
    private,common,atom,ans=model(X,edge)
    entro=torch.log2(torch.std(private[0],-1))
    print(ans.shape,entro.shape)
