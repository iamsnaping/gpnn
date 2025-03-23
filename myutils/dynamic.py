import torch
from torch import nn
import sys
sys.path.append('/home/wu_tian_ci/GAFL')
from myutils.common import *
from myutils.gcn import *
import einops


class GraphEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.frame_level=Flatter(config.dims,config.fl.out1,config.fl.out2,config.dropout,config.eps)
        self.video_level=Flatter(config.dims,config.vl.out1,config.vl.out2,config.dropout,config.eps)
    
    def forward(self,X):
        fl=self.frame_level(X)
        vl=self.video_level(fl)
        return vl

class ScoreNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer1=FFN(config.dims,config.eps,config.dims*4,config.dropout)
        self.layer2=nn.Sequential(nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims),nn.Sigmoid)


class DynamicGraphEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.frame_level=Flatter(config.dims,config.fl.out1,config.fl.out2,config.dropout,config.eps)
        self.video_level=Flatter(config.dims,config.vl.out1,config.vl.out2,config.dropout,config.eps)
    
    def forward(self,X):
        fl=self.frame_level(X)
        vl=self.video_level(fl)
        return vl



def pairwise_distance(x):
    x_inner = -2*torch.matmul(x, x.transpose(-1, -2))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    scores= x_square + x_inner + x_square.transpose(-1, -2)
    return scores



def min_max_norm(x):
    flatten_score_min = x.min(axis=-1, keepdim=True).values
    flatten_score_max = x.max(axis=-1, keepdim=True).values
    norm_flatten_score = (x - flatten_score_min) / (flatten_score_max - flatten_score_min + 1e-5)
    return norm_flatten_score


# X.shape bf n d
def cal_edge(x,threshold):
    scores=pairwise_distance(x)

    scores=min_max_norm(scores)
    batch_num,node_nums,_=scores.shape
    cnt=0
    index_list=[]
    for idx in range(batch_num):
        data=scores[idx]
        adjacency_matrix = (data > threshold).float()
        edge_index, edge_attr = dense_to_sparse(adjacency_matrix)
        edge_index[0]+=cnt*node_nums
        edge_index[1]+=cnt*node_nums
        index_list.append(edge_index)
        cnt+=1
    return torch.cat(index_list,dim=1)



class SingleDGC(nn.Module):

    def __init__(self, config,cal=True):
        super().__init__()
        self.threshold=config.knn.threshold
        self.gcn=GCNChain(config.gnn.indims,config.gnn.outdims,norm=cal)
        self.cal=cal
        self.f=config.frames
        self.a=config.actors
    
    def forward(self,X,edge_index):

        B,Actors,Dims=X.shape
        X=einops.rearrange(X,'b a d -> (b a) d')
        X=self.gcn(X,edge_index)
        X=einops.rearrange(X,'(b a) d -> b a d',b=B,a=Actors,d=Dims)
        if not self.cal:
            return X,edge_index
        edge_index=cal_edge(X,self.threshold)
        return X,edge_index

class DynamicGCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.threshold=config.knn.threshold
        self.layers=config.dgc.layers
        self.gcn=nn.ModuleList()
        for i in range(self.layers-1):
            self.gcn.append(SingleDGC(config,cal=True))
        self.gcn.append(SingleDGC(config,cal=False))
        self.norm=nn.Sequential(tnn.GraphNorm(config.gnn.outdims),nn.GELU())
            
    def forward(self,X,edge_index):
        final_X=X.clone()
        for layer in self.gcn:
            X,edge_index=layer(X,edge_index)
        
        X=self.norm(final_X+X)
        edge_index=cal_edge(X,self.threshold)
        return X,edge_index



