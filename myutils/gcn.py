from torch_geometric import nn as tnn
from torch_geometric.utils import dense_to_sparse

from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.data import Data,Batch
import torch
from torch import nn
import torch.nn.functional as F

# class GATFFN(nn.Module):

#     def __init__(self, input_dim,out_dim,norm=True):
#         super().__init__()
#         self.layer1=GATConv(in_channels=input_dim,out_channels=out_dim*4)
#         self.layer2=GATConv(in_channels=out_dim*4,out_channels=input_dim)
#         self.skip=tnn.MessageNorm(learn_scale=True)
#         if self.norm_:
#             self.norm=nn.Sequential(tnn.GraphNorm(input_dim),nn.GELU())
    
#     def forward(self,X,edge_index)




class GCNChain(nn.Module):
    def __init__(self,input_dim,hidden_dim,norm=True):
        super(GCNChain,self).__init__()
        self.norm_=norm

        self.layer=GCNConv(input_dim,hidden_dim)
        if self.norm_:
            self.norm=nn.Sequential(tnn.GraphNorm(hidden_dim),nn.GELU())
        self.skip=tnn.MessageNorm(learn_scale=True)
    def forward(self,X,edge_index):

        if self.norm_:
            return self.norm(self.skip(X,self.layer(X,edge_index))+X)
        else:
            return self.skip(X,self.layer(X,edge_index))+X

class GCNFFN(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(GCNFFN,self).__init__()
        self.ffn1=GCNChain(input_dim,hidden_dim)
        self.ffn2=GCNChain(hidden_dim,input_dim,norm=False)
        self.norm=tnn.MessageNorm(learn_scale=True)
        self.act=nn.Sequential(tnn.GraphNorm(input_dim),nn.GELU())
    def forward(self,X,edge_index):


        f=self.ffn1(X,edge_index)

        f=self.ffn2(f,edge_index)

        return self.act(self.norm(X,f)+X)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,layers,threshold=0.7):
        super(GCN, self).__init__()
        self.gcn=nn.ModuleList()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.threshold=threshold
        for i in range(layers):
            self.gcn.append(GCNFFN(input_dim,hidden_dim))


    @torch.no_grad()
    def pre_process(self, batch_data):
        processed_graphs = []
        batch_num=batch_data.shape[0]
        for idx in range(batch_num):
            data=batch_data[idx]
            cosin = torch.matmul(data, data.transpose(-1, -2))
            cosin = F.softmax(cosin, dim=-1)
            adjacency_matrix = (cosin > self.threshold).float()
            edge_index, edge_attr = dense_to_sparse(adjacency_matrix)

            # 构建 PyTorch Geometric 的数据对象
            graph = Data(x=data, edge_index=edge_index, edge_attr=edge_attr)
            processed_graphs.append(graph)

        # 将单个图的列表合并为一个批处理对象
        batch = Batch.from_data_list(processed_graphs)
        return batch

    def forward(self, data):
        data=self.pre_process(data)
        x,edge_index=data.x,data.edge_index
        for layer in self.gcn:
            x=layer(x,edge_index)
        return x

class TinyGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,layers=2):
        super().__init__()
        self.gcn=nn.ModuleList()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim

        for i in range(layers):
            self.gcn.append(GCNFFN(input_dim,hidden_dim))

    def forward(self, x,edge_index):
        for layer in self.gcn:
            x=layer(x,edge_index)
        return x,edge_index


def process_data(batch_data):
    threshold=0.5
    processed_graphs = []
    batch_num=batch_data.shape[0]
    for idx in range(batch_num):
        data=batch_data[idx]
        cosin = torch.matmul(data, data.transpose(-1, -2))
        cosin = F.softmax(cosin, dim=-1)
        adjacency_matrix = (cosin > threshold).float()
        edge_index, edge_attr = dense_to_sparse(adjacency_matrix)

        # 构建 PyTorch Geometric 的数据对象
        graph = Data(x=data, edge_index=edge_index, edge_attr=edge_attr)
        processed_graphs.append(graph)

    # 将单个图的列表合并为一个批处理对象
    batch = Batch.from_data_list(processed_graphs)
    return batch

