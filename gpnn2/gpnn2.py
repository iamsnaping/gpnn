"""
Created on Oct 07, 2017

@author: Siyuan Qi

Description of the file.

"""

import os

import torch
import torch.nn
import torch.nn as nn
import torch.autograd
from gpnn2.MessageFunction import MessageFunction
from gpnn2.LinkFunction import LinkFunction
from gpnn2.UpdateFunction import UpdateFunction
import einops
from torch_geometric import nn as tnn
from myutils.common import MLPs


class GPNNCell(torch.nn.Module):
    def __init__(self):
        super(GPNNCell, self).__init__()

        self.link_fun = LinkFunction('GraphConvLSTM')
        self.message_fun = MessageFunction('linear_concat')

        # self.update_funs = torch.nn.ModuleList([])
        # self.update_funs.append(UpdateFunction('gru'))
        # self.update_funs.append(UpdateFunction('gru'))
        self.merging=nn.Sequential(nn.Linear(768,768),nn.LayerNorm(768,eps=1e-12),nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=768 * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.tfm = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=2
        )



        # self._load_link_fun()
    # edge features [batch frames nodes edges dims] nodes==edges
    # node features [batch frames nodes dims]
    def forward(self, edge_features, node_features):

        weight_edge=self.link_fun(edge_features)
        m_v = self.message_fun(node_features, node_features, edge_features)
        m_v=self.merging(m_v)
        weight_edge=weight_edge.expand_as(m_v)
        edge_weighted=(weight_edge*m_v)
        edge_weighted=torch.sum(edge_weighted,-2)
        # sum aggregation
        node_features=node_features+edge_weighted

        return node_features




class GPNNCell2(torch.nn.Module):
    def __init__(self):
        super(GPNNCell2, self).__init__()
        self.message_fun = MessageFunction('linear_concat')
        self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*3,768),nn.GELU(),
                                    nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-12),nn.GELU())

        self.link_fun = LinkFunction('GraphConvLSTM')
        self.residual=tnn.MessageNorm(learn_scale=True)
        self.norm=nn.Sequential(nn.Linear(768,768),tnn.GraphNorm(768),nn.GELU())
        self.merging=nn.Sequential(nn.Linear(768,768),nn.LayerNorm(768,eps=1e-12),nn.GELU())

        # self._load_link_fun()
    # edge features [batch frames nodes edges dims] nodes==edges
    # node features [batch frames nodes dims]
    def forward(self, node_features,global_features):
        edge_feature_1=node_features.unsqueeze(-2).repeat(1,1,1,10,1)
        edge_feature_2=edge_feature_1.transpose(-2,-3)
        edge_features=self.edge_fun(torch.cat([global_features,edge_feature_1,edge_feature_2],dim=-1))

        weight_edge=self.link_fun(edge_features)

        m_v = self.message_fun(node_features, node_features, edge_features)
        m_v=self.merging(m_v)
        weight_edge=weight_edge.expand_as(m_v)
        edge_weighted=(weight_edge*m_v)
        edge_weighted=torch.sum(edge_weighted,-2)
        # sum aggregation
        # node_features=node_features+edge_weighted
        node_features=self.norm(self.residual(node_features,edge_weighted)+node_features)

        return node_features,edge_features



# edge
class GPNNCell4(torch.nn.Module):
    def __init__(self,config):
        super(GPNNCell4, self).__init__()
        self.message_fun = MessageFunction('linear_concat')
        self.edge_fun=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*3,config.dims),nn.GELU(),
                                    nn.Dropout(config.dropout),nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())

        self.link_fun = LinkFunction('one_edge')

        self.residual=tnn.MessageNorm(learn_scale=True)
        self.norm=nn.Sequential(nn.Linear(config.dims,config.dims),tnn.GraphNorm(config.dims),nn.GELU())

        self.residual_obj=tnn.MessageNorm(learn_scale=True)
        self.norm_obj=nn.Sequential(nn.Linear(config.dims,config.dims),tnn.GraphNorm(config.dims),nn.GELU())

        self.merging=nn.Sequential(nn.Linear(config.dims,config.dims),nn.LayerNorm(config.dims,eps=config.eps),nn.GELU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=768 * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.tfm = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.gpnn.enc_layer
        )
        self.edges=None
        self.visual=False

        # self._load_link_fun()
    # edge features [batch frames nodes-1 dims] nodes==edges
    # human feature [batch frames 1 dims]
    # obj features [batch frames nodes-1 dims]
    def forward(self, human_feature,obj_features,edge_features):
        B,F,N,D=obj_features.shape
        human_features=human_feature.repeat(1, 1, N, 1)

        # print('obj,human',human_feature.shape,human_features.shape,obj_features.shape)
        # breakpoint()
        tmp_edge=self.edge_fun(torch.cat([torch.cat([human_features,edge_features,obj_features],dim=-1), # human-obj
                                          torch.cat([obj_features,edge_features,human_features],dim=-1)],dim=-2))# obj-human
        # tmp_edge1=self.edge_fun(torch.cat([edge_features,obj_features],dim=-1))
        # tmp_edge2=self.edge_fun(torch.cat([edge_features,human_features],dim=-1))
        # tmp_edge=torch.cat([tmp_edge1,tmp_edge2],dim=-2)
        tmp_edge=self.tfm(einops.rearrange(tmp_edge,'b f n d -> (b n) f d'))
        tmp_edge=einops.rearrange(tmp_edge,'(b n) f d -> b f n d',b=B,n=N*2)

        weight_edge=self.link_fun(tmp_edge)
        if self.visual:
            self.edges=weight_edge.cpu().detach().numpy()
        node_features=torch.cat([human_features,obj_features],dim=-2)

        m_v = self.message_fun(node_features, node_features, tmp_edge)
        m_v=self.merging(m_v)
        weight_edge=weight_edge.expand_as(m_v)
   
        edge_weighted=(weight_edge*m_v)
        edge_weighted_human=edge_weighted[:,:,:N,:]
        edge_weighted_obj=edge_weighted[:,:,N:,:]
        edge_weighted_human=torch.sum(edge_weighted_human,-2,keepdim=True)
        # print('human',human_feature.shape,edge_weighted_human.shape)
        # sum aggregation
        # node_features=node_features+edge_weighted
        human_feature=self.norm(self.residual(human_feature,edge_weighted_human)+human_feature)
        obj_features=self.norm_obj(self.residual_obj(obj_features,edge_weighted_obj)+obj_features)
        # print('obj,human',human_feature.shape,obj_features.shape)
        return human_feature,obj_features


class GPNNCellText(torch.nn.Module):
    def __init__(self):
        super(GPNNCellText, self).__init__()
        self.message_fun = MessageFunction('linear_concat')
        self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*2,768),nn.GELU(),
                                    nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-12),nn.GELU())

        self.link_fun = LinkFunction('GraphConvLSTM')
        self.residual=tnn.MessageNorm(learn_scale=True)
        self.norm=nn.Sequential(nn.Linear(768,768),tnn.GraphNorm(768),nn.GELU())
        self.merging=nn.Sequential(nn.Linear(768,768),nn.LayerNorm(768,eps=1e-12),nn.GELU())

        # self._load_link_fun()
    # edge features [batch frames nodes edges dims] nodes==edges
    # node features [batch frames nodes dims]
    def forward(self, node_features):
        edge_feature_1=node_features.unsqueeze(-2).repeat(1,1,1,10,1)
        edge_feature_2=edge_feature_1.transpose(-2,-3)
        edge_features=self.edge_fun(torch.cat([edge_feature_1,edge_feature_2],dim=-1))

        weight_edge=self.link_fun(edge_features)

        m_v = self.message_fun(node_features, node_features, edge_features)
        m_v=self.merging(m_v)
        weight_edge=weight_edge.expand_as(m_v)
        edge_weighted=(weight_edge*m_v)
        edge_weighted=torch.sum(edge_weighted,-2)
        # sum aggregation
        # node_features=node_features+edge_weighted
        node_features=self.norm(self.residual(node_features,edge_weighted)+node_features)

        return node_features


# no global features no edges
class GPNNCell3(torch.nn.Module):
    def __init__(self):
        super(GPNNCell3, self).__init__()
        self.message_fun = MessageFunction('linear_concat')
        self.edge_fun=nn.Sequential(nn.Dropout(0.1),nn.Linear(768*2,768),nn.GELU(),
                                    nn.Dropout(0.1),nn.Linear(768,768),nn.LayerNorm(768,eps=1e-12),nn.GELU())

        self.link_fun = LinkFunction('GraphConvLSTM')
        self.residual=tnn.MessageNorm(learn_scale=True)
        self.norm=nn.Sequential(nn.Linear(768,768),tnn.GraphNorm(768),nn.GELU())
        self.merging=nn.Sequential(nn.Linear(768,768),nn.LayerNorm(768,eps=1e-12),nn.GELU())

        # self._load_link_fun()
    # edge features [batch frames nodes edges dims] nodes==edges
    # node features [batch frames nodes dims]
    def forward(self, node_features):
        edge_feature_1=node_features.unsqueeze(-2).repeat(1,1,1,10,1)
        edge_feature_2=edge_feature_1.transpose(-2,-3)
        edge_features=self.edge_fun(torch.cat([edge_feature_1,edge_feature_2],dim=-1))

        weight_edge=self.link_fun(edge_features)

        m_v = self.message_fun(node_features, node_features, edge_features)
        m_v=self.merging(m_v)
        weight_edge=weight_edge.expand_as(m_v)
        edge_weighted=(weight_edge*m_v)
        edge_weighted=torch.sum(edge_weighted,-2)
        # sum aggregation
        # node_features=node_features+edge_weighted
        node_features=self.norm(self.residual(node_features,edge_weighted)+node_features)

        return node_features


class GPNN2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer=config.gpnn.layer

        self.gpnn=nn.ModuleList()
        for i in range(self.layer):
            self.gpnn.append(GPNNCell2())

    
    # node features batch frames nodes dims
    def forward(self,node_features,global_feature):
        # origin_nodes=node_features.clone()
        for layer in self.gpnn:
            node_features,_=layer(node_features,global_feature)
            # node_features=node_features+origin_nodes
            
        # node_shape=node_features.shape
        # node_features=einops.rearrange(node_features,'b f n d -> (b n) f d')
        # node_features=self.tfm(node_features)
        # node_features=einops.rearrange(node_features,'(b n) f d -> b f n d',b=node_shape[0],n=node_shape[2]
        return node_features

class GPNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer=config.gpnn.layer

        self.gpnn=nn.ModuleList()
        for i in range(self.layer):
            self.gpnn.append(GPNNCell())
        # self.tfm = nn.TransformerEncoder(
        #     encoder_layer=encoder_layer, num_layers=2
        # )
    
    # node features batch frames nodes dims
    def forward(self,edge_features, node_features):
        origin=node_features.clone()
        for layer in self.gpnn:
            node_features=layer(edge_features,node_features)+origin
        # node_shape=node_features.shape
        # node_features=einops.rearrange(node_features,'b f n d -> (b n) f d')
        # node_features=self.tfm(node_features)
        # node_features=einops.rearrange(node_features,'(b n) f d -> b f n d',b=node_shape[0],n=node_shape[2])
        return node_features

class GPNNText(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer=config.gpnn.layer

        self.gpnn=nn.ModuleList()
        for i in range(self.layer):
            self.gpnn.append(GPNNCellText())
        # self.tfm = nn.TransformerEncoder(
        #     encoder_layer=encoder_layer, num_layers=2
        # )
    
    # node features batch frames nodes dims
    def forward(self,node_features):
        for layer in self.gpnn:
            node_features=layer(node_features)
        # node_shape=node_features.shape
        # node_features=einops.rearrange(node_features,'b f n d -> (b n) f d')
        # node_features=self.tfm(node_features)
        # node_features=einops.rearrange(node_features,'(b n) f d -> b f n d',b=node_shape[0],n=node_shape[2])
        return node_features

class GPNN3(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer=config.gpnn.layer

        self.gpnn=nn.ModuleList()
        for i in range(self.layer):
            self.gpnn.append(GPNNCell3())
        # self.tfm = nn.TransformerEncoder(
        #     encoder_layer=encoder_layer, num_layers=2
        # )
    
    # node features batch frames nodes dims
    def forward(self,node_features):
        for layer in self.gpnn:
            node_features=layer(node_features)
        # node_shape=node_features.shape
        # node_features=einops.rearrange(node_features,'b f n d -> (b n) f d')
        # node_features=self.tfm(node_features)
        # node_features=einops.rearrange(node_features,'(b n) f d -> b f n d',b=node_shape[0],n=node_shape[2])
        return node_features

class GPNN4(nn.Module):

    def __init__(self, config,layer):
        super().__init__()
        self.layer=layer

        self.gpnn=nn.ModuleList()
        for i in range(self.layer):
            self.gpnn.append(GPNNCell4(config))
        # self.pj=MLPs(config.dims,config.dropout,config.eps,config.gpnn.pj.layer)
        self.edges=[]   

    def visual(self):
        for layer in self.gpnn:
            self.edges.append(layer.edges)
        return self.edges


    def set_visual(self):
        for layer in self.gpnn:
            layer.visual=True


    # node features batch frames nodes dims
    def forward(self,node_features,obj_feature,edge_feature,prompt=None,task_id=None):
        for layer in self.gpnn:
            if prompt is not None:
                t_node=torch.cat([node_features,obj_feature],dim=-2)
                t_node=prompt(t_node,task_id)
                node_features=t_node[:,:,0,:].unsqueeze(-2)
                obj_feature=t_node[:,:,1:,:]
            node_features,obj_feature=layer(node_features,obj_feature,edge_feature)

        # t_node=self.pj(torch.cat([node_features,obj_feature],dim=-2))
        # return t_node[:,:,0,:].unsqueeze(-2),t_node[:,:,1:,:]
        return node_features,obj_feature


def main():
    pass


if __name__ == '__main__':
    main()
