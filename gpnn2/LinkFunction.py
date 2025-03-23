"""
Created on Oct 03, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch
import torch.nn
import torch.nn as nn
import einops


class LinkFunction(torch.nn.Module):
    def __init__(self, link_def):
        super(LinkFunction, self).__init__()
        self.l_definition = ''
        self.l_function = None
        self.learn_args = torch.nn.ParameterList([])
        self.learn_modules = torch.nn.ModuleList([])
        self.__set_link(link_def)

    def forward(self, edge_features):
        return self.l_function(edge_features)

    def __set_link(self, link_def):
        self.l_definition = link_def.lower()

        self.l_function = {
            'graphconv':        self.lstm_,
            'graphconvlstm':    self.lstm_,
            'one_edge': self.one_edge
        }.get(self.l_definition, None)

        if self.l_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + link_def)
            quit()

        init_parameters = {
            'graphconv':        self.init_lstm,
            'graphconvlstm':    self.init_lstm,
            'one_edge': self.init_edge
        }.get(self.l_definition, lambda x: (torch.nn.ParameterList([]), torch.nn.ModuleList([]), {}))

        init_parameters()

    def get_definition(self):
        return self.l_definition


    # Definition of linking functions
    # GraphConv



    # GraphConvLSTM
    # edge features [batch frames nodes edges dims] nodes==edges
    def lstm_(self,edge_features):
        edge_shape=edge_features.shape
        edge_feature_=einops.rearrange(edge_features,'b f n e d -> (b n e) f d')
        last_layer_output=self.tfm(edge_feature_)

        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)

        last_ouput=einops.rearrange(last_layer_output,'(b n e) f d -> b f n e d',b=edge_shape[0],f=edge_shape[1],n=edge_shape[2],e=edge_shape[3],d=1)
        return last_ouput
    # human-obj
    # edge features batch frames edges dims edge=node-1
    def one_edge(self,edge_features):

        for layer in self.learn_modules:
            edge_features = layer(edge_features)

        
        return edge_features
    
    def init_lstm(self):
        input_size = 768
        hidden_size = 768
        hidden_layers = 2
        # self.lstm = nn.LSTM(input_size, hidden_size, hidden_layers,bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=12,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.tfm = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=2
        )

        self.learn_modules.append(nn.Linear(hidden_size, 1))
        self.learn_modules.append(nn.Sigmoid())

    def init_edge(self):
        input_size = 768
        hidden_size = 768
        hidden_layers = 2
        # self.lstm = nn.LSTM(input_size, hidden_size, hidden_layers,bidirectional=True)

        self.learn_modules.append(nn.Linear(hidden_size, 1))
        self.learn_modules.append(nn.Sigmoid())

def main():
    pass


if __name__ == '__main__':
    main()
