"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""

import torch


class UpdateFunction(torch.nn.Module):
    def __init__(self, update_def):
        super(UpdateFunction, self).__init__()
        self.u_definition = ''
        self.u_function = None
        self.learn_args = torch.nn.ParameterList([])
        self.learn_modules = torch.nn.ModuleList([])
        self.__set_update(update_def)

    def forward(self, h_v, m_v):
        return self.u_function(h_v, m_v)

    # Set an update function
    def __set_update(self, update_def):
        self.u_definition = update_def.lower()

        self.u_function = {
            'gru':     self.u_gru,
        }.get(self.u_definition, None)

        if self.u_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

        init_parameters = {
            'gru':     self.init_gru,
        }.get(self.u_definition, lambda x: (torch.nn.ParameterList([]), torch.nn.ModuleList([]), {}))

        init_parameters()

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    # Definition of update functions
    # GRU: node state as hidden state, message as input
    def u_gru(self, h_v, m_v):
        output, h = self.learn_modules[0](m_v, h_v)
        return h

    def init_gru(self):
        node_feature_size = 768
        message_size = 768
        num_layers = 2
        bias = False
        dropout = False
        self.learn_modules.append(torch.nn.GRU(message_size, node_feature_size, num_layers=num_layers,batch_first=True))


def main():
    pass


if __name__ == '__main__':
    main()
