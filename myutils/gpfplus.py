import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot

from omegaconf import OmegaConf

def load_config(path='/home/wu_tian_ci/GAFL/configs/model.yaml'):
    config= OmegaConf.load(path)
    return config

# class GPFplusAtt(nn.Module):
#     # in channel dims of tensor p_num
#     def __init__(self,int,config):
#         super(GPFplusAtt, self).__init__()
#         self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
#         self.cls_embed=nn.Embedding(config.cls.ag,config.dims)
#         self.a = nn.Linear(in_channels, p_num)
#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.p_list)
#         self.a.reset_parameters()

#     def add(self, x: Tensor):
#         score = self.a(x)
#         # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
#         weight = F.softmax(score, dim=1)
#         p = weight.mm(self.p_list)

#         return x + p

class SimplePrompt(nn.Module):

# flag =true -> nn.embeding else linear
    def __init__(self, config,flag=False):
        super().__init__()
        self.pnums=config.finetune.p_nums
        self.dims=config.dims
        self.flag=flag
        if flag:
            
            self.tokens=nn.Embedding(config.cls.ag,config.dims*config.finetune.p_nums)
            self.ptokens=nn.Embedding(config.cls.ag,config.dims)
            # self.t_linear=nn.Linear(config.dims,config.dims)
            # self.p_linear=nn.Linear(config.dims,config.dims)
        else:
            self.tokens=nn.Linear(config.cls.ag,config.dims*config.finetune.p_nums)
            self.ptokens=nn.Linear(config.cls.ag,config.dims)


    def forward(self,X,task_id):
        b,f,n,d=X.shape
        # batch rels
        task_id=task_id.unsqueeze(1).unsqueeze(1).repeat(1,f,n,1)

        X=self.ptokens(task_id)+X
        
        return X

class GPFPlus(nn.Module):
# flag =true -> nn.embeding else linear
    def __init__(self, config,flag=False):
        super().__init__()
        self.pnums=config.finetune.p_nums
        self.dims=config.dims
        self.flag=flag
        if flag:
            self.tokens=nn.Embedding(config.cls.ag,config.dims*config.finetune.p_nums)
            self.ptokens=nn.Embedding(config.cls.ag,config.dims)
            # self.t_linear=nn.Linear(config.dims,config.dims)
            # self.p_linear=nn.Linear(config.dims,config.dims)
        else:
            self.tokens=nn.Linear(config.cls.ag,config.dims*config.finetune.p_nums)
            self.ptokens=nn.Linear(config.cls.ag,config.dims)
        # global:1 human:1 obj:9
        self.net=nn.Linear(config.dims,config.finetune.p_nums,nn.Sigmoid())

    def forward(self,X,task_id,detach=False):
        b,f,n,d=X.shape
        # batch rels
        task_id=task_id.unsqueeze(1).unsqueeze(1).repeat(1,f,n,1)
        # breakpoint()
        if self.flag:
            task_token=self.ptokens(task_id)
            # task_token=self.p_linear(torch.sum(task_token,dim=-2))+X
            task_token=torch.sum(task_token,dim=-2)+X
            # breakpoint()
        else:
            task_token=self.ptokens(task_id)+X
        # batch frames nodes 1 p_nums
        # weight=F.softmax(self.net(task_token).unsqueeze(-2),dim=-1)
        weight=self.net(task_token).unsqueeze(-2)
        # breakpoint()
        # batch frames nodes p_nums dims
        if self.flag:
            # prompt=self.t_linear(torch.sum(self.tokens(task_id),dim=-2).reshape(b,f,n,self.pnums,self.dims))
            prompt=torch.sum(self.tokens(task_id),dim=-2).reshape(b,f,n,self.pnums,self.dims)
        else:
            prompt=self.tokens(task_id).reshape(b,f,n,self.pnums,self.dims)
        # breakpoint()
        # prompt=self.tokens(task_id).reshape(b,f,n,self.pnums,self.dims)
        # batch frames nodes 1 dims
        prompt=weight@prompt
        prompt=prompt.squeeze(-2)
        if detach:
            X=X+prompt.detach()
        else:
            X=X+prompt
        return X


if __name__=='__main__':
    a=torch.randn(2,10,11,768)
    b=torch.randint(0,157,(2,157))
    config=load_config()
    gpf=GPFPlus(config,True)
    ans=gpf(a,b)
    print(ans.shape)

    

