


from typing import Optional
import torch.distributed as dist
import torch
from torch import Tensor

from torch_geometric.nn.inits import ones, zeros
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_mean



# global frame node norm
class GlobalNorm(torch.nn.Module):

    def __init__(self, dims: int, dim2:int,worldsize:int,momentum: int =0.1, eps: float = 1e-5):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.Tensor(1,16,dim2,1))
        self.bias = torch.nn.Parameter(torch.Tensor(1,16,dim2,1))
        self.momentum=momentum
        self.worldsize=worldsize
        self.mean_scale = torch.nn.Parameter(torch.Tensor(1,16,dim2,1))
        self.register_buffer('global_mean', torch.zeros(1,16,dim2,1))
        self.register_buffer('global_var', torch.ones(1,16,dim2,1))
        self.reset_parameters()
        

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)


    def forward(self, x):
        B, F, N, D = x.shape  
        global_mean=x.mean(dim=[0,3], keepdim=True)
        global_var=x.var(dim=[0,3],unbiased=False,keepdim=True)
        if self.training:
            with torch.no_grad():

                if dist.is_initialized():
                    dist.all_reduce(global_mean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(global_var, op=dist.ReduceOp.SUM)
                    global_mean = global_mean / self.worldsize  
                    global_var = global_var / self.worldsize
                self.global_mean.mul_(1-self.momentum).add_(self.momentum*global_mean)
                self.global_var.mul_(1-self.momentum).add_(self.momentum*global_var)
        else:
            global_mean=self.global_mean
            global_var=self.global_var
       
        out=x-global_mean
        std=(global_var+self.eps).sqrt()
        return out * self.weight/std +self.bias




    def __repr__(self):
        return f'{self.__class__.__name__}({self.dims})'

# graph norm
# pass
class GlobalNorm2(torch.nn.Module):

    def __init__(self, dims: int, dim2:int,worldsize:int,momentum: int =0.1, eps: float = 1e-5):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.Tensor(dims))
        self.bias = torch.nn.Parameter(torch.Tensor(dims))
        self.momentum=momentum
        self.worldsize=worldsize
        self.mean_scale = torch.nn.Parameter(torch.Tensor(dims))
        self.register_buffer('global_mean', torch.zeros(dims))
        self.register_buffer('global_var', torch.ones(dims))
        self.reset_parameters()
        
    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)


    def forward(self, x):
        B, F, N, D = x.shape  
        global_mean=x.mean(dim=[1,2], keepdim=True)
        # global_var=x.var(dim=[1,2],unbiased=False,keepdim=True)
        # if self.training:
        #     with torch.no_grad():
        #         if dist.is_initialized():
        #             dist.all_reduce(global_mean, op=dist.ReduceOp.SUM)
        #             dist.all_reduce(global_var, op=dist.ReduceOp.SUM)
        #             global_mean = global_mean / self.worldsize  
        #             global_var = global_var / self.worldsize

        #         self.global_mean.mul_(1-self.momentum).add_(self.momentum*global_mean)
        #         self.global_var.mul_(1-self.momentum).add_(self.momentum*global_var)
        # else:
        #     global_mean=self.global_mean
        #     global_var=self.global_var

        out=x-global_mean*self.mean_scale
        var=torch.sum(out.pow(2),dim=[1,2])/(F+N-1)
        std=(var+self.eps).sqrt()
        return out * self.weight/std +self.bias


    def __repr__(self):
        return f'{self.__class__.__name__}({self.dims})'

# global batch norm/global mean real-time var
class GlobalNorm3(torch.nn.Module):

    def __init__(self, dims: int, dim2:int,worldsize:int,momentum: int =0.1, eps: float = 1e-5):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.Tensor(1,16,1,1))
        self.bias = torch.nn.Parameter(torch.Tensor(1,16,1,1))
        self.momentum=momentum
        self.worldsize=worldsize
        self.mean_scale = torch.nn.Parameter(torch.Tensor(1,16,1,1))
        self.register_buffer('global_mean', torch.zeros(1, 16, 1, 1))
        # self.register_buffer('global_var', torch.ones(1, 16, dim2, dims))
        self.reset_parameters()
        
    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)


    def forward(self, x):
        B, F, N, D = x.shape  
        global_mean=x.mean(dim=[0,2,3], keepdim=True)
        # global_var=x.var(dim=0,unbiased=False,keepdim=True)
        if self.training:
            with torch.no_grad():
                if dist.is_initialized():
                    dist.all_reduce(global_mean, op=dist.ReduceOp.SUM)
                    global_mean = global_mean / self.worldsize  

                self.global_mean.mul_(1-self.momentum).add_(self.momentum*global_mean)
        else:
            global_mean=self.global_mean

        out=x-global_mean*self.mean_scale
        var=torch.sum(out.pow(2),dim=[2,3],keepdim=True)/(N+D-1)
        std=(var+self.eps).sqrt()
        return out * self.weight/std +self.bias

    def __repr__(self):
        return f'{self.__class__.__name__}({self.dims})'

# global batch frame norm/global mean real-time var
class GlobalNorm4(torch.nn.Module):

    def __init__(self, dims: int, dim2:int,worldsize:int,momentum: int =0.1, eps: float = 1e-5):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.Tensor(1,16,dim2,1))
        self.bias = torch.nn.Parameter(torch.Tensor(1,16,dim2,1))
        self.momentum=momentum
        self.worldsize=worldsize
        self.mean_scale = torch.nn.Parameter(torch.Tensor(1,16,dim2,1))
        self.register_buffer('global_mean', torch.zeros(1, 16, dim2, 1))

        self.reset_parameters()
        
    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)


    def forward(self, x):
        B, F, N, D = x.shape  
        global_mean=x.mean(dim=[0,-1], keepdim=True)
        if self.training:
            with torch.no_grad():
                if dist.is_initialized():
                    dist.all_reduce(global_mean, op=dist.ReduceOp.SUM)
                    global_mean = global_mean / self.worldsize  
                self.global_mean.mul_(1-self.momentum).add_(self.momentum*global_mean)
        else:
            global_mean=self.global_mean


        out=x-global_mean*self.mean_scale
        var=torch.sum(out.pow(2),dim=[-1],keepdim=True)/(D-1)
        std=(var+self.eps).sqrt()
        return out * self.weight/std +self.bias
    


    def __repr__(self):
        return f'{self.__class__.__name__}({self.dims})'


if __name__=='__main__':
    x=torch.randn([5,2,2,2])
    batch=torch.tensor([0,0,0,0,0])
    batchsize=1
    data=scatter_mean(x, batch, dim=0, dim_size=batchsize)[batch]
    breakpoint()