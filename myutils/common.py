import torch
from torch import nn



class FFN(nn.Module):
    def __init__(self,in_dim,eps=1e-12,hidden_dim=None,dropout=0.3,norm_=True):
        super().__init__()
        if hidden_dim is None:
            out_dim=in_dim*4
        else:
            out_dim=hidden_dim
        self.out_dim=out_dim
        self.in_dim=in_dim
        self.norm_=norm_
        self.layer=nn.Sequential(nn.Linear(in_dim,out_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(out_dim,in_dim),nn.Dropout(dropout))
        self.norm=nn.Sequential(nn.LayerNorm(in_dim,eps=eps),nn.GELU())
    
    def forward(self,X):
        if self.norm_:
            return self.norm(X+self.layer(X))
        else:
            return X+self.layer(X)
    


class ProjectionHeadFT(nn.Module):
    def __init__(self,in_dim,eps=1e-12,hidden_dim=None,dropout=0.3,norm_=True):
        super().__init__()
        if hidden_dim is None:
            out_dim=in_dim*4
        else:
            out_dim=hidden_dim
        self.out_dim=out_dim
        self.in_dim=in_dim
        self.norm_=norm_
        self.layer=nn.Sequential(nn.Linear(in_dim,out_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(out_dim,in_dim))
        self.norm=nn.Sequential(nn.LayerNorm(in_dim,eps=eps),nn.GELU())
    
    def forward(self,X):
        return self.norm(X+self.layer(X))

class Flatter(nn.Module):
    def __init__(self,in_dim,out_dim1,out_dim2,
                 dropout=0.3,eps=1e-12):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim1=out_dim1
        self.out_dim2=out_dim2
        # self.scores=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.LayerNorm(in_dim),
        #                           nn.GELU(),nn.Linear(in_dim,1),nn.Sigmoid())
        self.layer1=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,out_dim1),nn.GELU())
        self.flatten=nn.Flatten(start_dim=-2)
        self.layer2=nn.Sequential(nn.Dropout(dropout),nn.Linear(out_dim2,in_dim),nn.LayerNorm(in_dim,eps=eps),nn.GELU())

    def forward(self,X):
        # scores=self.scores()
        f=self.layer1(X)
        f=self.flatten(f)
        return self.layer2(f)




class DynamicFlatter(nn.Module):
    def __init__(self,in_dim,
                 dropout=0.1,eps=1e-5):
        super().__init__()
        self.dims=in_dim
        # score net

        # frame-level
        self.pj1=FFN(in_dim,eps,in_dim*4,dropout)
        # video-level
        self.pj2=FFN(in_dim,eps,in_dim*4,dropout)
        # self.pj2_2=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.LayerNorm(in_dim),nn.GELU())

        # # frame-level
        self.score1=nn.Sequential(nn.Linear(in_dim,in_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim//2,in_dim//4),
                                  nn.GELU(),nn.Linear(in_dim//4,1),nn.Sigmoid())
        # video-level
        self.score2=nn.Sequential(nn.Linear(in_dim,in_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim//2,in_dim//4),
                                  nn.GELU(),nn.Linear(in_dim//4,1),nn.Sigmoid())
        

        # fusion
        self.f1=FFN(in_dim,eps,in_dim*4,dropout)
        # self.f1=MLP(in_dim,in_dim,dropout,eps)

        self.f2=FFN(in_dim,eps,in_dim*4,dropout)
        self.vl_weight=[]
        self.fl_weight=[]
        self.visual=False
    

    def set_visual(self,flag):
        self.visual=flag
    

    def get_visual(self):
        return self.vl_weight,self.fl_weight
    
    def get_last_layer(self):
        return self.f2.weight
        
    
    # batch frame nodes dims
    def forward(self,X):
        b,f,n,d=X.shape
        frame_level=self.pj1(X)



        frame_scores=self.score1(frame_level)
        X=X*frame_scores
        X=torch.sum(X,dim=-2)
        # batch frame dims
        f1=self.f1(X)

        video_level=self.pj2(f1)
        video_scores=self.score2(video_level)
        f1=f1*video_scores
        f1=torch.sum(f1,dim=-2)
        f2=self.f2(f1)
        if self.visual:
            self.vl_weight.append(video_scores.cpu().detach())
            self.fl_weight.append(frame_scores.cpu().detach())
        return f2


class DynamicFlatterWithGate(nn.Module):
    def __init__(self,config,in_dim,
                 dropout=0.3,eps=1e-12):
        super().__init__()
        self.dims=in_dim
        # score net

        # frame-level
        self.pj1_1=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.GELU())
        self.pj1_2=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.LayerNorm(in_dim),nn.GELU())
        # video-level
        self.pj2_1=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.GELU())
        self.pj2_2=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.LayerNorm(in_dim),nn.GELU())

        # frame-level
        self.score1=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim//2,in_dim//4),nn.LayerNorm(in_dim//4),
                                  nn.GELU(),nn.Linear(in_dim//4,1),nn.Sigmoid())
        # video-level
        self.score2=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim//2,in_dim//4),nn.LayerNorm(in_dim//4),
                                  nn.GELU(),nn.Linear(in_dim//4,1),nn.Sigmoid())
        
        self.gate=GateFusion(config)

        # fusion
        self.f1=FFN(in_dim,eps,in_dim*4,dropout)
        # self.f1=MLP(in_dim,in_dim,dropout,eps)

        self.f2=FFN(in_dim,eps,in_dim*4,dropout)

    # batch frame nodes dims
    def forward(self,X,global_f):
        b,f,n,d=X.shape
        frame_level=self.pj1_2(self.pj1_1(X)+X)
        f_global_x=torch.mean(frame_level[:,:,:,:(self.dims//2)],dim=-2,keepdim=True)
        f_local_x=frame_level[:,:,:,(self.dims//2):]
        # print('global',f_global_x.shape,f_local_x.shape)
        frame_level=torch.cat([f_global_x.repeat(1,1,n,1),f_local_x],dim=-1)
        frame_scores=self.score1(frame_level)
        X=X*frame_scores
        X=torch.sum(X,dim=-2)
        # batch frame dims
        f1=self.f1(X)
        # breakpoint()
        f1=self.gate(f1,global_f)

        video_level=self.pj2_2(self.pj2_1(f1)+f1)
        v_global_x=torch.mean(video_level[:,:,:(self.dims//2)],dim=-2,keepdim=True)
        v_local_x=video_level[:,:,(self.dims//2):]
        video_level=torch.cat([v_global_x.repeat(1,f,1),v_local_x],dim=-1)
        video_scores=self.score2(video_level)
        f1=f1*video_scores
        f1=torch.sum(f1,dim=-2)
        f2=self.f2(f1)
        return f2



class GateFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.score1=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*2,1),nn.Sigmoid())
        self.score2=nn.Sequential(nn.Dropout(config.dropout),nn.Linear(config.dims*2,1),nn.Sigmoid())
    
    def forward(self,X1,X2):
        X=torch.cat([X1,X2],dim=-1)
        score1=self.score1(X)
        score2=self.score2(X)
        return X1*score1+score2*X2


class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,dropout,eps):
        super().__init__()
        self.lin=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,out_dim),nn.LayerNorm(out_dim,eps=eps),nn.GELU())
    def forward(self,X):
        return self.lin(X)
    
    def get_last_layer(self):
        return self.lin[1].weight


class Linear(nn.Module):
    def __init__(self,in_dim,out_dim,dropout,eps,norm=False):
        super().__init__()
        self.norm=norm
        if norm: 
            self.lin=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,out_dim),nn.LayerNorm(out_dim,eps=eps),nn.GELU())
        else:
            self.lin=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,out_dim),nn.GELU())
    def forward(self,X):
        return self.lin(X)

class MLPs(nn.Module):
    def __init__(self,in_dim,dropout,eps,layer=3):
        super().__init__()
        self.lin=nn.ModuleList()
        for i in range(layer-1):
            self.lin.append(Linear(in_dim,in_dim,dropout,eps))
        self.lin.append(Linear(in_dim,in_dim,dropout,eps,True))
    def forward(self,X):
        for layer in self.lin:
            X=layer(X)
        return X


class MLPCLS(nn.Module):
    def __init__(self,in_dim,out_dim,dropout,eps):
        super().__init__()
        self.layer=FFN(in_dim,eps,in_dim*4,dropout)
        self.norm=nn.Sequential(nn.LayerNorm(in_dim,eps=eps),nn.GELU())
        self.cls=nn.Linear(in_dim,out_dim)
    
    def forward(self,X):

        return self.cls(self.norm(X+self.layer(X)))
class TwoLayer(nn.Module):
    def __init__(self, in_dim,dropout,eps):
        super().__init__()
        self.layer=nn.Sequential(nn.Dropout(dropout),nn.Linear(in_dim,in_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(in_dim,in_dim))
        self.norm=nn.Sequential(nn.LayerNorm(in_dim,eps=eps),nn.GELU())
    def forward(self,X):

        return self.norm(X+self.layer(X))
