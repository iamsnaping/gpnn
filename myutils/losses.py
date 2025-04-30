from torch import nn
import torch
import einops
import sys
sys.path.append('/home/wu_tian_ci/GAFL')
from myutils.config import *
import torch.nn.functional as F
class Loss(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError





# b*f n d
class ContraLoss(Loss):
    def __init__(self, config):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss()
        self.temp=config.temp
    def __call__(self, x):
        batch_size = x.size(0)
        logits = torch.mm(x, x.transpose(1, 0))/self.temp
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=x.device)
        return self.loss(logits, targets)

class SingleCLS(Loss):
    def __init__(self,config):
        self.config=config
        self.loss=nn.CrossEntropyLoss()
    def __call__(self, ans,label):
        return self.loss(F.softmax(ans,dim=-1),label)
        
class Entropy(Loss):
    def __init__(self,config):
        self.config=config
        self.loss=nn.MSELoss()
    
    def __call__(self, common,private):
        common_entropy=torch.log2(torch.std(common,-1))
        private_entropy=torch.log2(torch.std(private,-1))
        return self.loss(common_entropy,private_entropy)
        
class ReconstructLoss(Loss):

    def __init__(self,config):
        super().__init__()
        self.config=config
        # self.loss=nn.KLDivLoss(reduction='sum')
        self.loss=nn.MSELoss(reduction='none')
        self.margin=config.loss.rec.margin
        self.max_epoch=config.max_epoch
        self.max_margin=config.loss.rec.max_mar
        self.mw=config.loss.rec.mw
        self.step=0
    
    def reset(self):
        self.step=0

    def __call__(self,target,reconstruct,epoch):
        # probs1=F.log_softmax(reconstruct,dim=-1)
        # probs2=F.softmax(target,dim=-1)
        # loss=F.relu(self.loss(probs1,probs2)-self.margin)/target.size(0)
        margin=self.max_margin-epoch/self.max_epoch*(self.max_margin-self.margin)*self.mw
        # margin=0
        # loss=torch.mean(F.relu(torch.mean(self.loss(target,reconstruct),dim=(1,2,3))-margin))
        loss=F.relu(torch.mean(self.loss(target,reconstruct))-margin)
        return loss*self.config.loss.rec.weight,round(loss.item(),2)

# batch frame nodes dims
class SeperationLoss(Loss):
    def __init__(self,config):
        self.config=config
        self.margin=config.loss.spe.margin
        self.max_margin=config.loss.spe.max_mar
        self.max_epoch=config.max_epoch
        self.mw=config.loss.spe.mw
        self.step=0
    
    def reset(self):
        self.step=0
    
    def __call__(self,x1,x2,epoch):
        # Subtract the mean 
        x1_mean = torch.mean(x1, (1,2,3), True)
        x1 = x1 - x1_mean
        x2_mean = torch.mean(x2, (1,2,3), True)
        x2 = x2 - x2_mean

        # print('mean',x1_mean,x2_mean)
        # Compute the cross correlation
        margin=self.max_margin-epoch/self.max_epoch*(self.max_margin-self.margin)*self.mw
        # margin=self.margin
        sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
        sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
        # margin=0
        # corr = torch.mean(F.relu(torch.abs(torch.mean(x1*x2,dim=(1,2,3)))/(sigma1*sigma2)-margin))
        # corr = F.relu(torch.mean(torch.abs(torch.mean(x1*x2,dim=(1,2,3)))/(sigma1*sigma2))-margin)
        corr=F.relu(torch.abs(torch.mean(x1*x2))/(sigma1*sigma2)-margin)
        # corr=torch.mean(F.relu(torch.abs(torch.mean(x1*x2,dim=(1,2,3)))/(sigma1*sigma2)))
        # breakpoint()
        return corr*self.config.loss.spe.weight,round(corr.item(),2)


class KLSeperation(Loss):
    def __init__(self,config):
        super().__init__()
        self.config=config
        # self.loss=nn.KLDivLoss(reduction='none')
        self.loss=nn.CosineSimilarity()
        self.margin=config.loss.cos.margin
    
    # X1、X2: batch cls
    # mask: batch cls
    def __call__(self, X1,X2,mask):
        # probs1=F.relu(F.sigmoid(X1)-self.margin)
        # probs2=F.relu(F.sigmoid(X2)-self.margin)
        probs1=mask*F.sigmoid(X1)
        probs2=mask*F.sigmoid(X2)
        cos=(1-self.loss(probs1,probs2)).mean()
        # breakpoint()
        return cos*self.config.loss.spe,round(cos.item(),3) 


class Criterion(nn.Module):
    def __init__(self,config):
        super(Criterion, self).__init__()
        self.loss_function = nn.BCEWithLogitsLoss()
        self.weight=config.loss.cls

    def forward(self, logits, labels,round_=2):
        if isinstance(logits,list):
            loss=sum([self.loss_function(l,labels) for l in logits])/len(logits)
        else:
            loss=self.loss_function(logits, labels)
        return loss*self.weight,round(loss.item(),round_)


class AdapterLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.rel=config.loss.adt.rel
        self.cls=config.loss.adt.cls
        self.cls_max=config.loss.adt.cls_max
        self.epoch_max=20
        self.cls_loss=nn.CrossEntropyLoss()
        self.rel_loss=nn.BCEWithLogitsLoss()
    

    # batch frames nums dims
    # mask b f n
    # 
    def forward(self,rel,cls,rel_l,cls_l,epochs):
        cls_loss=self.cls_loss(cls,cls_l)
        rel_loss=self.rel_loss(rel,rel_l)
        # cls_weight=self.cls+(self.cls_max-self.cls)*((epochs+self.epoch_max)/self.epoch_max)
        cls_weight=self.cls
        return rel_loss*self.rel+cls_loss*cls_weight,round(rel_loss.item(),2),round(cls_loss.item(),2)


# class AdapterLoss2(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.rel=config.loss.adt.rel
#         self.cls=config.loss.adt.cls
#         self.cls_max=.5
#         self.epoch_max=20
#         self.cls_loss=nn.CrossEntropyLoss()
#         self.rel_loss=nn.BCEWithLogitsLoss()
    

#     # batch frames nums dims
#     # mask b f n
#     # 
#     def forward(self,cls,rel_l,cls_l,epochs):
#         cls_loss=self.cls_loss(cls,cls_l)
#         cls_weight=self.cls+(self.cls_max-self.cls)*((epochs+self.epoch_max)/self.epoch_max)
#         return cls_loss*cls_weight,round(cls_loss.item(),2)


if __name__=='__main__':
    x=torch.randn(2,2,2,2)
    y=torch.randn(2,2,2,2)
    print(x,y)
    # x_mean=torch.mean(x.reshape(-1,),(1,2,3))
    print(torch.mean(x.reshape(2,-1),dim=-1),torch.mean(y.reshape(2,-1),dim=-1))
    speration=SeperationLoss(load_config())
    print(speration(x,y))