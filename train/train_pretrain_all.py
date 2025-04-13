import sys
sys.path.append('/home/wu_tian_ci/GAFL')
import torch.optim as optim
import torch
from myutils.extra_model import (MyModelGCN,
                                 Stlt,
                                 StltImage,
                                 MymodelGPNN3,
                                 MymodelGPNN4,
                                 MymodelGPNNText,
                                 MymodelGPNN6,
                                 GPNNMix,
                                 GPNNMix2,
                                 GPNNMix3,
                                 GPNNMix4,
                                 PureMix)
from myutils.mydataset import (CLIPFeatureDataset,
                               CLIPFeatureDatasetAll,
                               StltDataset,
                               DataConfig,
                               CLIPFeatureDatasetCLSREL,
                               CLIPFeatureDatasetCLSRELNonMask,
                               CLIPFeatureDatasetCLSRELOracle,
                               MixAns,
                               MixAns2)
from torch.utils.data import DataLoader
import torch.nn.functional as F
from myutils.config import *
from tqdm import tqdm
import argparse
import random
import logging
from torch import nn as nn
import os
import numpy as np
from myutils.data_utils import (
    add_weight_decay,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    save_checkpoint,
    getTimeStamp,
    MyEvaluatorActionGenome,
)
from myutils.losses import (Criterion,
                            AdapterLoss,
                            SeperationLoss,
                            ReconstructLoss,
                            KLSeperation)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from transformers.trainer_pt_utils import SequentialDistributedSampler,distributed_concat


import warnings

# 将所有警告转换为异常
# warnings.filterwarnings('error')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
# os.environ['OMP_NUM_THREADS'] = '2'

import torch.distributed as dist

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 

def weight_init(m):
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407) 
    if isinstance(m, nn.Linear) and m.weight.requires_grad==True:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm) and m.weight.requires_grad==True:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d) and m.weight.requires_grad==True:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d) and m.weight.requires_grad==True:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m,nn.Embedding) and m.weight.requires_grad==True:
        m.weight.data.normal_(0.0)
    elif isinstance(m,nn.Parameter) and m.weight.requires_grad==True:
        # nn.init.normal_(m.weight,0.0)
        m.weight.data.normal_(0.0)

def weight_init_dis(m):
    rank=torch.torch.distributed.get_rank()
    random.seed(13407+rank)
    np.random.seed(13407+rank)
    torch.manual_seed(13407+rank)
    if isinstance(m, nn.Linear) and m.weight.requires_grad==True:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm) and m.weight.requires_grad==True:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d) and m.weight.requires_grad==True:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d) and m.weight.requires_grad==True:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m,nn.Embedding) and m.weight.requires_grad==True:
        nn.init.normal_(m.weight)


def reduce_mean(name,tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone().to(tensor)
    # print(name,'tensor',tensor.device)
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def train(args,pretrain):
    config=load_config()
    test_dataset=CLIPFeatureDatasetAll('test',sample_each_clip=16,train=False)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=16)


    dataset=CLIPFeatureDatasetAll('train')
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    model=MyModelGCN(config,pretrain=pretrain).to(args.device)
    model.apply(weight_init)
    cri=Criterion()
    max_acc=999
    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    time_stamp=getTimeStamp()
    loss_record_path='/home/wu_tian_ci/GAFL/recoder/loss_value'
    t_path=time_stamp[0:8]
    loss_record_path=os.path.join(loss_record_path,t_path)
    if not os.path.exists(loss_record_path):
        os.mkdir(loss_record_path)
    loss_record_path=os.path.join(loss_record_path,time_stamp[8:12]+'loss.txt')
    if not os.path.exists(loss_record_path):
        f=open(loss_record_path,'w')
        f.write('begin to write\n')
        f.close()

    for epoch in range(args.epoch):
        model.train()
        txt_k=1
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                frames,bbx,mask,label=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                label=label.to(device).squeeze()
                ans=model(frames,mask,bbx)
                loss=cri(ans,label)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
                f=open(loss_record_path,'a')
                f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss:'+str(round(loss.item(),5))+ '\n')
                f.close()
                txt_k+=1
        evaluator.reset()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                frames,bbx,mask,label=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                label=label.to(device).squeeze()
                ans=model(frames,mask,bbx)
                evaluator.process(ans,label)
            metrics = evaluator.evaluate()

        if pretrain:
            save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'pretrain')
            print('saved')
        else:
            save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')


def train_rel_cls(args,pretrain):                                                   
    config=load_config()
    config2=load_config('/home/wu_tian_ci/GAFL/gpnn/config/config.yaml')['params_config']
    test_dataset=CLIPFeatureDatasetCLSREL('test',sample_each_clip=16,train=False)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=16)


    dataset=CLIPFeatureDatasetCLSREL('train')
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    # model=MyModelGCN(config,pretrain=pretrain).to(args.device)
    model=MymodelGPNN3(config,pretrain=pretrain).to(args.device)
    model.apply(weight_init)
    cri=Criterion(config)
    adapter=AdapterLoss(config)
    max_acc=999
    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    time_stamp=getTimeStamp()
    loss_record_path='/home/wu_tian_ci/GAFL/recoder/loss_value'
    t_path=time_stamp[0:8]
    loss_record_path=os.path.join(loss_record_path,t_path)
    if not os.path.exists(loss_record_path):
        os.mkdir(loss_record_path)
    loss_record_path=os.path.join(loss_record_path,time_stamp[8:12]+'loss.txt')
    if not os.path.exists(loss_record_path):
        f=open(loss_record_path,'w')
        f.write('begin to write\n')
        f.write('lr: '+str(args.lr)+' '+'epoch:'+str(args.epoch)+' '+args.sup+'\n')
        f.close()

    for epoch in range(args.epoch):
        model.train()
        txt_k=1
        counters=0
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                counters+=1
                frames,bbx,mask,label,cls_l,rel_l=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                label=label.to(device).squeeze()
                ans,rel,cls,rel_label,cls_label=model(frames,mask,bbx,rel_l,cls_l)
                # if counters==974:
                #     breakpoint()
                # action cls
                loss1,loss1_=cri(ans,label)
                # weighted relation obj_class
                loss2,loss2_1,loss2_2=adapter(rel,cls,rel_label,cls_label,epoch+1)
                loss=loss1+loss2
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                loss_str=str(loss1_)+'_'+str(loss2_1)+'_'+str(loss2_2)
                pbar.set_postfix({"Loss": loss_str})
                f=open(loss_record_path,'a')
                f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss rel cls:'+loss_str+ '\n')
                f.close()
                txt_k+=1
        evaluator.reset()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                frames,bbx,mask,label,cls_l,rel_l=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                label=label.to(device).squeeze()
                ans,rel,cls,rel_label,cls_label=model(frames,mask,bbx,rel_l,cls_l)
                evaluator.process(ans,label)
            metrics = evaluator.evaluate()

        if pretrain:
            save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'pretrain')
            print('saved')
        else:
            save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')


def train_rel_cls_no_mask(args,pretrain):                                                   
    config=load_config()
    
    test_dataset=CLIPFeatureDatasetCLSRELNonMask('test',sample_each_clip=16,train=False)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=16)


    dataset=CLIPFeatureDatasetCLSRELNonMask('train')
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    # model=MyModelGCN(config,pretrain=pretrain).to(args.device)
    model=MymodelGPNN6(config,pretrain=pretrain).to(args.device)
    model.apply(weight_init)
    cri=Criterion(config)
    adapter=AdapterLoss2(config)
    max_acc=999
    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)

    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    time_stamp=getTimeStamp()
    loss_record_path='/home/wu_tian_ci/GAFL/recoder/loss_value'
    t_path=time_stamp[0:8]
    loss_record_path=os.path.join(loss_record_path,t_path)
    if not os.path.exists(loss_record_path):
        os.mkdir(loss_record_path)
    loss_record_path=os.path.join(loss_record_path,time_stamp[8:12]+'loss.txt')
    if not os.path.exists(loss_record_path):
        f=open(loss_record_path,'w')
        f.write('begin to write\n')
        f.write('lr: '+str(args.lr)+' '+'epoch:'+str(args.epoch)+' '+args.sup+'\n')
        f.close()

    for epoch in range(args.epoch):
        model.train()
        txt_k=1
        counters=0
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                counters+=1
                frames,bbx,mask,label,cls_l,rel_l=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                label=label.to(device).squeeze()
                ans,global_cls,cls=model(frames,mask,bbx)
                # breakpoint()
                # if counters==974:
                #     breakpoint()
                # action cls
                loss1,loss1_=cri(ans,label)
                # weighted relation obj_class
                loss2,loss2_1=adapter(cls,rel_l,cls_l,epoch+1)
                loss3,loss3_1=cri(global_cls,label)
                loss=loss1+loss2+loss3
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                loss_str=str(loss1_)+'_'+str(loss3_1)+'_'+str(loss2_1)
                pbar.set_postfix({"Loss": loss_str})
                f=open(loss_record_path,'a')
                f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss rel cls:'+loss_str+ '\n')
                f.close()
                txt_k+=1
        evaluator.reset()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                frames,bbx,mask,label,cls_l,rel_l=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                # cls_l=cls_l.to(device)
                # rel_l=rel_l.to(device)
                label=label.to(device).squeeze()
                ans,global_cls,cls=model(frames,mask,bbx)
                evaluator.process(ans,label)
            metrics = evaluator.evaluate()

        if pretrain:
            save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'pretrain')
            print('saved')
        else:
            save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')

def train_rel_cls_text(args,pretrain):                                                   
    config=load_config()
    
    test_dataset=CLIPFeatureDatasetCLSRELNonMask('test',sample_each_clip=16,train=False)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=16)


    dataset=CLIPFeatureDatasetCLSRELNonMask('train')
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    # model=MyModelGCN(config,pretrain=pretrain).to(args.device)
    model=MymodelGPNNText(config,pretrain=pretrain).to(args.device)
    model.apply(weight_init)
    cri=Criterion(config)
    adapter=AdapterLoss(config)
    max_acc=999
    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    
    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    time_stamp=getTimeStamp()
    loss_record_path='/home/wu_tian_ci/GAFL/recoder/loss_value'
    t_path=time_stamp[0:8]
    loss_record_path=os.path.join(loss_record_path,t_path)
    if not os.path.exists(loss_record_path):
        os.mkdir(loss_record_path)
    loss_record_path=os.path.join(loss_record_path,time_stamp[8:12]+'loss.txt')
    if not os.path.exists(loss_record_path):
        f=open(loss_record_path,'w')
        f.write('begin to write\n')
        f.write('lr: '+str(args.lr)+' '+'epoch:'+str(args.epoch)+' '+args.sup+'\n')
        f.close()

    for epoch in range(args.epoch):
        model.train()
        txt_k=1
        counters=0
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                counters+=1
                frames,bbx,mask,label,cls_l,rel_l=batch
                bbx=bbx.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                label=label.to(device).squeeze()
                ans=model(cls_l,rel_l,bbx)
                # breakpoint()
                # if counters==974:
                #     breakpoint()
                # action cls
                loss1,loss1_=cri(ans,label)
                # weighted relation obj_class
                # loss2,loss2_1,loss2_2=adapter(rel,cls,rel_l,cls_l,epoch+1)
                loss=loss1
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                loss_str=str(loss1_)
                pbar.set_postfix({"Loss": loss_str})
                f=open(loss_record_path,'a')
                f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss rel cls:'+loss_str+ '\n')
                f.close()
                txt_k+=1
        evaluator.reset()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                frames,bbx,mask,label,cls_l,rel_l=batch
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                label=label.to(device).squeeze()
                ans=model(cls_l,rel_l,bbx)
                evaluator.process(ans,label)
            metrics = evaluator.evaluate()

        if pretrain:
            save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'pretrain')
            print('saved')
        else:
            save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')

def train_rel_cls_sperate(args,pretrain):
    config=load_config()
    config2=load_config('/home/wu_tian_ci/GAFL/gpnn/config/config.yaml')['params_config']
    test_dataset=CLIPFeatureDatasetCLSREL('test',sample_each_clip=16,train=False)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=16)


    dataset=CLIPFeatureDatasetCLSREL('train')
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    # model=MyModelGCN(config,pretrain=pretrain).to(args.device)
    model=MymodelGPNN4(config,config2,pretrain=pretrain).to(args.device)
    model.apply(weight_init)


    cri=Criterion(config)
    adapter=AdapterLoss(config)
    sloss=SeperationLoss(config)
    rloss=ReconstructLoss(config)

    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    time_stamp=getTimeStamp()
    loss_record_path='/home/wu_tian_ci/GAFL/recoder/loss_value'
    t_path=time_stamp[0:8]
    loss_record_path=os.path.join(loss_record_path,t_path)
    if not os.path.exists(loss_record_path):
        os.mkdir(loss_record_path)
    loss_record_path=os.path.join(loss_record_path,time_stamp[8:12]+'loss.txt')
    if not os.path.exists(loss_record_path):
        f=open(loss_record_path,'w')
        f.write('begin to write\n')
        f.write('lr: '+str(args.lr)+' '+'epoch:'+str(args.epoch)+' '+args.sup+'\n')
        f.close()
    counters=0
    for epoch in range(args.epoch):
        model.train()
        txt_k=1
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                counters+=1
                frames,bbx,mask,label,cls_l,rel_l=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                label=label.to(device).squeeze()


                ans_common,common_features,private_features,reconstruct_features,target_features,\
                ans_all,rel,cls,rel_label,cls_label=model(frames,mask,bbx,rel_l,cls_l)
                if len(label.shape)==1:
                    label.unsqueeze_(0)

                entropy_pri=torch.mean(torch.log2(torch.std(private_features,-1)))
                # entropy_com=torch.mean(torch.log2(torch.std(common_features,-1)))
                # action cls common
                loss1,loss1_1=cri(ans_common,label)
                # action cls common+private
                loss2,loss2_1=cri(ans_all,label)
                # sperate
                loss3,loss3_1=sloss(common_features,private_features)
                # reconstruct loss
                loss4,loss4_1=rloss(reconstruct_features,target_features)
                # weighted relation obj_class
                loss5,loss5_1,loss5_2=adapter(rel,cls,rel_label,cls_label)

                loss=loss1+loss3+loss4+loss5+loss2
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                loss_str=str(loss1_1)+'_'+str(loss2_1)+'_'+str(loss3_1)+'_'+str(loss4_1)+'_'+str(loss5_1)+'_'+str(loss5_2)+'_'+str(round(entropy_pri.item(),2))
                pbar.set_postfix({"Loss": loss_str})
                f=open(loss_record_path,'a')
                f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss rel cls:'+loss_str+ '\n')
                f.close()
                txt_k+=1
        evaluator.reset()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                frames,bbx,mask,label,cls_l,rel_l=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                label=label.to(device).squeeze()
                ans_common,common_features,private_features,reconstruct_features,target_features,\
                ans_all,rel,cls,rel_label,cls_label=model(frames,mask,bbx,rel_l,cls_l)
                if len(label.shape)==1:
                    label.unsqueeze_(0)
                evaluator.process(ans_common,label)
            metrics = evaluator.evaluate()

        if pretrain:
            save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'pretrain')
            print('saved')
        else:
            save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')

def load_model_dict(p,model,s=False):
    checkpoint=torch.load(p,'cpu')
    model_weight=checkpoint['model']
    model.load_state_dict(model_weight,strict=s)
    return model



def train_oracle(args,pretrain):
    config=load_config()
    config.prompt.type=args.prompt
    test_dataset=MixAns2('test',sample_each_clip=16,train=False,mapping_type=args.ds)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)
    dataset=MixAns2('train',sample_each_clip=16,train=True,mapping_type=args.ds)
    if args.loss==0:
        config.loss.spe.weight=0.
        config.loss.rec.weight=0.
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    # model=MyModelGCN(config,pretrain=pretrain).to(args.device)

    # print(args.model=='mix2',args.model)
    if args.model=='mix2':
        model=GPNNMix2(config,flag=pretrain).to(device)
    elif args.model=='mix':
        model=GPNNMix(config,flag=pretrain).to(device)
    elif args.model=='mix3':
        model=GPNNMix3(config,flag=pretrain,train_stage=args.stage).to(device)
    elif args.model=='mix4':
        model=GPNNMix4(config,flag=pretrain,train_stage=args.stage).to(device)
    else:
        raise NotImplementedError
    model.apply(weight_init)
    # load_model_dict('/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250319/1503/20_c:88.3695_p:58.54179_o:59.57755.pth')
    # model=load_model_dict('/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250319/1503/20_c:88.3695_p:58.54179_o:59.57755.pth'
    #                 ,model)
    # breakpoint()
    cri=Criterion(config)
    adapter=AdapterLoss(config)
    sloss=SeperationLoss(config)
    rloss=ReconstructLoss(config)


    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    # total common private middle
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    evaluator2 = MyEvaluatorActionGenome(len(test_dataset),158)
    evaluator3 = MyEvaluatorActionGenome(len(test_dataset),158)
    evaluator4 = MyEvaluatorActionGenome(len(test_dataset),157)
    time_stamp=getTimeStamp()
    loss_record_path='/home/wu_tian_ci/GAFL/recoder/loss_value'
    t_path=time_stamp[0:8]
    loss_record_path_=os.path.join(loss_record_path,t_path)
    if not os.path.exists(loss_record_path_):
        os.mkdir(loss_record_path_)
    loss_record_path=os.path.join(loss_record_path_,time_stamp[8:12]+'loss.txt')
    if not os.path.exists(loss_record_path):
        f=open(loss_record_path,'w')
        f.write('begin to write\n')
        f.write('lr: '+str(args.lr)+' '+'epoch:'+str(args.epoch)+' '+args.sup+'\n')
        f.close()
        if not os.path.exists(os.path.join('/home/wu_tian_ci/GAFL/recoder/checkpoint','pretrain',time_stamp[:8],time_stamp[8:12])):
            os.makedirs(os.path.join('/home/wu_tian_ci/GAFL/recoder/checkpoint','pretrain',time_stamp[:8],time_stamp[8:12]))
        write_config(config,args,[os.path.join(loss_record_path_,time_stamp[8:12]+'pretrain_config_args.txt'),
                                  os.path.join('/home/wu_tian_ci/GAFL/recoder/checkpoint','pretrain',time_stamp[:8],time_stamp[8:12],
                                               'config_args.txt')])
    counters=0
    for epoch in range(args.epoch):
        model.train()
        txt_k=1
        train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                counters+=1
                frames,bbx,mask,label,cls_ids,cls_l,rel_l,private_label,common_label,token_tensor,mask_=batch
                # breakpoint()
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                label=label.to(device).squeeze()
                cls_ids=cls_ids.to(device)

                token_tensor=token_tensor.to(device)
                private_label=private_label.to(device)
                common_label=common_label.to(device)
                mask_=mask_.to(device)

                
                if args.stage in [1] :
                    c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor,mask,mask_)
                    # breakpoint()
                elif args.stage in [3,5]:
                    m_ans,cls_ans,rel_ans=model(frames,cls_ids,rel_l,bbx,token_tensor)
                else:
                    raise NotImplementedError
                # loss=loss1+loss2+loss3+loss4+loss5+loss6
                if args.stage in [3,5]:
                    loss8,loss8_1=cri(m_ans,label)
                    loss5,loss5_1,loss5_2=adapter(rel_ans,cls_ans,rel_l,cls_l,epoch+1)
                    loss=loss8+loss5
                elif args.stage in [1]:
                    loss4,loss4_1=rloss(human_obj_feature,recs,epoch+1)
                    loss2,loss2_1,=cri(t_ans,label) 
                    loss1,loss1_1=cri(c_ans,common_label)
                    loss6,loss6_1=cri(p_ans,private_label)
                    d_weight=model.get_weight(loss6,loss1)
                    loss5,loss5_1,loss5_2=adapter(rel_ans,cls_ans,rel_l,cls_l,epoch+1)
                    loss3,loss3_1=sloss(c_features,p_features,epoch+1)
                    loss=loss1+loss5+loss6*config.loss.pri*d_weight+loss2+loss3+loss4
                else:
                    raise NotImplementedError

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)

                if args.stage in [1]:
                    # common private rel cls sep total recs
                    loss_str=str(loss1_1)+"_"+str(loss6_1)+"_"+str(loss5_1)+"_"+str(loss5_2)+"_"+str(loss3_1)+"_"+str(loss2_1)+"_"+str(loss4_1)
                elif args.stage in [3,5]:
                    loss_str=str(loss8_1)+"_"+str(loss5_2)
                else:
                    raise NotImplementedError

                pbar.set_postfix({"Loss": loss_str})
                f=open(loss_record_path,'a')
                f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss :'+loss_str+ '\n')
                f.close()
                txt_k+=1
        evaluator.reset()
        evaluator2.reset()
        evaluator3.reset()
        evaluator4.reset()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                frames,bbx,mask,label,cls_ids,cls_l,rel_l,private_label,common_label,token_tensor,mask_=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                cls_ids=cls_ids.to(device)
                token_tensor=token_tensor.to(device)
                private_label=private_label.to(device)
                common_label=common_label.to(device)

                label=label.to(device).squeeze()
                if args.stage in [1] :
                    c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor)
                elif args.stage in [3,5]:
                    m_ans,cls_ans,rel_ans=model(frames,cls_ids,rel_l,bbx,token_tensor)
                else:
                    raise NotImplementedError
                if len(label.shape)==1:
                    label.unsqueeze_(0)
                    common_label.unsqueeze_(0)
                    private_label.unsqueeze_(0)
                
                if args.stage in [1]:
                    evaluator.process(t_ans,label)
                    evaluator2.process(c_ans,common_label)
                    evaluator3.process(p_ans,private_label)
                elif args.stage in [3,5]:
                    evaluator4.process(m_ans,label)
                else:
                    raise NotImplementedError
                
            if args.stage in [1]:
                metrics2 = evaluator2.evaluate()
                metrics3 = evaluator3.evaluate()
                metrics = evaluator.evaluate()
            elif args.stage in [3,5]:
                metrics4=evaluator4.evaluate()
            else:
                raise NotImplementedError
            

        if args.stage in [1]:
            acc_str='t:'+str(round(metrics['map']*100,5))+'_c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))\

        elif args.stage in [3,5]:
            acc_str='o1:'+str(round(metrics4['map']*100,5))
        else:
            raise NotImplementedError
        
        save_checkpoint(epoch+1,model,acc_str,optimizer,scheduler,time_stamp,'pretrain')
        print('saved')
  
def train_oracle_continue(args,pretrain,p):
    config=load_config()
    config.prompt.type=args.prompt
    test_dataset=MixAns2('test',sample_each_clip=16,train=False,mapping_type=args.ds)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)


    dataset=MixAns2('train',sample_each_clip=16,train=True,mapping_type=args.ds)
    
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    # model=MyModelGCN(config,pretrain=pretrain).to(args.device)

    # print(args.model=='mix2',args.model)
    if args.model=='mix4':
        model=GPNNMix4(config,flag=pretrain,train_stage=args.stage).to(device)
    else:
        raise NotImplementedError

    # load_model_dict(p)
    model=load_model_dict(p,model,True)
    
    model.apply(weight_init)
    # breakpoint()
    cri=Criterion(config)

    # sloss2=KLSeperation(config)
    print('lr: ',args.lr)
    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.wr * num_batches * args.epoch,),
        num_training_steps=args.epoch * num_batches,
    )
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    evaluator2 = MyEvaluatorActionGenome(len(test_dataset),158)
    evaluator3 = MyEvaluatorActionGenome(len(test_dataset),158)
    evaluator4 = MyEvaluatorActionGenome(len(test_dataset),157)
    evaluator4_1 = MyEvaluatorActionGenome(len(test_dataset),157)
    time_stamp=getTimeStamp()
    loss_record_path='/home/wu_tian_ci/GAFL/recoder/loss_value'
    t_path=time_stamp[0:8]
    loss_record_path_=os.path.join(loss_record_path,t_path)
    if not os.path.exists(loss_record_path_):
        os.mkdir(loss_record_path_)
    loss_record_path=os.path.join(loss_record_path_,time_stamp[8:12]+'loss.txt')
    if not os.path.exists(loss_record_path):
        f=open(loss_record_path,'w')
        f.write('begin to write\n')
        f.write(p+'continue \n')
        f.write('lr: '+str(args.lr)+' '+'epoch:'+str(args.epoch)+' '+args.sup+'\n')
        f.close()
        if not os.path.exists(os.path.join('/home/wu_tian_ci/GAFL/recoder/checkpoint','train',time_stamp[:8],time_stamp[8:12])):
            os.makedirs(os.path.join('/home/wu_tian_ci/GAFL/recoder/checkpoint','train',time_stamp[:8],time_stamp[8:12]))
        write_config(config,args,[os.path.join(loss_record_path_,time_stamp[8:12]+'train_config_args.txt'),
                                  os.path.join('/home/wu_tian_ci/GAFL/recoder/checkpoint','train',time_stamp[:8],time_stamp[8:12],
                                               'config_args.txt')])
    counters=0
    for epoch in range(args.epoch):
        model.train()
        txt_k=1
        train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                counters+=1
                frames,bbx,mask,label,cls_ids,cls_l,rel_l,private_label,common_label,token_tensor,mask_=batch
                # breakpoint()
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                label=label.to(device).squeeze()
                cls_ids=cls_ids.to(device)
                token_tensor=token_tensor.to(device)
                private_label=private_label.to(device)
                common_label=common_label.to(device)
                mask_=mask_.to(device)

                # breakpoint()

                t_ans,m_ans1=model(frames,cls_ids,rel_l,bbx,token_tensor)


                # loss=loss1+loss2+loss3+loss4+loss5+loss6

                loss2,loss2_1,=cri(t_ans,label) 
                loss8,loss8_1=cri(m_ans1,label)
                loss=loss8+loss2



                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)


                loss_str=str(loss8_1)+"_"+str(loss2_1)
                pbar.set_postfix({"Loss": loss_str})
                f=open(loss_record_path,'a')
                f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss :'+loss_str+ '\n')
                f.close()
                txt_k+=1
        evaluator.reset()
        evaluator2.reset()
        evaluator3.reset()
        evaluator4.reset()
        evaluator4_1.reset()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                frames,bbx,mask,label,cls_ids,cls_l,rel_l,private_label,common_label,token_tensor,mask_=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                cls_ids=cls_ids.to(device)
                token_tensor=token_tensor.to(device)
                private_label=private_label.to(device)
                common_label=common_label.to(device)

                label=label.to(device).squeeze()

                t_ans,m_ans1=model(frames,cls_ids,rel_l,bbx,token_tensor)
                
                if len(label.shape)==1:
                    label.unsqueeze_(0)
                    t_ans.unsqueeze_(0)
                    common_label.unsqueeze_(0)
                    private_label.unsqueeze_(0)
                if len(m_ans1.shape)==1:
                        m_ans1.unsqueeze_(0)

                evaluator.process(t_ans,label)
                evaluator4.process(m_ans1,label)
                # evaluator4_1.process(m_ans2,label)

            metrics = evaluator.evaluate()

            metrics4 = evaluator4.evaluate()
            # metrics4_1 = evaluator4_1.evaluate()
        # total_ans acc   common_ans acc
 
        acc_str='t:'+str(round(metrics['map']*100,5))+'_o1:'+str(round(metrics4['map']*100,5))
        save_checkpoint(epoch+1,model,acc_str,optimizer,scheduler,time_stamp,'train')
        print('saved')


def train_clip_stlt(args):
    config=load_config()
    test_dataset=CLIPFeatureDatasetAll('test',sample_each_clip=16,train=False)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=16)


    dataset=CLIPFeatureDatasetAll('train')
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    model=StltImage(config).to(args.device)
    model.apply(weight_init_dis)
    cri=Criterion()
    max_acc=999
    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    time_stamp=getTimeStamp()
    loss_record_path='/home/wu_tian_ci/GAFL/recoder/loss_value'
    t_path=time_stamp[0:8]
    loss_record_path=os.path.join(loss_record_path,t_path)

    for epoch in range(args.epoch):
        model.train()
        txt_k=1
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                frames,bbx,mask,label=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                label=label.to(device).squeeze()
                ans=model(frames,bbx)
                loss=cri(ans,label)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
                txt_k+=1
        evaluator.reset()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                frames,bbx,mask,label=batch
                frames=frames.to(device)
                bbx=bbx.to(device)
                
                label=label.to(device).squeeze()
                ans=model(frames,bbx)
                evaluator.process(ans,label)
            metrics = evaluator.evaluate()
        print(epoch,":",round(metrics['map']*100,5))

def train_text(args):
    config=load_config()
    test_df=DataConfig(
        dataset_name='action_genome',
        dataset_path='test_6.json',
        labels_path='ag.json',
        videoid2size_path='ag.json',
        layout_num_frames=16,
        appearance_num_frames=16,
        videos_path='test_6.hdf5',
        train=False,
    )
    test_dataset=StltDataset(name='test',config=test_df)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=16)


    train_df=DataConfig(
        dataset_name='action_genome',
        dataset_path='train_6.json',
        labels_path='ag.json',
        videoid2size_path='ag.json',
        layout_num_frames=16,
        appearance_num_frames=16,
        videos_path='train_6.hdf5',
        train=True,
    )                       
    train_dataset=StltDataset(name='train',config=train_df)
    train_loader=DataLoader(train_dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    model=Stlt(config)
    model=model.to(device)
    criterion=Criterion(config)
    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(train_dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)

    for epoch in range(args.epoch):
        model.train(True)
        txt_k=1
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                video_id,cate,box,scor,ft,length,label=batch
                cate=cate.to(device)
                box=box.to(device)
                label=label.to(device).squeeze()
                ans=model(cate,box)
                loss,_=criterion(ans,label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
                # f=open(loss_record_path,'a')
                # f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss:'+str(round(loss.item(),5))+ '\n')
                # f.close()
                txt_k+=1
        model.train(False)
        evaluator.reset()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                video_id,cate,box,scor,ft,length,label=batch
                cate=cate.to(device)
                box=box.to(device)
                label=label.to(device).squeeze()
                ans=model(cate,box)
                evaluator.process(ans,label)
        metrics = evaluator.evaluate()
        if evaluator.is_best():
            logging.info("=================================")
            logging.info(f"Found new best on epoch {epoch+1}!")
            logging.info("=================================")
            # torch.save(model.state_dict(), args.save_model_path)
            # if args.save_backbone_path:
            #     torch.save(model.backbone.state_dict(), args.save_backbone_path)
        for m in metrics.keys():
            logging.info(f"{m}: {round(metrics[m] * 100, 2)}")
            print(epoch,m,round(metrics[m] * 100, 2))
        # if pretrain:
        #     save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'pretrain')
        #     print('saved')
        # else:
        #     save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')

def train_text2(args):
    config=load_config()
    test_df=DataConfig(
        dataset_name='action_genome',
        dataset_path='test_6.json',
        labels_path='ag.json',
        videoid2size_path='ag.json',
        layout_num_frames=16,
        appearance_num_frames=16,
        videos_path='test_6.hdf5',
        train=False,
    )
    test_dataset=StltDataset(name='test',config=test_df)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=16)


    train_df=DataConfig(
        dataset_name='action_genome',
        dataset_path='train_6.json',
        labels_path='ag.json',
        videoid2size_path='ag.json',
        layout_num_frames=16,
        appearance_num_frames=16,
        videos_path='train_6.hdf5',
        train=True,
    )
    train_dataset=StltDataset(name='train',config=train_df)
    train_loader=DataLoader(train_dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    model=MymodelGPNNText(config,pretrain=True).to(args.device)
    model=model.to(device)
    criterion=Criterion(config)
    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(train_dataset) // args.batchsize
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)

    for epoch in range(args.epoch):
        model.train(True)
        txt_k=1
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                video_id,cate,box,scor,ft,length,label=batch
                cate=cate.to(device)
                box=box.to(device)
                label=label.to(device).squeeze()
                ans=model(cate,box)
                loss,_=criterion(ans,label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
                # f=open(loss_record_path,'a')
                # f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss:'+str(round(loss.item(),5))+ '\n')
                # f.close()
                txt_k+=1
        model.train(False)
        evaluator.reset()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                video_id,cate,box,scor,ft,length,label=batch
                cate=cate.to(device)
                box=box.to(device)
                label=label.to(device).squeeze()
                ans=model(cate,box)
                evaluator.process(ans,label)
        metrics = evaluator.evaluate()
        if evaluator.is_best():
            logging.info("=================================")
            logging.info(f"Found new best on epoch {epoch+1}!")
            logging.info("=================================")
            # torch.save(model.state_dict(), args.save_model_path)
            # if args.save_backbone_path:
            #     torch.save(model.backbone.state_dict(), args.save_backbone_path)
        for m in metrics.keys():
            logging.info(f"{m}: {round(metrics[m] * 100, 2)}")
            print(epoch,m,round(metrics[m] * 100, 2))
        # if pretrain:
        #     save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'pretrain')
        #     print('saved')
        # else:
        #     save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')


def train_pure(args,pretrain):
    config=load_config()

    test_dataset=CLIPFeatureDatasetCLSRELOracle('test',sample_each_clip=16,train=False)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=16)


    dataset=CLIPFeatureDatasetCLSRELOracle('train',sample_each_clip=16,train=False)
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=16,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    # model=MyModelGCN(config,pretrain=pretrain).to(args.device)
    # print(args.model=='mix2',args.model)
    model=PureMix(config).to(device)
 
    model.apply(weight_init)


    cri=Criterion(config)
    adapter=AdapterLoss(config)
    sloss=SeperationLoss(config)
    rloss=ReconstructLoss(config)
    sloss2=KLSeperation(config)

    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)

    time_stamp=getTimeStamp()
    loss_record_path='/home/wu_tian_ci/GAFL/recoder/loss_value'
    t_path=time_stamp[0:8]
    loss_record_path=os.path.join(loss_record_path,t_path)
    if not os.path.exists(loss_record_path):
        os.mkdir(loss_record_path)
    loss_record_path=os.path.join(loss_record_path,time_stamp[8:12]+'loss.txt')
    if not os.path.exists(loss_record_path):
        f=open(loss_record_path,'w')
        f.write('begin to write\n')
        f.write('lr: '+str(args.lr)+' '+'epoch:'+str(args.epoch)+' '+args.sup+'\n')
        f.close()
    counters=0
    for epoch in range(args.epoch):
        model.train()
        txt_k=1
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                counters+=1
                frames,bbx,mask,label,cls_ids,cls_cls,rel=batch
                # breakpoint()
                frames=frames.to(device)
                bbx=bbx.to(device)
                
                cls_ids=cls_ids.to(device)
                cls_cls=cls_cls.to(device)
                rel=rel.to(device)
                label=label.to(device).squeeze()


                m_ans,rel_ans,cls_ans=model(frames,cls_ids,bbx)

                loss8,loss8_1=cri(m_ans,label)
    
                loss5,loss5_1,loss5_2=adapter(rel_ans,cls_ans,rel,cls_cls,epoch+1)

                loss=loss5+loss8

                # loss=loss1+loss2+loss4+loss5+loss6*config.loss.pri

                # loss=loss1+loss2+loss5

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                loss_str=str(loss5_1)+"_"+str(loss5_2)+"_"+str(loss8_1)

                pbar.set_postfix({"Loss": loss_str})
                f=open(loss_record_path,'a')
                f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss :'+loss_str+ '\n')
                f.close()
                txt_k+=1
        evaluator.reset()

        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                frames,bbx,mask,label,cls_ids,cls_cls,rel=batch
                # breakpoint()
                frames=frames.to(device)
                bbx=bbx.to(device)
                
                cls_ids=cls_ids.to(device)
                cls_cls=cls_cls.to(device)
                rel=rel.to(device)
                label=label.to(device).squeeze()

                
                
                if len(label.shape)==1:
                    label.unsqueeze_(0)
                m_ans,rel_ans,cls_ans=model(frames,cls_ids,bbx)
                evaluator.process(m_ans,label)
            metrics = evaluator.evaluate()

        # total_ans acc   common_ans acc
        
        acc_str='t:'+str(round(metrics['map']*100,5))

        
        save_checkpoint(epoch+1,model,acc_str,optimizer,scheduler,time_stamp,'pretrain')
        print('saved')

        # if pretrain:
        #     acc_str='t:'+str(round(metrics['map']*100,5))+'_c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))
        #     save_checkpoint(epoch+1,model,acc_str,optimizer,scheduler,time_stamp,'pretrain')
        #     print('saved')
        # else:
        #     save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')
 

#  a sample for test
def test_(args,pretrain):
    config=load_config()
    config.prompt.type=args.prompt
    test_dataset=MixAns2('test',sample_each_clip=16,train=False,mapping_type=args.ds)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)
    dataset=MixAns2('train',sample_each_clip=16,train=True,mapping_type=args.ds)
    if args.loss==0:
        config.loss.spe.weight=0.
        config.loss.rec.weight=0.
    device=args.device


    model=GPNNMix4(config,flag=pretrain,train_stage=args.stage).to(device)

    model.apply(weight_init)
    # load_model_dict('/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250319/1503/20_c:88.3695_p:58.54179_o:59.57755.pth')
    # model=load_model_dict('/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250319/1503/20_c:88.3695_p:58.54179_o:59.57755.pth'
    #                 ,model)
    # breakpoint()
    cri=Criterion(config)
    adapter=AdapterLoss(config)
    sloss=SeperationLoss(config)
    rloss=ReconstructLoss(config)


    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    # total common private middle
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    evaluator2 = MyEvaluatorActionGenome(len(test_dataset),158)
    evaluator3 = MyEvaluatorActionGenome(len(test_dataset),158)
    evaluator4 = MyEvaluatorActionGenome(len(test_dataset),157)
    counters=0
    for epoch in range(args.epoch):
        model.train()
        txt_k=1
        train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                counters+=1
                frames,bbx,mask,label,cls_ids,cls_l,rel_l,private_label,common_label,token_tensor,mask_=batch
                # breakpoint()
                frames=frames.to(device)
                bbx=bbx.to(device)
                mask=mask.to(device)
                cls_l=cls_l.to(device)
                rel_l=rel_l.to(device)
                label=label.to(device).squeeze()
                cls_ids=cls_ids.to(device)

                token_tensor=token_tensor.to(device)
                private_label=private_label.to(device)
                common_label=common_label.to(device)
                mask_=mask_.to(device)
                c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor,mask)
                breakpoint()
                
 

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Packs PIL images as HDF5.")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2",
        help="gpu device",
    )
    parser.add_argument(
        "--sup",
        type=str,
        default="nothing",
        help="sth to say",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="train epochs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="warmup epochs",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=32,
        help="batchsize",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=1e-3,
        help="learning rate",
    )
    parser.add_argument(
            "--clip_val",
            type=float,
            default=5.0,
            help="The gradient clipping value.",
        )
    parser.add_argument(
            "--wr",
            type=float,
            default=.1,
            help="warm up rate for continue",
        )
    parser.add_argument(
        "--model",
        type=str,
        default="mix4",
        help="model",
    )
    parser.add_argument(
        "--ds",
        type=int,
        default=5,
        help="dataset",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="train stage",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=0,
        help="train type 0:oracle 1:oracle continue 2:pure",
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=1,
        help="prompt type 0:smiple 1:gpfp",
    )
    parser.add_argument(
        "--loss",
        type=int,
        default=1,
        help="speration and reconstruction loss,0 no loss,1 loss",
    )


    set_seed(seed=3407)
    args = parser.parse_args()
    # test_(args,False)
    # train(args,True
    if args.tp ==0:
        train_oracle(args,False)
    elif args.tp==1:
        # p='/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250327/1238/20_t:61.87185_c:88.61021_p:58.9198_o1:59.48385.pth'
        p='/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250407/1849/20_t:56.0987_c:98.34421_p:53.22037.pth'
        train_oracle_continue(args,False,p)
    else:
        raise NotImplementedError
    # train_text2(args) 
    # train_rel_cls_no_mask(args,True)
    # train_rel_cls_sperate(args,True)
    # train_clip_stlt(args)
    