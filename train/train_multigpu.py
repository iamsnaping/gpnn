import sys
sys.path.append('/home/wu_tian_ci/GAFL')
import torch.optim as optim
import torch
from myutils.extra_model import (GPNNMix4)
from myutils.mydataset import (
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
from torch.nn import SyncBatchNorm


import warnings

# 将所有警告转换为异常
# warnings.filterwarnings('error')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
# os.environ['OMP_NUM_THREADS'] = '2'

import torch.distributed as dist
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2"
os.environ['OMP_NUM_THREADS'] = '1'
# def set_seed(seed=3407):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed) 

def set_seed(seed=3407,local_rank=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)


def weight_init_dis(m):

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


def reduce_mean(name,tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone().to(tensor)
    # print(name,'tensor',tensor.device)
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def load_model_dict(p,model,s=False):
    checkpoint=torch.load(p,'cpu')
    model_weight=checkpoint['model']
    model.load_state_dict(model_weight,strict=s)
    return model



def train_oracle(args,pretrain):
    config=load_config()
    config.prompt.type=args.prompt
    config.normtype=args.normtype
    if dist.is_initialized:
        worldsize=dist.get_world_size()
        config.worldsize=worldsize if worldsize!=-1 else config.worldsize
    if args.smw==0:
        config.loss.spe.mw=0
    if args.rmw==0:
        config.loss.rec.mw=0
    config.loss.rec.margin=args.rm
    config.loss.spe.margin=args.sm

    test_dataset=MixAns2('test',sample_each_clip=16,train=False,mapping_type=args.ds)
    testts=torch.utils.data.distributed.DistributedSampler(test_dataset,shuffle=False)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12,
                           sampler=testts)
    dataset=MixAns2('train',sample_each_clip=16,train=True,mapping_type=args.ds)
    traints=torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,
                            sampler=traints)
    if args.loss==0:
        config.loss.spe.weight=0.
        config.loss.rec.weight=0.
    device=args.device
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK']=str(args.local_rank)
    local_rank=args.local_rank
    set_seed(seed=args.seed,local_rank=local_rank)
    device = torch.device(local_rank)
    num_batches = len(dataset) // (args.batchsize*3)
    model=GPNNMix4(config,flag=pretrain,train_stage=args.stage).to(device)

    model.apply(weight_init_dis)
    if args.normtype==1:
        model=SyncBatchNorm.convert_sync_batchnorm(model)
    model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)

    
    cri=Criterion(config)
    adapter=AdapterLoss(config)
    sloss=SeperationLoss(config)
    rloss=ReconstructLoss(config)


    parameters = add_weight_decay(model.module, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)

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

    if local_rank==0:
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
        txt_k=0
        traints.set_epoch(epoch)
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

                
                if args.mt==0:
                    c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor)
                elif args.mt==1:
                    c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor,mask)
                elif args.mt==2:
                    c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor,mask,mask_)

                # loss=loss1+loss2+loss3+loss4+loss5+loss6


                loss4,loss4_1=rloss(human_obj_feature,recs,epoch+1)
                loss2,loss2_1,=cri(t_ans,label) 
                loss1,loss1_1=cri(c_ans,common_label)
                loss6,loss6_1=cri(p_ans,private_label)
                d_weight=model.module.get_weight(loss6,loss1)
                loss5,loss5_1,loss5_2=adapter(rel_ans,cls_ans,rel_l,cls_l,epoch+1)
                loss3,loss3_1=sloss(c_features,p_features,epoch+1)
                loss=loss1+loss5+loss6*config.loss.pri*d_weight+loss2+loss3+loss4


                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                # if local_rank==0:
                    
                scheduler.step()
                pbar.update(1)

                loss_str=str(loss1_1)+"_"+str(loss6_1)+"_"+str(loss5_1)+"_"+str(loss5_2)+"_"+str(loss3_1)+"_"+str(loss2_1)+"_"+str(loss4_1)
                pbar.set_postfix({"Loss": loss_str})
                if local_rank==0:
                    
                    f=open(loss_record_path,'a')
                    f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss :'+loss_str+ '\n')
                    f.close()
                    txt_k+=1
        model.eval()
        # common private total
        c_pre,c_lab=[],[]
        p_pre,p_lab=[],[]
        t_pre,t_lab=[],[]
        with torch.no_grad():
            testts.set_epoch(epoch)
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

                if args.mt==0:
                    c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor)
                elif args.mt==1:
                    c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor,mask)
                elif args.mt==2:
                    c_ans,p_ans,cls_ans,rel_ans,c_features,p_features,recs,human_obj_feature,t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor,mask,mask_)
                if len(label.shape)==1:
                    label.unsqueeze_(0)
                    common_label.unsqueeze_(0)
                    private_label.unsqueeze_(0)
                c_pre.append(c_ans)
                p_pre.append(p_ans)
                t_pre.append(t_ans)
                c_lab.append(common_label)
                p_lab.append(private_label)
                t_lab.append(label)


        p_pred = distributed_concat(torch.concat(p_pre, dim=0),len(testts.dataset))
        p_labe = distributed_concat(torch.concat(p_lab, dim=0),len(testts.dataset))
        c_pred = distributed_concat(torch.concat(c_pre, dim=0),len(testts.dataset))
        c_labe = distributed_concat(torch.concat(c_lab, dim=0),len(testts.dataset))
        t_pred = distributed_concat(torch.concat(t_pre, dim=0),len(testts.dataset))
        t_labe = distributed_concat(torch.concat(t_lab, dim=0),len(testts.dataset))
        if local_rank==0:
            evaluator.reset()
            evaluator2.reset()
            evaluator3.reset()
            evaluator4.reset()
            evaluator2.process(c_pred,c_labe)
            evaluator3.process(p_pred,p_labe)
            evaluator.process(t_pred,t_labe)
            metrics = evaluator.evaluate()
            metrics2 = evaluator2.evaluate()
            metrics3 = evaluator3.evaluate()
            if args.stage in [1,6,7]:
                acc_str='t:'+str(round(metrics['map']*100,5))+'_c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))            
            save_checkpoint(epoch+1,model.module,acc_str,optimizer,scheduler,time_stamp,'pretrain')
            print('saved')

def train_oracle_continue(args,pretrain,p):
    config=load_config()
    config.prompt.type=args.prompt
    config.normtype=args.normtype
    if dist.is_initialized:
        worldsize=dist.get_world_size()
        config.worldsize=worldsize if worldsize!=-1 else config.worldsize
    if args.smw==0:
        config.loss.spe.mw=0
    if args.rmw==0:
        config.loss.rec.mw=0
    config.loss.rec.margin=args.rm
    config.loss.spe.margin=args.sm
    test_dataset=MixAns2('test',sample_each_clip=16,train=False,mapping_type=args.ds)
    testts=torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12,
                           sampler=testts)
    dataset=MixAns2('train',sample_each_clip=16,train=True,mapping_type=args.ds)
    traints=torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,
                            sampler=traints)
    if args.loss==0:
        config.loss.spe.weight=0.
        config.loss.rec.weight=0.
    device=args.device
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK']=str(args.local_rank)
    set_seed(local_rank=args.local_rank)
    local_rank=args.local_rank
    device = torch.device(local_rank)
    model=GPNNMix4(config,flag=pretrain,train_stage=args.stage).to(device)

    model=load_model_dict(p,model,True)
    model.apply(weight_init_dis)
    model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank,find_unused_parameters=False)
    num_batches = len(dataset) // (args.batchsize*3)
    
    cri=Criterion(config)



    parameters = add_weight_decay(model.module, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    # total common private middle
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    evaluator4 = MyEvaluatorActionGenome(len(test_dataset),157)

    if local_rank==0:
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
            if not os.path.exists(os.path.join('/home/wu_tian_ci/GAFL/recoder/checkpoint','train',time_stamp[:8],time_stamp[8:12])):
                os.makedirs(os.path.join('/home/wu_tian_ci/GAFL/recoder/checkpoint','train',time_stamp[:8],time_stamp[8:12]))
            write_config(config,args,[os.path.join(loss_record_path_,time_stamp[8:12]+'train_config_args.txt'),
                                    os.path.join('/home/wu_tian_ci/GAFL/recoder/checkpoint','train',time_stamp[:8],time_stamp[8:12],
                                                'config_args.txt')])
    counters=0
    
    for epoch in range(args.epoch):
        model.train()
        txt_k=0
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

                
                t_ans,m_ans1=model(frames,cls_ids,rel_l,bbx,token_tensor)
                loss2,loss2_1,=cri(t_ans,label) 
                loss8,loss8_1=cri(m_ans1,label)
                loss=loss8+loss2
                # loss=loss1+loss2+loss3+loss4+loss5+loss6



                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                # if local_rank==0:
                    
                scheduler.step()
                pbar.update(1)

                loss_str=str(loss8_1)+"_"+str(loss2_1)
                pbar.set_postfix({"Loss": loss_str})
                if local_rank==0:
                    
                    f=open(loss_record_path,'a')
                    f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss :'+loss_str+ '\n')
                    f.close()
                    txt_k+=1
        model.eval()
        # common private total

        t_pre,t_lab=[],[]
        m_pre,m_lab=[],[]
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

                
                if len(label.shape)==1:
                    label.unsqueeze_(0)
                    common_label.unsqueeze_(0)
                    private_label.unsqueeze_(0)
                t_ans,m_ans1=model(frames,cls_ids,rel_l,bbx,token_tensor)
                t_pre.append(t_ans)
                t_lab.append(label)
                m_pre.append(m_ans1)
                m_lab.append(label)

        t_pred = distributed_concat(torch.concat(t_pre, dim=0),len(testts.dataset))
        t_labe = distributed_concat(torch.concat(t_lab, dim=0),len(testts.dataset))
        m_pred = distributed_concat(torch.concat(m_pre, dim=0),len(testts.dataset))
        m_labe = distributed_concat(torch.concat(m_lab, dim=0),len(testts.dataset))
        if local_rank==0:
            evaluator.reset()

            evaluator4.reset()
            evaluator4.process(m_pred,t_labe)
            evaluator.process(t_pred,m_labe)
            metrics = evaluator.evaluate()
            metrics4 = evaluator4.evaluate()

            acc_str='t:'+str(round(metrics['map']*100,5))+'_m:'+str(round(metrics4['map']*100,5))       
            save_checkpoint(epoch+1,model.module,acc_str,optimizer,scheduler,time_stamp,'train')
            print('saved')
  

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
        default=2,
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
        "--local-rank",
        type=int,
        default=0,
        help="local rank",
    )
    parser.add_argument(
        "--loss",
        type=int,
        default=1,
        help="speration and reconstruction loss,0 no loss,1 loss",
    )
    parser.add_argument(
        "--mt",
        type=int,
        default=0,
        help="mask type.0 no mask, 1 gpnn mask, 2 all mask",
    )
    parser.add_argument(
            "--smw",
            type=int,
            default=1,
            help="seperation margin weight",
        )
    parser.add_argument(
            "--rmw",
            type=int,
            default=1,
            help="reconstruction margin weight",
    )
    parser.add_argument(
            "--sm",
            type=float,
            default=.2,
            help="seperation margin",
        )
    parser.add_argument(
            "--rm",
            type=float,
            default=.2,
            help="reconstruction margin",
        )
    parser.add_argument(
        "--p_index",
        type=int,
        default=0,
        help="continue path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="continue path",
    )
    parser.add_argument(
        "--normtype",
        type=int,
        default=0,
        help="normtype",
    )
    
    torch.distributed.init_process_group("nccl")
    args = parser.parse_args()
    # test_(args,False)
    # train(args,True
    if args.tp ==0:
        train_oracle(args,False)
    elif args.tp==1:
        print('continue')
        # p='/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250327/1238/20_t:61.87185_c:88.61021_p:58.9198_o1:59.48385.pth'
        p=['/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250421/0010/20_t:60.43938_c:90.17998_p:48.82862.pth']
        train_oracle_continue(args,False,p[args.p_index])
    else:
        raise NotImplementedError
    # train_text2(args) 
    # train_rel_cls_no_mask(args,True)
    # train_rel_cls_sperate(args,True)
    # train_clip_stlt(args)
    