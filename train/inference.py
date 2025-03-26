import sys
sys.path.append('/home/wu_tian_ci/GAFL')
import torch.optim as optim
import torch
from myutils.extra_model import (GPNNMix3,
                                 PureMix)
from myutils.mydataset import (
                               CLIPFeatureDatasetCLSRELOracle,
                               MixAns2,
                               TestMixAns)
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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from transformers.trainer_pt_utils import SequentialDistributedSampler,distributed_concat


import warnings

# 将所有警告转换为异常
# warnings.filterwarnings('error')



def load_model_dict(p,model):

    p2='/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250319/1503/20_c:88.3695_p:58.54179_o:59.57755.pth'
    checkpoint2=torch.load(p2,'cpu')
    model_weight2=checkpoint2['model']
    model.load_state_dict(model_weight2,strict=False)
    checkpoint=torch.load(p,'cpu')
    model_weight=checkpoint['model']
    model.load_state_dict(model_weight,strict=False)
    return model



def test_oracle(args,pretrain,p):
    config=load_config()
    config.prompt.type=args.prompt
    
    time_stamp=getTimeStamp()
    loss_record_path='/home/wu_tian_ci/GAFL/recoder/test'
    t_path=time_stamp[0:8]
    loss_record_path_=os.path.join(loss_record_path,t_path)
    if not os.path.exists(loss_record_path_):
        os.makedirs(loss_record_path_)
    loss_record_path=os.path.join(loss_record_path_,time_stamp[8:12]+'ans.txt')
    if not os.path.exists(loss_record_path):
        f=open(loss_record_path,'w')
        f.write('begin to write\n')
        f.close()

        write_config(config,args,[os.path.join(loss_record_path_,time_stamp[8:12]+'config_args.txt')])
   
    
    device=args.device
    model=GPNNMix3(config,flag=pretrain,train_stage=args.stage).to(device)
    model=load_model_dict(p,model)

    model.eval()
    with torch.no_grad():
        for test_i in range(1,17):

            test_dataset=TestMixAns('test',args.dt,test_i,sample_each_clip=16,train=False)
            test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)
            # test_dataset=MixAns2('test',sample_each_clip=16,train=False,mapping_type=3)
            # test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)

            evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
            evaluator2 = MyEvaluatorActionGenome(len(test_dataset),158)
            evaluator3 = MyEvaluatorActionGenome(len(test_dataset),158)
            evaluator4 = MyEvaluatorActionGenome(len(test_dataset),157)

            evaluator.reset()
            evaluator2.reset()
            evaluator3.reset()
            evaluator4.reset()
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

                c_ans,p_ans,m_ans1,t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor)
    
                if len(label.shape)==1:
                    label.unsqueeze_(0)
                if len(common_label):
                    common_label.unsqueeze_(0)
                if len(private_label):
                    private_label.unsqueeze_(0)

                evaluator.process(t_ans,label)
                evaluator2.process(c_ans,common_label)
                evaluator3.process(p_ans,private_label)
                evaluator4.process(m_ans1,label)
            metrics = evaluator.evaluate()
            metrics2 = evaluator2.evaluate()
            metrics3 = evaluator3.evaluate()
            metrics4 = evaluator4.evaluate()
            # metrics4_1 = evaluator4_1.evaluate()
        # total_ans acc   common_ans acc

            acc_str=str(test_i)+' c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))\
                    +'_o1:'+str(round(metrics4['map']*100,5))+'_o2:'+str(round(metrics['map']*100,5))
            print(acc_str)
    

    # if pretrain:
    #     acc_str='t:'+str(round(metrics['map']*100,5))+'_c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))
    #     save_checkpoint(epoch+1,model,acc_str,optimizer,scheduler,time_stamp,'pretrain')
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

    
    print('saved')

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 

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
        "--batchsize",
        type=int,
        default=32,
        help="batchsize",
    )

    parser.add_argument(
        "--ds",
        type=int,
        default=2,
        help="dataset",
    )

    parser.add_argument(
        "--prompt",
        type=int,
        default=1,
        help="prompt type 0:smiple 1:gpfp",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=3,
        help="train stage",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=0,
        help="train type 0:oracle 1:oracle continue 2:pure",
    )
    parser.add_argument(
        "--dt",
        type=int,
        default=1,
        help="dt type 1:no padding 1  2:padding",
    )
    set_seed(seed=3407)
    args = parser.parse_args()
    # train(args,True
    p='/home/wu_tian_ci/GAFL/recoder/checkpoint/train/20250324/2211/1_t:68.13982m:64.26654.pth'
    if args.tp==2:
        train_pure(args,False)
    elif args.tp ==0:
        test_oracle(args,False,p)
    else:
        raise NotImplementedError
    # train_text2(args) 
    # train_rel_cls_no_mask(args,True)
    # train_rel_cls_sperate(args,True)
    # train_clip_stlt(args)
    