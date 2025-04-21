import sys
sys.path.append('/home/wu_tian_ci/GAFL')

from myutils.extra_model import (GPNNMix3_Test,
                                 PureMix,
                                 GPNNMix4)
from myutils.mydataset import (
                               MixAns2,
                               MixAns3,
                               InfDataset)
import json

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
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
os.environ['OMP_NUM_THREADS'] = '1'

from mytest.draw import draw_list_multi
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from transformers.trainer_pt_utils import SequentialDistributedSampler,distributed_concat
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
import torch.optim as optim
import torch

# 将所有警告转换为异常
# warnings.filterwarnings('error')



def load_model_dict2(model,p1=None,p2=None):
    if p1 is None:
        raise FileNotFoundError
    checkpoint2=torch.load(p1,'cpu')
    model_weight2=checkpoint2['model']
    model.load_state_dict(model_weight2,strict=True)
    if p2 is None:
        return model
    checkpoint=torch.load(p2,'cpu')
    model_weight=checkpoint['model']
    missing_keys, unexpected_keys=model.load_state_dict(model_weight,strict=False)
    # print(model_weight.keys())
    return model

def load_model_dict(p,model,s=True):
    checkpoint=torch.load(p,'cpu')
    model_weight=checkpoint['model']
    model.load_state_dict(model_weight,strict=s)
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
    model=GPNNMix3_Test(config,flag=pretrain,train_stage=args.stage).to(device)
    model=load_model_dict(p,model)
    min_index=0 if args.dt == 2 else 1

    model.eval()
    acc_list=[[],[],[],[]]
    names=['common','private','middle','total']
    xx=[i for i in range(min_index,17)]
    with torch.no_grad():
        for test_i in range(min_index,17):

            # test_dataset=TestMixAns('test',args.dt,test_i,sample_each_clip=16,train=False)
            # test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)
            test_dataset=MixAns2('test',sample_each_clip=16,train=False,mapping_type=2)
            test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)

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
            # acc_list[0].append(metrics2['map']*100)
            # acc_list[1].append(metrics3['map']*100)
            # acc_list[2].append(metrics4['map']*100)
            # acc_list[3].append(metrics['map']*100)
            acc_str=str(test_i)+' c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))\
                    +'_o1:'+str(round(metrics4['map']*100,5))+'_o2:'+str(round(metrics['map']*100,5))
            print(acc_str)
    name='padding' if args.dt==2 else 'not padding'
    draw_list_multi(acc_list,xx,name,names)

    # if pretrain:
    #     acc_str='t:'+str(round(metrics['map']*100,5))+'_c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))
    #     save_checkpoint(epoch+1,model,acc_str,optimizer,scheduler,time_stamp,'pretrain')
    #     print('saved')
    # else:
    #     save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')
 


# 16 x 16 grid
def test_oracle2(args,pretrain,p1,p2):
    config=load_config()
    config.prompt.type=args.prompt
    write_path='/home/wu_tian_ci/GAFL/train/sh_file/write'+str(args.dt)+'1.txt'
    write_ans_path='/home/wu_tian_ci/GAFL/train/sh_file/write_ans'+str(args.dt)+'1.txt'
    # time_stamp=getTimeStamp()
    # loss_record_path='/home/wu_tian_ci/GAFL/recoder/test'
    # t_path=time_stamp[0:8]
    # loss_record_path_=os.path.join(loss_record_path,t_path)
    # if not os.path.exists(loss_record_path_):
    #     os.makedirs(loss_record_path_)
    # loss_record_path=os.path.join(loss_record_path_,time_stamp[8:12]+'ans.txt')
    # if not os.path.exists(loss_record_path):
    #     f=open(loss_record_path,'w')
    #     f.write('begin to write\n')
    #     f.close()

    #     write_config(config,args,[os.path.join(loss_record_path_,time_stamp[8:12]+'config_args.txt')])
    flag=False
    if args.dt==2:
        flag=True
    
    device=args.device
    model=GPNNMix4(config,flag=pretrain,train_stage=args.stage).to(device)
    model=load_model_dict2(model,p1,p2)
    min_index=1 if args.dt>1 else 2

    model.eval()
    # common private total
    acc_list=[[ [ [] for k in range(j+1) ] for j in range(16) ] for i in range(4)]
    # names=['common','private','middle','total']
    # xx=[i for i in range(min_index,17)]
    print('dataset type',args.dt)
    f=open(write_path,'w')
    f2=open(write_ans_path,'w')
    with torch.no_grad():
        for test_i in range(min_index,17):
            test_ii=test_i+1 if args.dt>1 else test_i
            for test_j in range(1,test_ii):
                # test_dataset=TestMixAns('test',args.dt,test_i,sample_each_clip=16,train=False)
                # test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)
                test_dataset=MixAns3('test',sample_each_clip=16,train=False,mapping_type=args.dt,test_i=test_i,test_j=test_j)

                test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)

                evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
                evaluator2 = MyEvaluatorActionGenome(len(test_dataset),158,flag)
                evaluator3 = MyEvaluatorActionGenome(len(test_dataset),158)
                evaluator4 = MyEvaluatorActionGenome(len(test_dataset),157)

                evaluator.reset()
                evaluator2.reset()
                evaluator3.reset()
                evaluator4.reset()

                for batch in test_loader:
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

                    c_ans,p_ans,t_ans,m_ans=model(frames,cls_ids,rel_l,bbx,token_tensor)
        
                    if len(label.shape)==1:
                        label.unsqueeze_(0)
                    if len(common_label):
                        common_label.unsqueeze_(0)
                    if len(private_label):
                        private_label.unsqueeze_(0)

                    evaluator.process(t_ans,label)
       
                    evaluator2.process(c_ans,common_label)
                    evaluator3.process(p_ans,private_label)
                    evaluator4.process(m_ans,label)
                metrics = evaluator.evaluate()
                metrics2 = evaluator2.evaluate()
                metrics3 = evaluator3.evaluate()
                metrics4 = evaluator4.evaluate()

                acc_list[0][test_i-1][test_j-1].append(metrics['map'])
                acc_list[1][test_i-1][test_j-1].append(metrics2['map'])
                acc_list[2][test_i-1][test_j-1].append(metrics3['map'])
                acc_list[3][test_i-1][test_j-1].append(metrics4['map'])

        for i in range(min_index-1,16):
            t_ans_list=[]
            c_ans_list=[]
            p_ans_list=[]
            m_ans_list=[]
            f.write('video:'+str(i)+'\n')
            kk=i+1 if args.dt>1 else i
            for j in range(kk):
                t=np.mean(acc_list[0][i][j])
                t_ans_list.append(t)
                c=np.mean(acc_list[1][i][j])
                c_ans_list.append(c)
                p=np.mean(acc_list[2][i][j])
                p_ans_list.append(p)
                m=np.mean(acc_list[3][i][j])
                m_ans_list.append(m)
            print_acc='video:'+str(i+1)+' '+'total:'+str(round(np.mean(t_ans_list)*100,3))+'%±'+str(round(np.std(t_ans_list)*100,3))+'%\t'\
                        +'common:'+str(round(np.mean(c_ans_list)*100,3))+'%±'+str(round(np.std(c_ans_list)*100,3))+'%\t'\
                        +'private:'+str(round(np.mean(p_ans_list)*100,3))+'%±'+str(round(np.std(p_ans_list)*100,3))+'%\t'\
                        +'middle:'+str(round(np.mean(m_ans_list)*100,3))+'%±'+str(round(np.std(m_ans_list)*100,5))+'%\t\n'
            f.write(print_acc)
            f2.write(print_acc)
            f.write(str(acc_list[0][i])+'\n')
            f.write(str(acc_list[1][i])+'\n')
            f.write(str(acc_list[2][i])+'\n')
            f.write(str(acc_list[3][i])+'\n')
            print(print_acc)
    f.close()
    f2.close()

                
    # name='padding' if args.dt==2 else 'not padding'


    # draw_list_multi(acc_list,xx,name,names)

    # if pretrain:
    #     acc_str='t:'+str(round(metrics['map']*100,5))+'_c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))
    #     save_checkpoint(epoch+1,model,acc_str,optimizer,scheduler,time_stamp,'pretrain')
    #     print('saved')
    # else:
    #     save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')

# type 5
# dt all rignt, all wrong, padding, all
# model: 10 single,11 dual
# backbone only
def test_oracle4(args,pretrain,p1,p2):
    config=load_config()
    config.prompt.type=args.prompt

    device=args.device
    model=GPNNMix4(config,flag=pretrain,train_stage=args.stage).to(device)
    model=load_model_dict2(model,p1,p2)
    min_index=1

    model.eval()
    # common private total
    acc_list=[ [ [] for k in range(j+1) ] for j in range(16) ] 
    # names=['common','private','middle','total']
    # xx=[i for i in range(min_index,17)]
    print('dataset type',args.dt)
    with torch.no_grad():
        # 
        for test_i in range(min_index,17):
            for test_j in range(1,test_i+1):
                # test_dataset=TestMixAns('test',args.dt,test_i,sample_each_clip=16,train=False)
                # test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)
                test_dataset=MixAns3('test',sample_each_clip=16,train=False,mapping_type=args.dt,test_i=test_i,test_j=test_j)

                test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12)

                evaluator = MyEvaluatorActionGenome(len(test_dataset),157)

                evaluator.reset()


                for batch in test_loader:
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

                    t_ans=model(frames,cls_ids,rel_l,bbx,token_tensor)
        
                    if len(label.shape)==1:
                        label.unsqueeze_(0)
                    if len(common_label):
                        common_label.unsqueeze_(0)
                    if len(private_label):
                        private_label.unsqueeze_(0)

                    evaluator.process(t_ans,label)

                metrics = evaluator.evaluate()


                acc_list[test_i-1][test_j-1].append(metrics['map'])


        for i in range(16):
            t_ans_list=[]

            f.write('video:'+str(i)+'\n')
            for j in range(i+1):
                t=np.mean(acc_list[0][i][j])
                t_ans_list.append(t)

            print_acc='video:'+str(i+1)+' '+'total:'+str(round(np.mean(t_ans_list)*100,3))+'%±'+str(round(np.std(t_ans_list)*100,3))+'%'

            print(print_acc)

                
    # name='padding' if args.dt==2 else 'not padding'


    # draw_list_multi(acc_list,xx,name,names)

    # if pretrain:
    #     acc_str='t:'+str(round(metrics['map']*100,5))+'_c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))
    #     save_checkpoint(epoch+1,model,acc_str,optimizer,scheduler,time_stamp,'pretrain')
    #     print('saved')
    # else:
    #     save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')
 
def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)




def test_oracle5(args,pretrain,p1,p2):
    config=load_config()
    config.prompt.type=args.prompt

    test_dataset=InfDataset('test',sample_each_clip=16,train=False,mapping_type=args.dt)
    testts=torch.utils.data.distributed.DistributedSampler(test_dataset,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12,
                           sampler=testts)

    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK']=str(args.local_rank)
    local_rank=args.local_rank
    device = torch.device(local_rank)
    model=GPNNMix4(config,flag=pretrain,train_stage=args.stage).to(device)

    model=load_model_dict2(model,p1,p2)
    model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank,find_unused_parameters=False)
    
    # total common private middle
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    evaluator4 = MyEvaluatorActionGenome(len(test_dataset),157)
    evaluator2 = MyEvaluatorActionGenome(len(test_dataset),158)
    evaluator3 = MyEvaluatorActionGenome(len(test_dataset),158)


    model.eval()

    t_pre,t_lab=[],[]
    m_pre,m_lab=[],[]
    c_pre,c_lab=[],[]
    p_pre,p_lab=[],[]
    idxes=[]
    with torch.no_grad():
        for batch in tqdm(test_loader):
            frames,bbx,mask,label,cls_ids,cls_l,rel_l,private_label,common_label,token_tensor,mask_,idx=batch
            frames=frames.to(device)
            idx=idx.to(device)
            idxes.append(idx)
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
            c_ans,p_ans,t_ans,m_ans1=model(frames,cls_ids,rel_l,bbx,token_tensor)

            t_pre.append(t_ans)
            t_lab.append(label)
            m_pre.append(m_ans1)
            m_lab.append(label)
            c_pre.append(c_ans)
            c_lab.append(common_label)
            p_pre.append(p_ans)
            p_lab.append(private_label)

    t_pred = distributed_concat(torch.concat(t_pre, dim=0),len(testts.dataset))
    t_labe = distributed_concat(torch.concat(t_lab, dim=0),len(testts.dataset))
    m_pred = distributed_concat(torch.concat(m_pre, dim=0),len(testts.dataset))
    m_labe = distributed_concat(torch.concat(m_lab, dim=0),len(testts.dataset))
    c_pred = distributed_concat(torch.concat(c_pre, dim=0),len(testts.dataset))
    c_labe = distributed_concat(torch.concat(c_lab, dim=0),len(testts.dataset))
    p_pred = distributed_concat(torch.concat(p_pre, dim=0),len(testts.dataset))
    p_labe = distributed_concat(torch.concat(p_lab, dim=0),len(testts.dataset))
    idxes_=distributed_concat(torch.concat(idxes, dim=0),len(testts.dataset))

    if local_rank==0:
        '''
        ctp='/home/wu_tian_ci/GAFL/mytest/code_test'
        js_p=os.path.join(ctp,'multi.json')
        sa_dict={}
        for ind in range(idxes_.shape[0]):
            sa_dict[idxes_[ind].detach().item()]=[
                t_pred[ind].cpu().detach().tolist(),
                t_labe[ind].cpu().detach().tolist(),
                m_pred[ind].cpu().detach().tolist(),
                m_labe[ind].cpu().detach().tolist(),
                c_pred[ind].cpu().detach().tolist(),
                c_labe[ind].cpu().detach().tolist(),
                p_pred[ind].cpu().detach().tolist(),
                p_labe[ind].cpu().detach().tolist()
            ]
        json.dump(sa_dict,open(js_p,'w'))
        '''
        evaluator.reset()
        evaluator4.reset()
        evaluator2.reset()
        evaluator3.reset()
        evaluator4.process(m_pred,t_labe)
        evaluator.process(t_pred,m_labe)
        evaluator2.process(c_pred,c_labe)
        evaluator3.process(p_pred,p_labe)
        print(m_pred.shape,t_labe.shape)
        print(evaluator.index)
        print(evaluator2.index)
        print(evaluator3.index)
        print(evaluator4.index)
        metrics = evaluator.evaluate()
        metrics4 = evaluator4.evaluate()
        metrics2 = evaluator2.evaluate()
        metrics3 = evaluator3.evaluate()

        acc_str='t:'+str(round(metrics['map']*100,5))+'_m:'+str(round(metrics4['map']*100,5))+'c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))              
        print(acc_str)

def test_oracle3(args,pretrain,p1,p2):
    config=load_config()
    time_stamp=getTimeStamp()
    config.prompt.type=args.prompt
    write_path=os.path.join('/home/wu_tian_ci/GAFL/recoder/inference/detail',time_stamp[:8])
    write_ans_path=os.path.join('/home/wu_tian_ci/GAFL/recoder/inference/ans',time_stamp[:8])
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    if not os.path.exists(write_ans_path):
        os.makedirs(write_ans_path)
    write_path=os.path.join(write_path,time_stamp[8:12]+'.txt')
    write_ans_path=os.path.join(write_ans_path,time_stamp[8:12]+'.txt')

    device=args.device
    model=GPNNMix4(config,flag=pretrain,train_stage=args.stage).to(device)
    model=load_model_dict2(model,p1)

    model.eval()
    # common private total
    acc_list=[]

    # names=['common','private','middle','total']
    # xx=[i for i in range(min_index,17)]
    print('dataset type',args.dt)
    f=open(write_path,'w')
    f2=open(write_ans_path,'w')
    t_pre,t_lab=[],[]
    m_pre,m_lab=[],[]
    c_pre,c_lab=[],[]
    p_pre,p_lab=[],[]
    idxes=[]

    with torch.no_grad():

        test_dataset=InfDataset('test',sample_each_clip=16,train=False,mapping_type=args.dt)
        # test_dataset=MixAns2('test',sample_each_clip=16,train=False,mapping_type=2)
        print(len(test_dataset))
        test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=12,drop_last=False,shuffle=False)

        evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
        evaluator2 = MyEvaluatorActionGenome(len(test_dataset),158)
        evaluator3 = MyEvaluatorActionGenome(len(test_dataset),158)
        evaluator4 = MyEvaluatorActionGenome(len(test_dataset),157)

        evaluator.reset()
        evaluator2.reset()
        evaluator3.reset()
        evaluator4.reset()

        for batch in tqdm(test_loader):
            frames,bbx,mask,label,cls_ids,cls_l,rel_l,private_label,common_label,token_tensor,mask_,idx=batch
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

            c_ans,p_ans,t_ans,m_ans=model(frames,cls_ids,rel_l,bbx,token_tensor)

            if len(label.shape)==1:
                label.unsqueeze_(0)
            if len(common_label)==1:
                common_label.unsqueeze_(0)
            if len(private_label)==1:
                private_label.unsqueeze_(0)

            assert t_ans.shape==label.shape
            assert m_ans.shape==label.shape
            t_pre.append(t_ans)
            t_lab.append(label)
            m_pre.append(m_ans)
            m_lab.append(label)
            c_pre.append(c_ans)
            c_lab.append(common_label)
            p_pre.append(p_ans)
            p_lab.append(private_label)
            idx=idx.to(device)
            idxes.append(idx)
    t_pred=torch.concat(t_pre,dim=0)
    t_labe=torch.concat(t_lab,dim=0)

    m_pred=torch.concat(m_pre,dim=0)
    m_labe=torch.concat(m_lab,dim=0)

    c_pred=torch.concat(c_pre,dim=0)
    c_labe=torch.concat(c_lab,dim=0)

    p_pred=torch.concat(p_pre,dim=0)
    p_labe=torch.concat(p_lab,dim=0)
    idxes_=torch.concat(idxes,dim=0)
    '''
    ctp='/home/wu_tian_ci/GAFL/mytest/code_test'
    js_p=os.path.join(ctp,'single.json')
    sa_dict={}
    for ind in range(idxes_.shape[0]):
        sa_dict[idxes_[ind].detach().item()]=[
            t_pred[ind].cpu().detach().tolist(),
            t_labe[ind].cpu().detach().tolist(),
            m_pred[ind].cpu().detach().tolist(),
            m_labe[ind].cpu().detach().tolist(),
            c_pred[ind].cpu().detach().tolist(),
            c_labe[ind].cpu().detach().tolist(),
            p_pred[ind].cpu().detach().tolist(),
            p_labe[ind].cpu().detach().tolist()
        ]
    json.dump(sa_dict,open(js_p,'w'))
    '''
    evaluator.process(t_pred,t_labe)
    evaluator2.process(c_pred,c_labe)
    evaluator3.process(p_pred,p_labe)
    evaluator4.process(m_pred,m_labe)
    metrics = evaluator.evaluate()
    metrics2 = evaluator2.evaluate()
    metrics3 = evaluator3.evaluate()
    metrics4 = evaluator4.evaluate()

    acc_list.append(metrics['map'])
    acc_list.append(metrics2['map'])
    acc_list.append(metrics3['map'])
    acc_list.append(metrics4['map'])

    print_acc='video: all '+' '+'total:'+str(round(acc_list[0]*100,5))+'%\t'\
                +'common:'+str(round(acc_list[1]*100,5))+'%\t'\
                +'private:'+str(round(acc_list[2]*100,5))+'%\t'\
                +'middle:'+str(round(acc_list[3]*100,5))+'%\t\n'
    print(print_acc)
    f.write(print_acc)
    f2.write(print_acc)
    f.write(str(acc_list[0])+'\n')
    f.write(str(acc_list[1])+'\n')
    f.write(str(acc_list[2])+'\n')
    f.write(str(acc_list[3])+'\n')

    f.close()
    f2.close()

                
    # name='padding' if args.dt==2 else 'not padding'


    # draw_list_multi(acc_list,xx,name,names)

    # if pretrain:
    #     acc_str='t:'+str(round(metrics['map']*100,5))+'_c:'+str(round(metrics2['map']*100,5))+'_p:'+str(round(metrics3['map']*100,5))
    #     save_checkpoint(epoch+1,model,acc_str,optimizer,scheduler,time_stamp,'pretrain')
    #     print('saved')
    # else:
    #     save_checkpoint(epoch+1,model,round(metrics['map']*100,5),optimizer,scheduler,time_stamp,'train')
 


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
    parser.add_argument(
        "--p_index",
        type=int,
        default=0,
        help="load path",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="local rank",
    )
    set_seed(seed=3407)
    args = parser.parse_args()
    # train(args,True
    # p='/home/wu_tian_ci/GAFL/recoder/checkpoint/train/20250324/2211/1_t:68.13982m:64.26654.pth'
    p1=['/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250416/1449/20_t:59.87919_c:90.81566_p:49.80294.pth',
           '/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250415/2037/20_t:59.42587_c:91.1116_p:47.09343.pth',
           '/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250416/2052/20_t:59.44598_c:91.46358_p:53.10185.pth',
           '/home/wu_tian_ci/GAFL/recoder/checkpoint/pretrain/20250420/1044/18_t:60.8355_c:90.10944_p:50.15488.pth']
    p2=['/home/wu_tian_ci/GAFL/recoder/checkpoint/train/20250417/1929/5_t:71.05293_m:59.90562.pth',
        '/home/wu_tian_ci/GAFL/recoder/checkpoint/train/20250417/2051/5_t:73.07588_m:59.84188.pth',
        '',
        '']
    if args.tp ==0:
        test_oracle(args,False,p1)
    elif args.tp==1:
        test_oracle2(args,False,p1,p2)
    elif args.tp==2:
        test_oracle3(args,False,p1[args.p_index],p2[args.p_index])
    elif args.tp==3:
        torch.distributed.init_process_group("nccl")
        test_oracle5(args,False,p1[args.p_index],p2[args.p_index])
    else:
        raise NotImplementedError
    # train_text2(args) 
    # train_rel_cls_no_mask(args,True)
    # train_rel_cls_sperate(args,True)
    # train_clip_stlt(args)
    