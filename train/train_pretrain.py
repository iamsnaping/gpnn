import sys
sys.path.append('/home/wu_tian_ci/GAFL')
import torch.optim as optim
import torch
from myutils.extra_model import MyModel,MyModel_T,PureT,TextT,MyModel_CLS
from myutils.mydataset import *
import torch.nn.functional as F
from myutils.config import *
from tqdm import tqdm
import argparse
import random
from torch import nn as nn
from myutils.data_utils import (
    add_weight_decay,
    get_linear_schedule_with_warmup,
    save_checkpoint,
    getTimeStamp
)


def weight_init_dis(m):
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
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

@torch.no_grad()
def accuracy(test_loader,device,model):
    model.eval()
    rights=[]
    totals=[]
    for it in tqdm(test_loader):
        frames,bbx,mask,label=it
        frames=frames.to(device)
        bbx=bbx.to(device)
        mask=mask.to(device)
        label=label.to(device).squeeze()
        totals.append(label.shape[0])
        ans=model(frames,mask,bbx)
        ans=torch.argmax(F.softmax(ans,dim=-1),dim=-1)
        eq_elements=torch.eq(ans,label)
        rights.append(torch.sum(eq_elements).item())
       
    return sum(rights)/sum(totals)



def train(args,pretrain):
    config=load_config()
    dataset=TextDataset('test')
    test_loader=DataLoader(dataset,batch_size=256,num_workers=8)


    dataset=TextDataset('train')
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=8,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    model=MyModel_CLS(config,pretrain=pretrain).to(args.device)
    model.apply(weight_init_dis)
    loss=torch.nn.CrossEntropyLoss()
    max_acc=999
    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(dataset) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )

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
                l=loss(ans,label)
                optimizer.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                pbar.set_postfix({"Loss": l.item()})
                f=open(loss_record_path,'a')
                f.write('epoch: '+str(epoch)+' K:'+str(txt_k)+' loss:'+str(round(l.item(),5))+ '\n')
                f.close()
                txt_k+=1
        acc_=accuracy(test_loader,device,model)
        if pretrain:
            save_checkpoint(epoch+1,model,round(acc_*100,5),optimizer,scheduler,time_stamp,'pretrain')
            print('saved')
        else:
            save_checkpoint(epoch+1,model,round(acc_*100,5),optimizer,scheduler,time_stamp,'train')


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Packs PIL images as HDF5.")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2",
        help="gpu device",
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
        default=2,
        help="warmup epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
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

    args = parser.parse_args()
    train(args,True)
    