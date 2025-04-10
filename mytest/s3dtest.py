import sys
sys.path.append('/home/wu_tian_ci/GAFL')
import torch.optim as optim
import torch

from myutils.mydataset import *
import torch.nn.functional as F
from myutils.s3d import *
from tqdm import tqdm

from myutils.config import *
from tqdm import tqdm
import argparse
import random
from torch import nn as nn
from myutils.data_utils import (
    add_weight_decay,
    get_linear_schedule_with_warmup,
    save_checkpoint,
    getTimeStamp,
    MyEvaluatorActionGenome,
    Criterion
)


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
@torch.no_grad()
def accuracy2(test_loader,device,model):
    model.eval()
    rights=[]
    totals=[]
    for it in tqdm(test_loader):
        frames,label=it
        frames=frames.to(device)
        label=label.to(device).squeeze()
        totals.append(label.shape[0])
        ans=model(frames)
        ans=torch.argmax(F.softmax(ans,dim=-1),dim=-1)
        eq_elements=torch.eq(ans,label)
        rights.append(torch.sum(eq_elements).item())
       
    return sum(rights)/sum(totals)

def load_model(num_class=157):
    model=S3D(num_class)
    weight_path='/home/wu_tian_ci/GAFL/model_weight/s3d/S3D_kinetics400.pt'
    weight_dict = torch.load(weight_path)
    model_dict = model.state_dict()
    for name, param in weight_dict.items():
        if 'module' in name:
            name = '.'.join(name.split('.')[1:])
        if name in model_dict:
            if param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                print (' size? ' + name, param.size(), model_dict[name].size())
        else:
            print (' name? ' + name)
    
    return model



def train(args,pretrain):
    # test_dataset=ImageDatsetALL('test',sample_each_clip=16)
    test_dataset=AppearanceDataset('test',True)
    test_loader=DataLoader(test_dataset,batch_size=args.batchsize*4,num_workers=16)


    # dataset=ImageDatsetALL('train',sample_each_clip=16)
    dataset=AppearanceDataset('train',True)
    train_loader=DataLoader(dataset,batch_size=args.batchsize,num_workers=12,shuffle=True)
    device=args.device
    # model=MyModel(config,pretrain=pretrain).to(args.device)
    model=load_model()
    loss=Criterion()

    parameters = add_weight_decay(model, args.decay)
    optimizer = optim.AdamW(parameters, lr=args.lr)
    num_batches = len(train_loader) // args.batchsize
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup * num_batches,
        num_training_steps=args.epoch * num_batches,
    )
    model.to(device)
    evaluator = MyEvaluatorActionGenome(len(test_dataset),157)
    for epoch in range(args.epoch):
        model.train()
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                frames,label=batch
                frames=frames.to(device)
                label=label.to(device)
                ans=model(frames)
                l=loss(ans,label)
                optimizer.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                pbar.set_postfix({"Loss": l.item()})

        evaluator.reset()
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                frames,label=batch
                frames=frames.to(device)
                label=label.to(device)
                ans=model(frames)
                evaluator.process(ans,label)
            metrics = evaluator.evaluate()
        print("acc:", round(metrics['map'],5))

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Packs PIL images as HDF5.")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
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
        default=4,
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
    