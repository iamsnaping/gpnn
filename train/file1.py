import sys
sys.path.append('/home/wu_tian_ci/GAFL')
import torch.optim as optim
import torch
from myutils.extra_model import MyModel,MyModel_T,PureT,TextT,MyModel_CLS,MyModelGCN,TextT2,Stlt
from myutils.mydataset import *
import torch.nn.functional as F
from myutils.config import *
from tqdm import tqdm
import argparse
import random
import logging
from torch import nn as nn
from myutils.data_utils import (
    add_weight_decay,
    get_linear_schedule_with_warmup,
    save_checkpoint,
    getTimeStamp,
    MyEvaluatorActionGenome,
    Criterion
)

def train_2(args):
    config=load_config()
    device=args.device
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
    model=Stlt(config)
    model=model.to(device)

    criterion = Criterion()
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
        txt_k=0
        with tqdm(total=len(train_loader)) as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                video_id,cate,box,scor,ft,length,label=batch
                cate=cate.to(device)
                box=box.to(device)
                label=label.to(device).squeeze()
                ans=model(cate,box)
                loss = criterion(ans,label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})
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
                evaluator.process(ans, label)
        metrics = evaluator.evaluate()
        if evaluator.is_best():
            logging.info("=================================")
            logging.info(f"Found new best on epoch {epoch+1}!")
            logging.info("=================================")
        for m in metrics.keys():
            logging.info(f"{m}: {round(metrics[m] * 100, 2)}")
            print(epoch,m,round(metrics[m] * 100, 2))

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
        default=32,
        help="warmup epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=7e-5,
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
    train_2(args)
    