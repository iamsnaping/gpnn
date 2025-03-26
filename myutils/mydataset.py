import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from torch.utils.data import DataLoader
from PIL import Image
import sys
import io
sys.path.append('/home/wu_tian_ci/GAFL')
from myutils.data_utils import (sample_appearance_indices,
                                VideoColorJitter,
                                IdentityTransform,
                                sample_train_layout_indices,
                                get_test_layout_indices,
                                fix_box)
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    Resize,
    ToTensor,
)
from configs.configs import DataConfig
from torchvision.transforms import functional as TF
import math
from natsort import natsorted
import re
class CLIPFeatureDataset(Dataset):
    def __init__(self,name,sample_each_clip=8,train=True):
        super().__init__()
        self.sample_rate=sample_each_clip
        self.name=name
        self.train=train
        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset",name+'.json'),'r')
        )
        self.keys=list(self.json.keys())
        self.video_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5',name+'.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
    
    def __len__(self):
        return len(self.keys)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        key=self.keys[idx]
        # key -> cls_video_id
        # frame feature videos[key][value[0~]]
        # bbx videos[key][value[0~]bbx]
        # mask list videos[key][value[0~]mask]
        frame_ids=self.json[key]
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
        video_size=self.video2size[key.split('_')[1].split('.')[0]]
        frames=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]]),dtype=np.float16)).reshape(1,10,512) for index in indices]
        bbx=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]+'bbx']),dtype=np.float64)).reshape(1,10,4) for index in indices]
        mask=[np.frombuffer(np.array(self.videos[key][frame_ids[index]+'mask']),dtype=np.int64) for index in indices]
        mask_=[]
        # print(mask)
        extra_zero=np.array([0] * (10 - len(mask[0])))
        for m in mask:
            _=torch.from_numpy(np.concatenate([m,extra_zero])).unsqueeze(0)
            mask_.append(_)
        label=torch.tensor(int(key.split('_')[0][1:]),dtype=torch.long).unsqueeze(0)
        frames=torch.concat(frames,dim=0).float()
        bbx=torch.concat(bbx,dim=0).float()
        bbx[:,:,0]/=video_size[0]
        bbx[:,:,1]/=video_size[1]
        bbx[:,:,2]/=video_size[0]
        bbx[:,:,3]/=video_size[1]
        mask=torch.concat(mask_,dim=0).long()
        mask_tensor_expanded = mask.bool().unsqueeze(-1).expand(-1, -1, 512)
        frames[~mask_tensor_expanded]=0.

        return frames,bbx,mask,label


class TextDataset(Dataset):
    def __init__(self,name,sample_each_clip=16,train=True):
        super().__init__()
        self.sample_rate=sample_each_clip
        self.name=name
        self.train=train
        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset",name+'.json'),'r')
        )
        self.keys=list(self.json.keys())
        self.video_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5',name+'.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
    
    def __len__(self):
        return len(self.keys)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        key=self.keys[idx]
        # key -> cls_video_id
        # frame feature videos[key][value[0~]]
        # bbx videos[key][value[0~]bbx]
        # mask list videos[key][value[0~]mask]
        frame_ids=self.json[key]
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
        video_size=self.video2size[key.split('_')[1].split('.')[0]]
        frames=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]+'obj_cls']),dtype=np.int16)).reshape(1,10,1) for index in indices]
        bbx=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]+'bbx']),dtype=np.float64)).reshape(1,10,4) for index in indices]
        mask=[np.frombuffer(np.array(self.videos[key][frame_ids[index]+'mask']),dtype=np.int64) for index in indices]
        mask_=[]
        # print(mask)
        extra_zero=np.array([0] * (10 - len(mask[0])))
        for m in mask:
            _=torch.from_numpy(np.concatenate([m,extra_zero])).unsqueeze(0)
            mask_.append(_)
        label=torch.tensor(int(key.split('_')[0][1:]),dtype=torch.long).unsqueeze(0)
        frames=torch.concat(frames,dim=0).long()
        bbx=torch.concat(bbx,dim=0).float()
        bbx[:,:,0]/=video_size[0]
        bbx[:,:,1]/=video_size[1]
        bbx[:,:,2]/=video_size[0]
        bbx[:,:,3]/=video_size[1]
        mask=torch.concat(mask_,dim=0).long()
        # mask_tensor_expanded = mask.bool().unsqueeze(-1).expand(-1, -1, 512)
        # frames[~mask_tensor_expanded]=0.

        return frames,bbx,mask,label

class ImageDatset(Dataset):
    def __init__(self,name,sample_each_clip=8,train=True):
        super().__init__()
        self.sample_rate=sample_each_clip
        self.name=name
        self.train=train
        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset",name+'.json'),'r')
        )
        self.resize = Resize((math.floor(256 * 1.15),math.floor(384*1.15)))
        self.transforms = Compose(
            [
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.keys=list(self.json.keys())
        self.video_path=os.path.join('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/data_save/hdf5file',name+'_6.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
    
    def __len__(self):
        return len(self.keys)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        key=self.keys[idx]
        # key -> cls_video_id
        # frame feature videos[key][value[0~]]
        # bbx videos[key][value[0~]bbx]
        # mask list videos[key][value[0~]mask]
        frame_ids=self.json[key]
        video_id=key.split('_')[1].split('.')[0]
        label=torch.tensor(int(key.split('_')[0][1:]),dtype=torch.long).unsqueeze(0)
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
   

        raw_video_frames = [
            self.resize(
                Image.open(io.BytesIO(np.array(self.videos[video_id][frame_ids[index].split('.')[0]])))
            )
            for index in indices
        ]
        augment = IdentityTransform()
        if self.train:
            augment = VideoColorJitter()
            top, left, height, width = RandomCrop.get_params(
                raw_video_frames[0],
                (256, 384),
            )

        video_frames = []
        for i in range(len(raw_video_frames)):
            frame = raw_video_frames[i]
            frame = augment(frame)
            frame = (
                TF.crop(frame, top, left, height, width)
                if self.train
                else TF.center_crop(frame, (256,384))
            )
            frame = self.transforms(frame)
            video_frames.append(frame)

        video_frames = torch.stack(video_frames, dim=0).transpose(0,1)
        return video_frames,label

class CLIPFeatureDatasetAll(Dataset):
    def __init__(self,name,sample_each_clip=16,train=True):
        super().__init__()
        self.sample_rate=sample_each_clip
        self.name=name
        self.train=train
        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all",name+'.json'),'r')
        )
        self.mask=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all",name+'_mask.json'),'r')
        )
        self.bbx=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all",name+'_bbx.json'),'r')
        )
        keys=list(self.json.keys())
        self.keys=[item for item in keys if 'label' not in item]
        self.label=[i for i in keys if 'label' in i]
        self.video_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all',name+'.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
        self.num_cls=157
    
    def __len__(self):
        return len(self.keys)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        key=self.keys[idx]
        # key -> cls_video_id
        # frame feature videos[key][value[0~]]
        # bbx videos[key][value[0~]bbx]
        # mask list videos[key][value[0~]mask]
        frame_ids=self.json[key]
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
        video_size=self.video2size[key.split('.')[0]]
        frames=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]]),dtype=np.float16)).reshape(1,10,512) for index in indices]

        bbx=torch.tensor([self.bbx[key][index] for index in indices],dtype=torch.float32)
        mask=np.array([self.mask[key][index] for index in indices],dtype=np.int64)

        mask_=[]
        # print(mask)
        extra_zero=np.array([0] * (10 - len(mask[0])))
        for m in mask:
            _=torch.from_numpy(np.concatenate([m,extra_zero])).unsqueeze(0)
            mask_.append(_)
        label_idx=self.json[self.label[idx]]
        label_idx=[int(x) for x in label_idx]
        label=torch.zeros(self.num_cls,dtype=torch.float32)
        frames=torch.concat(frames,dim=0).float()
        label[label_idx]=1.0
        bbx[:,:,0]/=video_size[0]
        bbx[:,:,1]/=video_size[1]
        bbx[:,:,2]/=video_size[0]
        bbx[:,:,3]/=video_size[1]
        mask=torch.concat(mask_,dim=0).long()
        mask_tensor_expanded = mask.bool().unsqueeze(-1).expand(-1, -1, 512)
        frames[~mask_tensor_expanded]=0.

        return frames,bbx,mask,label

class CLIPFeatureDatasetCLSREL(Dataset):
    def __init__(self,name,sample_each_clip=16,node_nums=10,train=True):
        super().__init__()
        self.sample_rate=sample_each_clip
        self.node_nums=node_nums
        self.name=name
        self.train=train
        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
        self.mask=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_mask.json'),'r')
        )
        self.bbx=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_bbx.json'),'r')
        )
        self.cls=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_obj_cls.json'),'r')
        )
        self.rel=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_rel.json'),'r')
        )
        keys=list(self.json.keys())
        self.keys=[item for item in keys if 'label' not in item]
        self.label=[i for i in keys if 'label' in i]
        self.video_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all_cls_rel',name+'.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
        self.num_cls=157
        self.obj_cls_num=38
        self.rel_num=30
    
    def __len__(self):
        return len(self.keys)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        key=self.keys[idx]
        # key -> cls_video_id
        # frame feature videos[key][value[0~]]
        # bbx videos[key][value[0~]bbx]
        # mask list videos[key][value[0~]mask]
        frame_ids=self.json[key]
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
        video_size=self.video2size[key.split('.')[0]]
        frames=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]]),dtype=np.float16)).reshape(1,11,512) for index in indices]

        bbx=torch.tensor([self.bbx[key][index] for index in indices],dtype=torch.float32)
        # mask=np.array([self.mask[key][index] for index in indices],dtype=np.int64)
        mask=torch.tensor([self.mask[key][index] for index in indices],dtype=torch.long)
        cls_ids=torch.tensor([self.cls[key][index] for index in indices],dtype=torch.long)
        rel_ids=[self.rel[key][index] for index in indices]
        mask_=[]
        # print(mask)
        
        # extra_zero=np.array([0] * (10 - len(mask[0])))
        # for m in mask:
        #     _=torch.from_numpy(np.concatenate([m,extra_zero])).unsqueeze(0)
        #     mask_.append(_)
        label_idx=self.json[self.label[idx]]
        label_idx=[int(x) for x in label_idx]
        label=torch.zeros(self.num_cls,dtype=torch.float32)
        frames=torch.concat(frames,dim=0).float()
        label[label_idx]=1.0
        bbx[:,:,0]/=video_size[0]
        bbx[:,:,1]/=video_size[1]
        bbx[:,:,2]/=video_size[0]
        bbx[:,:,3]/=video_size[1]
        zero_tensor=torch.tensor([[0.,0.,1.,1.]]).to(bbx)
        zero_tensor=zero_tensor.unsqueeze(0).repeat(16,1,1)
        bbx=torch.cat([zero_tensor,bbx],dim=-2)
        # mask=torch.concat(mask_,dim=0).long()

        # mask=torch.concat(mask,dim=0).long()
        # mask_tensor_expanded = mask.bool().unsqueeze(-1).expand(-1, -1, 512)
        # frames[~mask_tensor_expanded]=0.
        rel=torch.zeros((self.sample_rate,self.node_nums,self.rel_num),dtype=torch.float32)
        cls=torch.zeros((self.sample_rate,self.node_nums,self.obj_cls_num),dtype=torch.float32)
        cls.scatter_(2,cls_ids.unsqueeze(-1),1.)
        for i in range(self.sample_rate):
            for j in range(self.node_nums):
                rel[i][j][rel_ids[i][j]]=1.
        # breakpoint()
        # cls=torch.zeros(self.obj_cls_num,dtype=torch.float32)
        # rel=torch.zeros(self.rel_num,dtype=torch.float32)
        return frames,bbx,mask,label,cls,rel

class CLIPFeatureDatasetCLSRELNonMask(Dataset):
    def __init__(self,name,sample_each_clip=16,node_nums=10,train=True):
        super().__init__()
        self.sample_rate=sample_each_clip
        self.node_nums=node_nums
        self.name=name
        self.train=train
        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
        self.mask=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_mask.json'),'r')
        )
        self.bbx=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_bbx.json'),'r')
        )
        self.cls=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_obj_cls.json'),'r')
        )
        self.rel=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_rel.json'),'r')
        )
        keys=list(self.json.keys())
        self.keys=[item for item in keys if 'label' not in item]
        self.label=[i for i in keys if 'label' in i]
        self.video_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all_cls_rel',name+'.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
        self.num_cls=157
        self.obj_cls_num=38
        self.rel_num=30
    
    def __len__(self):
        return len(self.keys)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    #  person relation -> 0
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        key=self.keys[idx]
        # key -> cls_video_id
        # frame feature videos[key][value[0~]]
        # bbx videos[key][value[0~]bbx]
        # mask list videos[key][value[0~]mask]
        frame_ids=self.json[key]
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
        video_size=self.video2size[key.split('.')[0]]
        frames=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]]),dtype=np.float16)).reshape(1,11,512) for index in indices]

        bbx=torch.tensor([self.bbx[key][index] for index in indices],dtype=torch.float32)
        # mask=np.array([self.mask[key][index] for index in indices],dtype=np.int64)
        mask=torch.tensor([self.mask[key][index] for index in indices],dtype=torch.long)
        cls_ids=torch.tensor([self.cls[key][index] for index in indices],dtype=torch.long)
        rel_ids=[self.rel[key][index] for index in indices]

        # print(mask)
        
        # extra_zero=np.array([0] * (10 - len(mask[0])))
        # for m in mask:
        #     _=torch.from_numpy(np.concatenate([m,extra_zero])).unsqueeze(0)
        #     mask_.append(_)
        label_idx=self.json[self.label[idx]]
        label_idx=[int(x) for x in label_idx]
        label=torch.zeros(self.num_cls,dtype=torch.float32)
        frames=torch.concat(frames,dim=0).float()
        label[label_idx]=1.0
        bbx[:,:,0]/=video_size[0]
        bbx[:,:,1]/=video_size[1]
        bbx[:,:,2]/=video_size[0]
        bbx[:,:,3]/=video_size[1]
        zero_tensor=torch.tensor([[0.,0.,1.,1.]]).to(bbx)
        zero_tensor=zero_tensor.unsqueeze(0).repeat(16,1,1)
        bbx=torch.cat([zero_tensor,bbx],dim=-2)
        # mask=torch.concat(mask_,dim=0).long()

        # mask=torch.concat(mask,dim=0).long()
        # mask_tensor_expanded = mask.bool().unsqueeze(-1).expand(-1, -1, 512)
        # frames[~mask_tensor_expanded]=0.
        rel=torch.zeros((self.sample_rate,self.node_nums,self.rel_num),dtype=torch.float32)
        cls=torch.zeros((self.sample_rate,self.node_nums,self.obj_cls_num),dtype=torch.float32)
        cls.scatter_(2,cls_ids.unsqueeze(-1),1.)
        for i in range(self.sample_rate):
            for j in range(self.node_nums):
                rel[i][j][rel_ids[i][j]]=1.
        # breakpoint()
        # cls=torch.zeros(self.obj_cls_num,dtype=torch.float32)
        # rel=torch.zeros(self.rel_num,dtype=torch.float32)
        return frames,bbx,mask,label,cls,rel

class CLIPFeatureDatasetCLSRELOracle(Dataset):
    def __init__(self,name,sample_each_clip=16,node_nums=10,train=True):
        super().__init__()
        self.sample_rate=sample_each_clip
        self.node_nums=node_nums
        self.name=name
        self.train=train
        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
        self.mask=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_mask.json'),'r')
        )
        self.bbx=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_bbx.json'),'r')
        )
        self.cls=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_obj_cls.json'),'r')
        )
        self.rel=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_rel.json'),'r')
        )
        keys=list(self.json.keys())
        self.keys=[item for item in keys if 'label' not in item]
        self.label=[i for i in keys if 'label' in i]
        self.video_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all_cls_rel',name+'.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
        self.num_cls=157
        self.obj_cls_num=38
        self.rel_num=30
    
    def __len__(self):
        return len(self.keys)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    #  person relation -> 0
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        key=self.keys[idx]
        # key -> cls_video_id
        # frame feature videos[key][value[0~]]
        # bbx videos[key][value[0~]bbx]
        # mask list videos[key][value[0~]mask]
        frame_ids=self.json[key]
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
        video_size=self.video2size[key.split('.')[0]]
        frames=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]]),dtype=np.float16)).reshape(1,11,512) for index in indices]

        bbx=torch.tensor([self.bbx[key][index] for index in indices],dtype=torch.float32)
        # mask=np.array([self.mask[key][index] for index in indices],dtype=np.int64)
        mask=torch.tensor([self.mask[key][index] for index in indices],dtype=torch.long)
        cls_ids=torch.tensor([self.cls[key][index] for index in indices],dtype=torch.long)
        rel_ids=[self.rel[key][index] for index in indices]
        # print(rel_ids)
        # rel_ids=torch.tensor([self.rel[key][index] for index in indices],dtype=torch.long)

        # print(mask)
        
        # extra_zero=np.array([0] * (10 - len(mask[0])))
        # for m in mask:
        #     _=torch.from_numpy(np.concatenate([m,extra_zero])).unsqueeze(0)
        #     mask_.append(_)
        label_name=key+'_label'
        label_idx=self.json[label_name]
        label_idx=[int(x) for x in label_idx]
        label=torch.zeros(self.num_cls,dtype=torch.float32)
        frames=torch.concat(frames,dim=0).float()
        label[label_idx]=1.0
        bbx[:,:,0]/=video_size[0]
        bbx[:,:,1]/=video_size[1]
        bbx[:,:,2]/=video_size[0]
        bbx[:,:,3]/=video_size[1]
        zero_tensor=torch.tensor([[0.,0.,1.,1.]]).to(bbx)
        zero_tensor=zero_tensor.unsqueeze(0).repeat(16,1,1)
        bbx=torch.cat([zero_tensor,bbx],dim=-2)
        # mask=torch.concat(mask_,dim=0).long()

        # mask=torch.concat(mask,dim=0).long()
        # mask_tensor_expanded = mask.bool().unsqueeze(-1).expand(-1, -1, 512)
        # frames[~mask_tensor_expanded]=0.
        rel=torch.zeros((self.sample_rate,self.node_nums,self.rel_num),dtype=torch.float32)
        cls_cls=torch.zeros((self.sample_rate,self.node_nums,self.obj_cls_num),dtype=torch.float32)
        cls_cls.scatter_(2,cls_ids.unsqueeze(-1),1.)
        for i in range(self.sample_rate):
            for j in range(self.node_nums):
                rel[i][j][rel_ids[i][j]]=1.
        # breakpoint()
        # cls=torch.zeros(self.obj_cls_num,dtype=torch.float32)
        # rel=torch.zeros(self.rel_num,dtype=torch.float32)
        return frames,bbx,mask,label,cls_ids,cls_cls,rel[:,1:,:]

class MixAns(Dataset):
    def __init__(self,name,sample_each_clip=16,node_nums=10,train=True):
        super().__init__()
        self.sample_rate=sample_each_clip
        self.node_nums=node_nums
        self.name=name
        self.train=train
        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
        self.mapping=json.load(
            open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping',name+'.json'))
        )
        self.mask=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_mask.json'),'r')
        )
        self.bbx=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_bbx.json'),'r')
        )
        self.cls=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_obj_cls.json'),'r')
        )
        self.rel=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_rel.json'),'r')
        )
        # keys=list(self.json.keys())
        # self.keys=[item for item in keys if 'label' not in item]
        # self.label=[i for i in keys if 'label' in i]
        self.video_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all_cls_rel',name+'.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
        self.num_cls=157
        self.obj_cls_num=38
        self.rel_num=30
    
    def __len__(self):
        return len(self.mapping)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    #  person relation -> 0
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        # key=self.keys[idx]
        tmp=self.mapping[idx]
        key=tmp['id']
        private_ans=tmp['private']
        common_ans=tmp['common']
        tokens=tmp['token']
        # key -> cls_video_id
        # frame feature videos[key][value[0~]]
        # bbx videos[key][value[0~]bbx]
        # mask list videos[key][value[0~]mask]
        frame_ids=self.json[key]
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
        video_size=self.video2size[key.split('.')[0]]
        frames=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]]),dtype=np.float16)).reshape(1,11,512) for index in indices]

        bbx=torch.tensor([self.bbx[key][index] for index in indices],dtype=torch.float32)
        # mask=np.array([self.mask[key][index] for index in indices],dtype=np.int64)
        mask=torch.tensor([self.mask[key][index] for index in indices],dtype=torch.long)
        cls_ids=torch.tensor([self.cls[key][index] for index in indices],dtype=torch.long)
        rel_ids=[self.rel[key][index] for index in indices]

        
        label_name=key+'_label'
        label_idx=self.json[label_name]
        label_idx=[int(x) for x in label_idx]
        label=torch.zeros(self.num_cls,dtype=torch.float32)

        private_label=torch.zeros(self.num_cls+1,dtype=torch.float32)
        common_label=torch.zeros(self.num_cls+1,dtype=torch.float32)
        token_tensor=torch.zeros(self.num_cls,dtype=torch.float32)


        frames=torch.concat(frames,dim=0).float()
        label[label_idx]=1.0
        private_label[private_ans]=1.0
        common_label[common_ans]=1.0
        token_tensor[tokens]=1.0




        bbx[:,:,0]/=video_size[0]
        bbx[:,:,1]/=video_size[1]
        bbx[:,:,2]/=video_size[0]
        bbx[:,:,3]/=video_size[1]
        zero_tensor=torch.tensor([[0.,0.,1.,1.]]).to(bbx)
        zero_tensor=zero_tensor.unsqueeze(0).repeat(16,1,1)
        bbx=torch.cat([zero_tensor,bbx],dim=-2)
        # mask=torch.concat(mask_,dim=0).long()

        # mask=torch.concat(mask,dim=0).long()
        # mask_tensor_expanded = mask.bool().unsqueeze(-1).expand(-1, -1, 512)
        # frames[~mask_tensor_expanded]=0.
        rel=torch.zeros((self.sample_rate,self.node_nums,self.rel_num),dtype=torch.float32)
        cls_cls=torch.zeros((self.sample_rate,self.node_nums,self.obj_cls_num),dtype=torch.float32)
        cls_cls.scatter_(2,cls_ids.unsqueeze(-1),1.)
        for i in range(self.sample_rate):
            for j in range(self.node_nums):
                rel[i][j][rel_ids[i][j]]=1.
        # breakpoint()
        # cls=torch.zeros(self.obj_cls_num,dtype=torch.float32)
        # rel=torch.zeros(self.rel_num,dtype=torch.float32)
        return frames,bbx,mask,label,cls_ids,cls_cls,rel[:,1:,:],private_label,common_label,token_tensor

class MixAns2(Dataset):
    def __init__(self,name,sample_each_clip=16,node_nums=10,mapping_type=2,train=True):
        super().__init__()
        self.sample_rate=sample_each_clip
        self.node_nums=node_nums
        self.name=name
        self.train=train
        self.mapping_type=mapping_type
        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
        if mapping_type==1:
            self.mapping=json.load(
                open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping_expand',name+'.json'))
            )
        elif mapping_type==2:
            self.mapping=json.load(
                open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping_expand2',name+'.json'))
            )
        elif mapping_type==3:
            self.mapping=json.load(
                open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping_expand3',name+'.json'))
            )
        elif mapping_type==4:
            self.mapping=json.load(
                open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping_seperate','type_1','test1.json'),'r')
        )
        else:
            raise ModuleNotFoundError
        self.mask=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_mask.json'),'r')
        )
        self.bbx=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_bbx.json'),'r')
        )
        self.cls=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_obj_cls.json'),'r')
        )
        self.rel=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_rel.json'),'r')
        )
        # keys=list(self.json.keys())
        # self.keys=[item for item in keys if 'label' not in item]
        # self.label=[i for i in keys if 'label' in i]
        self.video_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all_cls_rel',name+'.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
        self.num_cls=157
        self.obj_cls_num=38
        self.rel_num=30
    
    def __len__(self):
        return len(self.mapping)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    #  person relation -> 0
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        # key=self.keys[idx]
        tmp=self.mapping[idx]
        key=tmp['id']
        private_ans=tmp['private']
        common_ans=tmp['common']
        tokens=tmp['token']
        # key -> cls_video_id
        # frame feature videos[key][value[0~]]
        # bbx videos[key][value[0~]bbx]
        # mask list videos[key][value[0~]mask]
        frame_ids=self.json[key]
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
        video_size=self.video2size[key.split('.')[0]]
        frames=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]]),dtype=np.float16)).reshape(1,11,512) for index in indices]

        bbx=torch.tensor([self.bbx[key][index] for index in indices],dtype=torch.float32)
        # mask=np.array([self.mask[key][index] for index in indices],dtype=np.int64)
        mask=torch.tensor([self.mask[key][index] for index in indices],dtype=torch.long)
        cls_ids=torch.tensor([self.cls[key][index] for index in indices],dtype=torch.long)
        rel_ids=[self.rel[key][index] for index in indices]

        
        label_name=key+'_label'
        label_idx=self.json[label_name]
        label_idx=[int(x) for x in label_idx]
        label=torch.zeros(self.num_cls,dtype=torch.float32)
        mask_=torch.zeros(self.num_cls+1,dtype=torch.long)

        private_label=torch.zeros(self.num_cls+1,dtype=torch.float32)
        common_label=torch.zeros(self.num_cls+1,dtype=torch.float32)
        token_tensor=torch.zeros(self.num_cls,dtype=torch.float32)


        frames=torch.concat(frames,dim=0).float()
        label[label_idx]=1.0
        mask_[label_idx]=1
        private_label[private_ans]=1.0
        common_label[common_ans]=1.0
        token_tensor[tokens]=1.0
        # token_tensor=torch.tensor(tokens,dtype=torch.long)




        bbx[:,:,0]/=video_size[0]
        bbx[:,:,1]/=video_size[1]
        bbx[:,:,2]/=video_size[0]
        bbx[:,:,3]/=video_size[1]
        zero_tensor=torch.tensor([[0.,0.,1.,1.]]).to(bbx)
        zero_tensor=zero_tensor.unsqueeze(0).repeat(16,1,1)
        bbx=torch.cat([zero_tensor,bbx],dim=-2)
        # mask=torch.concat(mask_,dim=0).long()

        # mask=torch.concat(mask,dim=0).long()
        # mask_tensor_expanded = mask.bool().unsqueeze(-1).expand(-1, -1, 512)
        # frames[~mask_tensor_expanded]=0.
        rel=torch.zeros((self.sample_rate,self.node_nums,self.rel_num),dtype=torch.float32)
        cls_cls=torch.zeros((self.sample_rate,self.node_nums,self.obj_cls_num),dtype=torch.float32)
        cls_cls.scatter_(2,cls_ids.unsqueeze(-1),1.)
        for i in range(self.sample_rate):
            for j in range(self.node_nums):
                rel[i][j][rel_ids[i][j]]=1.
        # breakpoint()
        # cls=torch.zeros(self.obj_cls_num,dtype=torch.float32)
        # rel=torch.zeros(self.rel_num,dtype=torch.float32)
        return frames,bbx,mask,label,cls_ids,cls_cls,rel[:,1:,:],private_label,common_label,token_tensor,mask_

class TestMixAns(Dataset):
    def __init__(self,name,d_t,d_i,sample_each_clip=16,node_nums=10,train=True):
        super().__init__()
        self.d_type='type_1' if d_t==1 else 'type_2'
        self.d_index=d_i
        self.sample_rate=sample_each_clip
        self.node_nums=node_nums
        self.name=name
        self.train=train

        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )

        self.mapping=json.load(
                open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping_seperate',self.d_type,name+str(self.d_index)+'.json'),'r')
        )
        self.mask=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_mask.json'),'r')
        )
        self.bbx=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_bbx.json'),'r')
        )
        self.cls=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_obj_cls.json'),'r')
        )
        self.rel=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'_rel.json'),'r')
        )
        # keys=list(self.json.keys())
        # self.keys=[item for item in keys if 'label' not in item]
        # self.label=[i for i in keys if 'label' in i]
        self.video_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all_cls_rel',name+'.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
        self.num_cls=157
        self.obj_cls_num=38
        self.rel_num=30
    
    def __len__(self):
        return len(self.mapping)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    #  person relation -> 0
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        # key=self.keys[idx]
        tmp=self.mapping[idx]
        key=tmp['id']
        private_ans=tmp['private']
        common_ans=tmp['common']
        tokens=tmp['token']
        # key -> cls_video_id
        # frame feature videos[key][value[0~]]
        # bbx videos[key][value[0~]bbx]
        # mask list videos[key][value[0~]mask]
        frame_ids=self.json[key]
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
        video_size=self.video2size[key.split('.')[0]]
        frames=[torch.from_numpy(np.frombuffer(np.array(self.videos[key][frame_ids[index]]),dtype=np.float16)).reshape(1,11,512) for index in indices]

        bbx=torch.tensor([self.bbx[key][index] for index in indices],dtype=torch.float32)
        # mask=np.array([self.mask[key][index] for index in indices],dtype=np.int64)
        mask=torch.tensor([self.mask[key][index] for index in indices],dtype=torch.long)
        cls_ids=torch.tensor([self.cls[key][index] for index in indices],dtype=torch.long)
        rel_ids=[self.rel[key][index] for index in indices]

        
        label_name=key+'_label'
        label_idx=self.json[label_name]
        label_idx=[int(x) for x in label_idx]
        label=torch.zeros(self.num_cls,dtype=torch.float32)
        mask_=torch.zeros(self.num_cls+1,dtype=torch.long)

        private_label=torch.zeros(self.num_cls+1,dtype=torch.float32)
        common_label=torch.zeros(self.num_cls+1,dtype=torch.float32)
        token_tensor=torch.zeros(self.num_cls,dtype=torch.float32)


        frames=torch.concat(frames,dim=0).float()
        label[label_idx]=1.0
        mask_[label_idx]=1
        private_label[private_ans]=1.0
        common_label[common_ans]=1.0
        token_tensor[tokens]=1.0
        # token_tensor=torch.tensor(tokens,dtype=torch.long)




        bbx[:,:,0]/=video_size[0]
        bbx[:,:,1]/=video_size[1]
        bbx[:,:,2]/=video_size[0]
        bbx[:,:,3]/=video_size[1]
        zero_tensor=torch.tensor([[0.,0.,1.,1.]]).to(bbx)
        zero_tensor=zero_tensor.unsqueeze(0).repeat(16,1,1)
        bbx=torch.cat([zero_tensor,bbx],dim=-2)
        # mask=torch.concat(mask_,dim=0).long()

        # mask=torch.concat(mask,dim=0).long()
        # mask_tensor_expanded = mask.bool().unsqueeze(-1).expand(-1, -1, 512)
        # frames[~mask_tensor_expanded]=0.
        rel=torch.zeros((self.sample_rate,self.node_nums,self.rel_num),dtype=torch.float32)
        cls_cls=torch.zeros((self.sample_rate,self.node_nums,self.obj_cls_num),dtype=torch.float32)
        cls_cls.scatter_(2,cls_ids.unsqueeze(-1),1.)
        for i in range(self.sample_rate):
            for j in range(self.node_nums):
                rel[i][j][rel_ids[i][j]]=1.
        # breakpoint()
        # cls=torch.zeros(self.obj_cls_num,dtype=torch.float32)
        # rel=torch.zeros(self.rel_num,dtype=torch.float32)
        return frames,bbx,mask,label,cls_ids,cls_cls,rel[:,1:,:],private_label,common_label,token_tensor,mask_


class ImageDatsetALL(Dataset):
    def __init__(self,name,sample_each_clip=16,train=True):
        super().__init__()
        self.sample_rate=sample_each_clip
        self.name=name
        self.train=train
        self.json=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all",name+'.json'),'r')
        )
        self.resize = Resize((math.floor(256 * 1.15),math.floor(384*1.15)))
        self.transforms = Compose(
            [
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        keys=list(self.json.keys())
        self.keys=[item for item in keys if 'label' not in item]
        self.label=[i for i in keys if 'label' in i]
        self.video_path=os.path.join('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/data_save/hdf5file',name+'_6.hdf5')
        self.video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
        self.num_cls=157
    
    def __len__(self):
        return len(self.keys)
    
    def open_video(self):
        self.videos = h5py.File(
            self.video_path, 
            "r", libver="latest", swmr=True
        )
    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_video()
        key=self.keys[idx]

        frame_ids=self.json[key]
        
        label_idx=self.json[self.label[idx]]
        label_idx=[int(x) for x in label_idx]
        label=torch.zeros(self.num_cls,dtype=torch.float32)

        label[label_idx]=1.0
        indices = sample_appearance_indices(
            self.sample_rate, len(frame_ids),self.train 
        )
   
        # print(key,frame_ids)
        raw_video_frames = [
            self.resize(
                Image.open(io.BytesIO(np.array(self.videos[key.split('.')[0]][frame_ids[index].split('.')[0]])))
            )
            for index in indices
        ]
        augment = IdentityTransform()
        if self.train:
            augment = VideoColorJitter()
            top, left, height, width = RandomCrop.get_params(
                raw_video_frames[0],
                (256, 384),
            )

        video_frames = []
        for i in range(len(raw_video_frames)):
            frame = raw_video_frames[i]
            frame = augment(frame)
            frame = (
                TF.crop(frame, top, left, height, width)
                if self.train
                else TF.center_crop(frame, (256,384))
            )
            frame = self.transforms(frame)
            video_frames.append(frame)

        video_frames = torch.stack(video_frames, dim=0).transpose(0,1)
        return video_frames,label

class AppearanceDataset(Dataset):
    def __init__(self,name, train):
        self.train=train

        self.json_file = json.load(open(os.path.join('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/data_save/jsonfile',
                                                     name+'_6.json')))
        self.labels = json.load(open(os.path.join('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/videolabel','ag.json')))
        self.videoid2size = json.load(open(os.path.join('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size','ag.json')))
        self.resize = Resize((math.floor(256 * 1.15),math.floor(384*1.15)))
        self.transforms = Compose(
            [
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.video_path=os.path.join('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/data_save/hdf5file',name+'_.hdf5')

    def __len__(self):
        return len(self.json_file)

    def open_videos(self):
        self.videos = h5py.File(
            self.video_path, "r", libver="latest", swmr=True
        )

    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_videos()
        video_id = self.json_file[idx]["id"]
        num_frames = len(self.videos[video_id])
        indices = sample_appearance_indices(
            16,num_frames, self.train
        )
        # Load all frames
        video_base_path='/home/wtc/revisiting-spatial-temporal-layouts/data/action_genome/frames'
        video_path=natsorted(os.listdir(os.path.join(video_base_path,video_id+'.mp4')))
        raw_video_frames = [
            self.resize(
                Image.open(io.BytesIO(np.array(self.videos[video_id][video_path[index].split('.')[0]])))
            )
            for index in indices
        ]
        augment = IdentityTransform()
        if self.train:
            augment = VideoColorJitter()
            top, left, height, width = RandomCrop.get_params(
                raw_video_frames[0],
                (256, 384),
            )

        video_frames = []
        for i in range(len(raw_video_frames)):
            frame = raw_video_frames[i]
            frame = augment(frame)
            frame = (
                TF.crop(frame, top, left, height, width)
                if self.config.train
                else TF.center_crop(frame, (256,384))
            )
            frame = self.transforms(frame)
            video_frames.append(frame)

        video_frames = torch.stack(video_frames, dim=0).transpose(0, 1)
        # Obtain video label
        # print(self.labels)

        action_list = [int(action[1:]) for action in self.json_file[idx]["actions"]]
        actions = torch.zeros(len(self.labels), dtype=torch.float)
        actions[action_list] = 1.0


        # video_label = torch.tensor(
        #     int(self.labels[re.sub("[\[\]]", "", self.json_file[idx]["template"])])
        # )

        return video_frames,actions

class StltDataset(Dataset):
    def __init__(self,name,config):
        self.config=config
        self.json_file = json.load(open(os.path.join('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/data_save/jsonfile',
                                                     name+'_6.json')))
        self.labels = json.load(open(os.path.join('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/videolabel','ag.json')))
        self.videoid2size = json.load(open(os.path.join('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size','ag.json')))
        # Find max num objects
        max_objects = -1
        for video in self.json_file:
            for video_frame in video["frames"]:
                cur_num_objects = 0
                for frame_object in video_frame["frame_objects"]:
                    if frame_object["score"] >= self.config.score_threshold:
                        cur_num_objects += 1
                max_objects = max(max_objects, cur_num_objects)
        self.max_actions=0
        # max_objects=10
        # relations -> 9
        # tokens=144
        self.max_objects=10

    def __len__(self):
        return len(self.json_file)

    def __getitem__(self, idx: int):
        video_id = self.json_file[idx]["id"]
        video_size = torch.tensor(self.videoid2size[video_id]).repeat(2)
        boxes, categories, scores, frame_types = [], [], [], []
        num_frames = len(self.json_file[idx]["frames"])
        indices = (
            sample_train_layout_indices(self.config.layout_num_frames, num_frames)
            if self.config.train
            else get_test_layout_indices(self.config.layout_num_frames, num_frames)
        )
        while len(indices)<self.config.layout_num_frames:
            indices.append(indices[-1])
        
        # print('num_frames',self.config.layout_num_frames,len(indices))
        # breakpoint()
        for index in indices:
            frame = self.json_file[idx]["frames"][index]
            # Prepare CLS object
           
            frame_scores = []
            frame_boxes = []
            frame_categories = []
            # Iterate over the other objects
            for element in frame["frame_objects"]:
                if element["score"] < self.config.score_threshold:
                    continue
                # Prepare box
                box = [element["x1"], element["y1"], element["x2"], element["y2"]]
                box = fix_box(
                    box, (video_size[1].item(), video_size[0].item())
                )  # Height, Width
                box = torch.tensor(box) / video_size
                frame_boxes.append(box)
                # Prepare category
                frame_categories.append(self.config.category2id[element["category"]])
                # Prepare scores
                frame_scores.append(element["score"])
            # Ensure that everything is of the same length and pad to the max number of objects
            assert len(frame_boxes) == len(frame_categories)
            while len(frame_boxes) != self.max_objects:
                frame_boxes.append(torch.full((4,), 0.0))
                frame_categories.append(0)
                frame_scores.append(0.0)
            categories.append(torch.tensor(frame_categories))
            scores.append(torch.tensor(frame_scores))
            boxes.append(torch.stack(frame_boxes, dim=0))
        # Prepare extract element
        # Boxes

        # Length
        length = torch.tensor(len(categories))
        # Frame types
        frame_types.append(self.config.frame2type["extract"])
        # Get action(s)
        actions = self.get_actions(self.json_file[idx])

        return video_id,torch.stack(categories, dim=0),torch.stack(boxes, dim=0),torch.stack(scores, dim=0),torch.tensor(frame_types),length, actions


    def get_actions(self, sample):
        if self.config.dataset_name == "something":
            actions = torch.tensor(
                int(self.labels[re.sub("[\[\]]", "", sample["template"])])
            )
        elif self.config.dataset_name == "action_genome":
            action_list = [int(action[1:]) for action in sample["actions"]]
            actions = torch.zeros(len(self.labels), dtype=torch.float)
            actions[action_list] = 1.0
        self.max_actions=max(len(action_list),self.max_actions)
        return actions


from tqdm import tqdm
if __name__=='__main__':
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
    test_loader=DataLoader(test_dataset,batch_size=64,num_workers=16)


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
    train_loader=DataLoader(train_dataset,batch_size=64,num_workers=12,shuffle=True)
    for b in test_loader:
        ...
    print(test_dataset.max_actions)



