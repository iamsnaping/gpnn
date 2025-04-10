'''
创建一个映射表。
dict
{"class_video_id":[frame_id1,frame_id2....]}


'''
import pickle
import json
import pandas as pd
import re
from natsort import natsorted
import os
import  math
from copy import deepcopy as dp
import numpy as np
import cv2
import torch
import clip
from tqdm import tqdm
from PIL import Image
import h5py
import random
import sys
sys.path.append('/home/wu_tian_ci/GAFL')
from mytest.draw import draw_list_multi
# def get_cls_num():
#     base_path='/home/wu_tian_ci/GAFL/data/action_genome/object_classes.txt'
#     f=open(base_path,'r')
#     content=f.readlines()
#     cls_num={}
#     count=0

#     for c in content:
#         c_=c.replace('\n','')
#         cls_num[c_]=count
#         count+=1

#     return cls_num
from torch.utils.data import Dataset




def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 

def get_cls_num():
    cls_num={
                "pad": 0,
                "cls": 1,
                "chair": 2,
                "book": 3,
                "medicine": 4,
                "vacuum": 5,
                "food": 6,
                "groceries": 7,
                "floor": 8,
                "mirror": 9,
                "closet/cabinet": 10,
                "doorway": 11,
                "paper/notebook": 12,
                "picture": 13,
                "phone/camera": 14,
                "sofa/couch": 15,
                "sandwich": 16,
                "cup/glass/bottle": 17,
                "towel": 18,
                "box": 19,
                "blanket": 20,
                "television": 21,
                "bag": 22,
                "refrigerator": 23,
                "table": 24,
                "light": 25,
                "broom": 26,
                "shoe": 27,
                "doorknob": 28,
                "bed": 29,
                "window": 30,
                "shelf": 31,
                "door": 32,
                "pillow": 33,
                "laptop": 34,
                "dish": 35,
                "clothes": 36,
                "person": 37,
            }
    return cls_num


def process_cls(repeat_cls):
    pattern=r'c(\d{3})'
    all_=re.findall(pattern,repeat_cls)
    cls_num=len(all_)
    sorted=natsorted(all_)
    combined=','.join(sorted)
    return combined,cls_num

def add_zero(before):
    after=(10-len(before))*'0'+before
    return after

def get_dict():
    base_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'
    file_list=os.listdir(base_path)
    file_dict={}
    for f in file_list:
        # if 
        ff=f.split('.')[0]
        file_dict[ff]=1
    return file_dict

def get_frame_dict():
    object_bbox= pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
    )
    count_zero_dict={}
    keys=object_bbox.keys()
    for key in keys:
        obj_list=[]
        objs=object_bbox[key]
        for obj in objs:
            if  not obj['visible']:
                continue
            obj_list.append(obj['bbox'])
        if person_bbox[key]['bbox'].shape == (1,4):
            obj_list.append(person_bbox[key]['bbox'])
        if len(obj_list)==0:
            count_zero_dict[key]=1

    return count_zero_dict

def trans_(csv_name,file_path):
    except_list=['c026_LKH9A.mp4'
                 ,'c154_FC2SK.mp4'
                 ,'c020_FC2SK.mp4'
                 ,'c130_FC2SK.mp4'
                 ,'c116_FC2SK.mp4'
                 ,'c123_FC2SK.mp4'
                 ,'c006_FC2SK.mp4'
                 ,'c014_FC2SK.mp4']
    file_dict=get_dict()
    zero_dict=get_frame_dict()
    all_basepath='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/allframes'
    anno_basepath='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'
    if csv_name=='test':
        csv_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_test.csv'
    else:
        csv_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_train.csv'
    df=pd.read_csv(csv_path)
    json_savepath=os.path.join('/home/wu_tian_ci/GAFL/json_dataset',file_path+'.json')
    # video length
    lengths=df['length']
    # class
    actions=df['actions']
    # video id
    ids=df['id']

    pattern1=r'\b\d+\.\d+\b'
    pattern2 = r'c\d{3}'

    positive_count=0
    total_cls=0

    json_dict={}
    count_dict={}
    max_len=0
    min_len=999
    video_dict={}
    skip_count=0
    print('zero_keys',len(zero_dict.keys()))
    for _,l,a in zip(ids,lengths,actions):
        if file_dict.get(_) is None:
            skip_count+=1
            continue
        try:
            all_numbers=re.findall(pattern1,a)
            all_classes=re.findall(pattern2,a)
        except:
            continue
        video_id=_+'.mp4'
        origin_id=os.path.join(all_basepath,video_id)
        filter_id=os.path.join(anno_basepath,video_id)
        try:
            origin_frames=os.listdir(origin_id)
            filter_frames=os.listdir(filter_id)
        except:
            continue
        int_filter=[int(x.split('.')[0]) for x in filter_frames]
        int_filter=natsorted(int_filter)
        float_number=[float(X) for X in all_numbers]
        positive_count+=1
        max_float=max(float_number)
        max_float=max(max_float,float(l))
        # if max_float > float(l):
        #     counts+=1
            # print(max_float,' ',float(l))
        num_dict={}

        total_cls+=len(all_classes)
        
        for i in range(0,len(all_numbers),2):
            if num_dict.get(all_numbers[i]+all_numbers[i+1]) is None:
                num_dict[all_numbers[i]+all_numbers[i+1]]=all_classes[i//2]
            else:
                num_dict[all_numbers[i]+all_numbers[i+1]]+=','+all_classes[i//2]

        repeat_cls_str=''
        for key,item in num_dict.items():
            if len(item)>4:
                repeat_cls_str+=item
        frames_len=len(origin_frames)
        for i in range(0,len(all_numbers),2):
            clip_cls=all_classes[i//2]
            if clip_cls in repeat_cls_str:
                continue
            cls_vid=clip_cls+'_'+video_id
            cls_list=[]
            begin_idx=math.floor(float(all_numbers[i])/max_float*frames_len)
            end_idx=math.ceil(float(all_numbers[i+1])/max_float*frames_len)
            for f in int_filter:
                if begin_idx <=f and f<=end_idx:
                    f_name=add_zero(str(f)+'.png')
                    f_id=video_id+'/'+f_name
                    if zero_dict.get(f_id) is not None:
                        print('skip frame')
                        continue
                    cls_list.append(f_name)
            real_cls_id=dp(cls_vid)
            if real_cls_id in except_list:
                continue
            if count_dict.get(cls_vid) is not None:
                real_cls_id=cls_vid+str(count_dict[cls_vid])
                count_dict[cls_vid]+=1
            else:
                count_dict[cls_vid]=1
            json_dict[real_cls_id]=cls_list
            max_len=max(max_len,len(cls_list))
            min_len=min(min_len,len(cls_list))
            if len(cls_list)==0:
                video_dict[video_id]=1
                # print(real_cls_id,begin_idx,end_idx,all_numbers[i],all_numbers[i+1],frames_len)
    video_dict_keys=list(video_dict.keys())
    key_list=','.join(video_dict_keys)
    keys=json_dict.keys()
    delete_keys=[]
    for key in keys:
        video_name=key.split('_')[-1]
        if video_name[-1]=='1':
            video_name=video_name[:-1]
        if video_name in key_list:
            delete_keys.append(key)
    print('skip video',len(video_dict.keys()))
    print(len(keys),max_len,min_len)
    for key in delete_keys:
        del json_dict[key]
    print('skip video',len(video_dict.keys()))
    print(len(keys),max_len,min_len)
    json.dump(json_dict,open(json_savepath,'w'))
    print('skip',skip_count)

def process_img(img,bbx_list,mask_list,preprocess):
    tmp_list=torch.zeros(10,3,224,224)
    for i in range(len(mask_list)):
        if mask_list[i]==1:
            tmp_list[i]=preprocess(img.crop(bbx_list[i]))
    return tmp_list

def process_img_all(img,bbx_list,mask_list,preprocess,device=''):
    tmp_list=torch.zeros(11,3,224,224).to(device)
    tmp_list[0]=preprocess(img)
    for i in range(len(mask_list)):
        if mask_list[i]==1:
            tmp_list[i+1]=preprocess(img.crop(bbx_list[i]))
    return tmp_list

def pkl_process(name,device):
    hdf5_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all_features',name+'.hdf5')
    img_basepath='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'
    device=device
    model,preprocess=clip.load('ViT-B/16',device=device,jit=True)
    json_name=json.load(open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset',name+'.json'),'r'))
    object_bbox= pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
    )
    count_zero_dict={}
    cls_num=get_cls_num()
    print(name,'keys',len(json_name.keys()))
    
    hf=h5py.File(hdf5_path, "a", swmr=True)

    for key,value in tqdm(json_name.items()):
        add_value=[add_zero(x) for x in value]
        key_split=key.split('_')[1]

        grp=hf.create_group(key)

        # print('before',key_split)
        key_split=key_split if key_split[-2]=='p' else key_split[:-1]
        id_list=[(key_split+'/'+x) for x in add_value]
        cls_list=[]
        for id_ in id_list:
            objs=object_bbox[id_]
            for obj in objs:
                if not obj['visible']:
                    continue
                cls_list.append(cls_num[obj['class'].replace('/','')])
            if person_bbox[id_]['bbox'].shape == (1,4):
                cls_list.append(cls_num['person'])

        cls_list=np.unique(cls_list)
        cls_list.sort()
        
        for id_ in id_list:
            objs=object_bbox[id_]
            obj_dict={}
            mask_list=[]
            bbx_list=[]
            person_flag=False
            for obj in objs:
                if not obj['visible']:
                    continue
                obj_dict[cls_num[obj['class'].replace('/','')]]=obj['bbox']
            if person_bbox[id_]['bbox'].shape == (1,4):
                person_flag=True
                obj_dict[cls_num['person']]=person_bbox[id_]['bbox']
            for cls in cls_list:
                if obj_dict.get(cls) is None:
                    mask_list.append(0)
                    bbx_list.append([0.,0.,0.,0.])
                    continue
                bbx=obj_dict[cls]
                if type(bbx) == np.ndarray:
                    bbx=list(bbx.squeeze())
                else:
                    bbx=list(bbx)
                mask_list.append(1)
                bbx_list.append(bbx)
            if len(bbx_list) != len(mask_list):
                breakpoint

            bbx_list=bbx_list+[[0.,0.,0.,0.]]*(10-len(bbx_list))
            mask_list=mask_list+[0]*(10-len(mask_list))

            new_bbx=[(x[0],x[1],x[2]+x[0],x[3]+x[1]) for x in bbx_list]
            if person_flag: 
                new_bbx[0]=dp(bbx_list[0])
            # for bbx,mask,bbx_ in zip(new_bbx,mask_list,bbx_list):
            #     if (bbx[0]>= bbx[2] or bbx[1] >= bbx[3]) and mask==1:
            #         print(id_,bbx,bbx_)
            #         breakpoint()

            img=Image.open(os.path.join(img_basepath,id_))
            try:
                tmp_list=process_img(img,new_bbx,mask_list,preprocess)
            except:
                print(id_,bbx_list)
            tmp_list=tmp_list.to(device)
            with torch.no_grad():
                tmp_feature=model.encode_image(tmp_list)
                mask_tensor=~torch.tensor(mask_list,dtype=torch.bool,device=tmp_feature.device)
                tmp_feature[mask_tensor]=torch.zeros_like(tmp_feature[mask_tensor],device=tmp_feature.device)
                tmp_feature=tmp_feature.cpu().detach().numpy()
            print(tmp_feature.dtype)
            print(tmp_feature.shape)
            breakpoint()
            grp.create_dataset(id_.split('/')[1], data=tmp_feature)
            grp.create_dataset(id_.split('/')[1]+'bbx',data=np.array(bbx_list))
            grp.create_dataset(id_.split('/')[1]+'mask',data=np.array(mask_list))

    # hf.close()
            


    print('zero_keys',len(count_zero_dict.keys()))
    # for key in count_zero_dict.keys():
    #     print(key)

def get_relation(*kwags):
    relation2id={
            'lookingat':28,
            'notlookingat':1,
            'unsure':2,
            'above':3,
            'beneath':4,
            'infrontof':5,
            'behind':6,
            'onthesideof':7,
            'in':8,
            'carrying':9,
            'coveredby':10,
            'drinkingfrom':11,
            'eating':12,
            'haveitontheback':13,
            'holding':14,
            'leaningon':15,
            'lyingon':16,
            'notcontacting':17,
            'otherrelationship':18,
            'sittingon':19,
            'standingon':20,
            'touching':21,
            'twisting':22,
            'wearing':23,
            'wiping':24,
            'writingon':25,
            'norelation':26,
            'None':27,
            'cls':29,
            'pad':0
            }
    max_rel_len=9
    atten,spat,conta=kwags
    atte=[relation2id[a.replace('_','')] for a in atten]
    spa=[relation2id[a.replace('_','')] for a in spat]
    cont=[relation2id[a.replace('_','')] for a in conta]
    relation_list=atte+spa+cont
    # print(relation_list)
    return relation_list
    


# def pkl_process_all_(name,device):
#     hdf5_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all_features',name+'.hdf5')
#     img_basepath='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'
#     device=device
#     model,preprocess=clip.load('ViT-B/16',device=device,jit=True)
#     json_name=json.load(open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset',name+'.json'),'r'))
#     object_bbox= pickle.load(
#         open('/home/wu_tian_ci/GAFL/data/action_genome/object_bbox_and_relationship.pkl',"rb")
#     )
#     person_bbox = pickle.load(
#         open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
#     )
#     count_zero_dict={}
#     cls_num=get_cls_num()
#     print(name,'keys',len(json_name.keys()))
    
#     # hf=h5py.File(hdf5_path, "a", swmr=True)

#     for key,value in tqdm(json_name.items()):
#         add_value=[add_zero(x) for x in value]
#         key_split=key.split('_')[1]

#         # grp=hf.create_group(key)

#         # print('before',key_split)
#         key_split=key_split if key_split[-2]=='p' else key_split[:-1]
#         id_list=[(key_split+'/'+x) for x in add_value]
#         cls_list=[]
#         for id_ in id_list:
#             objs=object_bbox[id_]
#             for obj in objs:
#                 if not obj['visible']:
#                     continue
#                 cls_list.append(cls_num[obj['class'].replace('/','')])
#             if person_bbox[id_]['bbox'].shape == (1,4):
#                 cls_list.append(cls_num['person'])

#         cls_list=np.unique(cls_list)
#         cls_list.sort()
#         '''
#         for id_ in id_list:
#             objs=object_bbox[id_]
#             obj_dict={}
#             rel_dict={}
#             mask_list=[]
#             bbx_list=[]
#             rel_list=[]
#             person_flag=False
#             for obj in objs:
#                 if not obj['visible']:
#                     continue
#                 obj_dict[cls_num[obj['class'].replace('/','')]]=obj['bbox']
#                 rel_dict[cls_num[obj['class'].replace('/','')]]=get_relation(obj['attention_relationship'],obj['spatial_relationship'],obj['contacting_relationship'])
#             if person_bbox[id_]['bbox'].shape == (1,4):
#                 person_flag=True
#                 obj_dict[cls_num['person']]=person_bbox[id_]['bbox']
#             for cls in cls_list:
#                 if obj_dict.get(cls) is None:
#                     mask_list.append(0)
#                     bbx_list.append([0.,0.,0.,0.])
#                     rel_list.append([0])
#                     continue
#                 bbx=obj_dict[cls]
#                 if type(bbx) == np.ndarray:
#                     bbx=list(bbx.squeeze())
#                 else:
#                     bbx=list(bbx)
#                 mask_list.append(1)
#                 bbx_list.append(bbx)
#                 rel_list.append(rel_dict[cls])
#             if len(bbx_list) != len(mask_list):
#                 breakpoint

#             bbx_list=bbx_list+[[0.,0.,0.,0.]]*(10-len(bbx_list))
#             mask_list=mask_list+[0]*(10-len(mask_list))

#             new_bbx=[(x[0],x[1],x[2]+x[0],x[3]+x[1]) for x in bbx_list]
#             if person_flag: 
#                 new_bbx[0]=dp(bbx_list[0])
#             # for bbx,mask,bbx_ in zip(new_bbx,mask_list,bbx_list):
#             #     if (bbx[0]>= bbx[2] or bbx[1] >= bbx[3]) and mask==1:
#             #         print(id_,bbx,bbx_)
#             #         breakpoint()

#             img=Image.open(os.path.join(img_basepath,id_))
#             try:
#                 tmp_list=process_img_all(img,new_bbx,mask_list,preprocess)
#                 mask_list.insert(0,1)
#             except:
#                 print(id_,bbx_list)
#             tmp_list=tmp_list.to(device)
#             with torch.no_grad():
#                 tmp_feature=model.encode_image(tmp_list)
#                 mask_tensor=~torch.tensor(mask_list,dtype=torch.bool,device=tmp_feature.device)
#                 tmp_feature[mask_tensor]=torch.zeros_like(tmp_feature[mask_tensor],device=tmp_feature.device)
#                 tmp_feature=tmp_feature.cpu().detach().numpy()
#             print(tmp_feature.dtype)
#             print(tmp_feature.shape)
#             breakpoint()
#             grp.create_dataset(id_.split('/')[1], data=tmp_feature)
#             grp.create_dataset(id_.split('/')[1]+'bbx',data=np.array(bbx_list))
#             grp.create_dataset(id_.split('/')[1]+'mask',data=np.array(mask_list))

#     # hf.close()
            


#     print('zero_keys',len(count_zero_dict.keys()))
#     # for key in count_zero_dict.keys():
#     #     print(key)
#     '''




def cls_process(name):
    hdf5_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5',name+'.hdf5')
    json_name=json.load(open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset',name+'.json'),'r'))
    object_bbox= pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
    )
    count_zero_dict={}
    cls_num=get_cls_num()
    print(name,'keys',len(json_name.keys()))
    
    hf=h5py.File(hdf5_path, "a", swmr=True)

    for key,value in tqdm(json_name.items()):
        add_value=[add_zero(x) for x in value]
        key_split=key.split('_')[1]

        grp=hf[key]

        # print('before',key_split)
        key_split=key_split if key_split[-2]=='p' else key_split[:-1]
        id_list=[(key_split+'/'+x) for x in add_value]
        cls_list=[]
        for id_ in id_list:
            objs=object_bbox[id_]
            for obj in objs:
                if not obj['visible']:
                    continue
                cls_list.append(cls_num[obj['class'].replace('/','')])
            if person_bbox[id_]['bbox'].shape == (1,4):
                cls_list.append(cls_num['person'])

        cls_list=np.unique(cls_list)
        cls_list.sort()
        
        for id_ in id_list:
            objs=object_bbox[id_]
            obj_dict={}
            mask_list=[]
            bbx_list=[]
            cls_save=[]
            for obj in objs:
                if not obj['visible']:
                    continue
                obj_dict[cls_num[obj['class'].replace('/','')]]=obj['bbox']
            if person_bbox[id_]['bbox'].shape == (1,4):
                obj_dict[cls_num['person']]=person_bbox[id_]['bbox']
            for cls in cls_list:
                if obj_dict.get(cls) is None:
                    mask_list.append(0)
                    cls_save.append(0)
                    bbx_list.append([0.,0.,0.,0.])
                    continue
                bbx=obj_dict[cls]
                if type(bbx) == np.ndarray:
                    bbx=list(bbx.squeeze())
                else:
                    bbx=list(bbx)
                cls_save.append(cls)
                mask_list.append(1)
                bbx_list.append(bbx)
            if len(bbx_list) != len(mask_list) or (len(bbx_list)!=len(cls_save)):
                breakpoint
            cls_save=cls_save+[0]*(10-len(cls_save))
            print(np.array(cls_save).shape)
            breakpoint()
            # grp.create_dataset(id_.split('/')[1]+'obj_cls',data=np.array(cls_save,dtype=np.int16))
 

    # hf.close()
            


    print('zero_keys',len(count_zero_dict.keys()))
    # for key in count_zero_dict.keys():
    #     print(key)

def print_box():
    key='AHLVF.mp4'
    object_bbox= pickle.load(
        open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/v2/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/person_bbox.pkl', "rb")
    )
    json_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all','train.json')
    json_file=json.load(open(json_path,'r'))
    frame_ids=json_file[key]
    for frame_id in frame_ids:
        vf_id=key+'/'+frame_id
        objs=object_bbox[vf_id]
        for obj in objs:
            if not obj['visible']:
                continue
            print('obj:',obj['bbox'][0],obj['bbox'][1])
        if person_bbox[vf_id]['bbox'].shape == (1,4):
            print('per:',person_bbox[vf_id]['bbox'][0][0],person_bbox[vf_id]['bbox'][0][1])
        

def print_ex(name):
    json_name=json.load(open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset',name+'.json'),'r'))
    object_bbox= pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
    )
    cls_num=get_cls_num()
    print(name,'keys',len(json_name.keys()))
    # hf=h5py.File(hdf5_path, "a", swmr=True)

    for key,value in tqdm(json_name.items()):
        add_value=[add_zero(x) for x in value]
        key_split=key.split('_')[1]
        # grp=hf.create_group(key)
        # print('before',key_split)
        key_split=key_split if key_split[-2]=='p' else key_split[:-1]
        id_list=[(key_split+'/'+x) for x in add_value]
        cls_list=[]
        for id_ in id_list:
            objs=object_bbox[id_]
            for obj in objs:
                if not obj['visible']:
                    continue
                cls_list.append(cls_num[obj['class'].replace('/','')])
            if person_bbox[id_]['bbox'].shape == (1,4):
                cls_list.append(cls_num['person'])

        cls_list=np.unique(cls_list)
        cls_list.sort()
        
        for id_ in id_list:
            objs=object_bbox[id_]

            for obj in objs:
                if not obj['visible']:
                    continue
                bbx=obj['bbox']
                if bbx[2]<=0 or bbx[3]<=0:
                    print('obj',id_,bbx)
                
            if person_bbox[id_]['bbox'].shape == (1,4):
                p_bbx=person_bbox[id_]['bbox'][0]
                if p_bbx[0]>= p_bbx[2] or p_bbx[1]>= p_bbx[3]:
                    print('person',id_,p_bbx)
        
'''()()
    count_dict={}
    length_dict={}
    for rc in repeat_cls:
        sorted_str,_=process_cls(rc)
        total_repeat+=_
        if length_dict.get(_) is None:
            length_dict[_]=1
        else:
            length_dict[_]+=1
        if count_dict.get(sorted_str) is None:
            count_dict[sorted_str]=1
        else:
            count_dict[sorted_str]+=1
    keys=list(count_dict.keys())
    print(len(keys))
    # for key,item in count_dict.items():
        # print(key,item)
    for key,value in length_dict.items():
        print(key,value)
    print(len(ids),counts,repeat,positive_count,sample_repeat,total_cls,total_repeat)

'''



def tally(csv_name):  
    all_basepath=''
    anno_basepath=''
    if csv_name=='test':
        csv_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_test.csv'
    else:
        csv_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_train.csv'
    df=pd.read_csv(csv_path)
    lengths=df['length']
    actions=df['actions']
    ids=df['id']
    sample='c077 12.10 18.00;c079 11.80 17.30;c080 13.00 18.00;c076 11.80 17.50;c075 5.40 14.10'
    pattern1=r'\b\d+\.\d+\b'
    pattern2 = r'c\d{3}'
    counts=0
    repeat=0
    repeat_cls=[]
    sample_repeat=0
    positive_count=0
    total_cls=0
    total_repeat=0
    for _,l,a in zip(ids,lengths,actions):
        try:
            all_numbers=re.findall(pattern1,a)
            all_classes=re.findall(pattern2,a)
        except:
            # print('error',l,a,_)
            continue
        float_number=[float(X) for X in all_numbers]
        positive_count+=1
        max_float=max(float_number)
        if max_float > float(l):
            counts+=1
            # print(max_float,' ',float(l))
        num_dict={}
        repeat_flag=False
        total_cls+=len(all_classes)
        for i in range(0,len(all_numbers),2):
            if num_dict.get(all_numbers[i]+all_numbers[i+1]) is None:
                num_dict[all_numbers[i]+all_numbers[i+1]]=all_classes[i//2]
            else:
                repeat+=1
                repeat_flag=True
                num_dict[all_numbers[i]+all_numbers[i+1]]+=','+all_classes[i//2]
        if repeat_flag:
            sample_repeat+=1
        for key,item in num_dict.items():
            if len(item)>4:
                repeat_cls.append(item)
    count_dict={}
    length_dict={}
    for rc in repeat_cls:
        sorted_str,_=process_cls(rc)
        total_repeat+=_
        if length_dict.get(_) is None:
            length_dict[_]=1
        else:
            length_dict[_]+=1
        if count_dict.get(sorted_str) is None:
            count_dict[sorted_str]=1
        else:
            count_dict[sorted_str]+=1

    keys=list(count_dict.keys())
    print(len(keys))
    # for key,item in count_dict.items():
        # print(key,item)
    for key,value in length_dict.items():
        print(key,value)
    print(len(ids),counts,repeat,positive_count,sample_repeat,total_cls,total_repeat)

def test(name):
    if name=='test':
        json_path='/home/wu_tian_ci/GAFL/json_dataset/test.json'
    else:
        json_path='/home/wu_tian_ci/GAFL/json_dataset/train.json'
    print(name)
    f=json.load(open(json_path))
    max_len,min_len=0,99
    mean_list=[]

    for key,value in f.items():
        max_len=max(max_len,len(value))
        min_len=min(min_len,len(value))
        mean_list.append(len(value))
        if len(value) ==0:
            print(key)
    import numpy as np
    print(max_len,min_len,np.mean(mean_list))

def cut_test():
    png_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames/0EJAG.mp4/000100.png'
    name='0EJAG.mp4/000100.png'
    base_path='/home/wu_tian_ci/GAFL/json_dataset'
    object_bbox= pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
    )
    objs=object_bbox[name]
    obj=objs[0]
    print(obj['visible'],obj['class'],obj['bbox'])
    obbx=obj['bbox']
    f_bbx=[obbx[0],obbx[1],obbx[0]+obbx[2],obbx[1]+obbx[3]]
    print('fbbx',f_bbx)
    pbbx=person_bbox[name]['bbox'][0]
    img=Image.open(png_path)
    food_img=img.crop(tuple(f_bbx))
    person_img=img.crop(tuple(pbbx))
    food_img.save(os.path.join(base_path,'food.png'))
    person_img.save(os.path.join(base_path,'person.png'))

def tally_bbx_in_video_rel():  

    object_bbox= pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
    )

    cls_num=get_cls_num()

    base_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'
    video_names=os.listdir(base_path)
    count_len=dict()
    video_times=dict()
    duplicate_flag=False
    for video_id in video_names:
        frames_list=os.listdir(os.path.join(base_path,video_id))
        cls_list=[]
        for frame in frames_list:
            key=video_id+'/'+frame
            objs=object_bbox[key]
            obj_list=[]
            for obj in objs:
                if not obj['visible']:
                        continue
                cls_list.append(cls_num[obj['class'].replace('/','')])
                obj_list.append(cls_num[obj['class'].replace('/','')])
            if person_bbox[key]['bbox'].shape == (1,4):
                cls_list.append(cls_num['person'])
                obj_list.append(cls_num['person'])
            before_len=len(obj_list)
            after_len=np.unique(obj_list).size
            if before_len!=after_len:
                duplicate_flag=True

        cls_list=np.unique(cls_list)

        video_times[video_id]=cls_list.size
        if count_len.get(cls_list.size) is None:
            count_len[cls_list.size]=1
        else:
            count_len[cls_list.size]+=1
    
    for key,value in video_times.items():
        if value >10 or value<1:
            print(key)
    print('duplicate:',duplicate_flag)

def pkl_process_all(name,device):
    # skip_videos=['R4SJJ.mp4',
    #              'FC2SK.mp4',
    #              'C10FA.mp4',
    #              'X2LBW.mp4',
    #              'OZIJ7.mp4',
    #              'LKH9A.mp4']
    hdf5_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all',name+'.hdf5')
    img_basepath='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'
    device=device
    model,preprocess=clip.load('ViT-B/16',device=device,jit=True)
    json_name=json.load(open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all',name+'.json'),'r'))

    bbx_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all',name+'_bbx.json')
    mask_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all',name+'_mask.json')
    bbx_json=json.load(open(bbx_path,'r'))
    mask_json=json.load(open(mask_path))
    print(name,'keys',len(json_name.keys()))
    all_keys=json_name.keys()
    file_keys=[item for item in all_keys if 'label' not in item]
    label_keys=[i for i in all_keys if 'label' in i]
    assert len(file_keys) == len(label_keys)
    hf=h5py.File(hdf5_path, "a", swmr=True)

    for key in tqdm(file_keys):
        frames_ids=json_name[key]
        masks_list=mask_json[key]
        bbxes_list=bbx_json[key]
        grp=hf.create_group(key)
        feature_list=[]
        for frame_id,mask_list,bbx_list in zip(frames_ids,masks_list,bbxes_list):

            img=Image.open(os.path.join(img_basepath,key,frame_id))
            try:
                tmp_list=process_img(img,bbx_list,mask_list,preprocess)
            except:
                print(key,frame_id,bbx_list,len(mask_list),len(bbx_list))
            tmp_list=tmp_list.to(device)
            tmp_feature=model.encode_image(tmp_list)
            mask_tensor=~torch.tensor(mask_list,dtype=torch.bool,device=tmp_feature.device)
            tmp_feature[mask_tensor]=torch.zeros_like(tmp_feature[mask_tensor],device=tmp_feature.device)
            feature_list=tmp_feature.cpu().detach().numpy()
            grp.create_dataset(frame_id, data=feature_list)
            # grp.create_dataset(frame_id.split('/')[1]+'bbx',data=np.array(bbx_list))
            # grp.create_dataset(frame_id.split('/')[1]+'mask',data=np.array(mask_list))

    hf.close()

def pkl_process_all_cls_rel(name,device):
    # skip_videos=['R4SJJ.mp4',
    #              'FC2SK.mp4',
    #              'C10FA.mp4',
    #              'X2LBW.mp4',
    #              'OZIJ7.mp4',
    #              'LKH9A.mp4']
    hdf5_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all_cls_rel',name+'.hdf5')
    img_basepath='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'
    device=device
    model,preprocess=clip.load('ViT-B/16',device=device,jit=True)
    json_name=json.load(open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel',name+'.json'),'r'))

    bbx_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel',name+'_bbx.json')
    mask_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel',name+'_mask.json')
    bbx_json=json.load(open(bbx_path,'r'))
    mask_json=json.load(open(mask_path))
    print(name,'keys',len(json_name.keys()))
    all_keys=json_name.keys()
    file_keys=[item for item in all_keys if 'label' not in item]
    label_keys=[i for i in all_keys if 'label' in i]
    assert len(file_keys) == len(label_keys)
    hf=h5py.File(hdf5_path, "a", swmr=True)
    FLAG_TENSOR=torch.tensor([False]).to(device)
    for key in tqdm(file_keys):
        frames_ids=json_name[key]
        masks_list=mask_json[key]
        bbxes_list=bbx_json[key]
        grp=hf.create_group(key)
        feature_list=[]
        for frame_id,mask_list,bbx_list in zip(frames_ids,masks_list,bbxes_list):

            img=Image.open(os.path.join(img_basepath,key,frame_id))
            # tmp_list=process_img_all(img,bbx_list,mask_list,preprocess)

            try:
                tmp_list=process_img_all(img,bbx_list,mask_list,preprocess)
            except:
                print(key,frame_id,bbx_list,len(mask_list),len(bbx_list))
                # breakpoint()
            tmp_list=tmp_list.to(device)
            tmp_feature=model.encode_image(tmp_list)
            mask_list.insert(0,1)
            mask_tensor=~torch.tensor(mask_list,dtype=torch.bool,device=tmp_feature.device)
            tmp_feature[mask_tensor]=torch.zeros_like(tmp_feature[mask_tensor],device=tmp_feature.device)
            assert mask_tensor[0]==FLAG_TENSOR
            feature_list=tmp_feature.cpu().detach().numpy()
            grp.create_dataset(frame_id, data=feature_list)


    hf.close()


def pkl_process_all_cls_rel_shift(name,device):
    # skip_videos=['R4SJJ.mp4',
    #              'FC2SK.mp4',
    #              'C10FA.mp4',
    #              'X2LBW.mp4',
    #              'OZIJ7.mp4',
    #              'LKH9A.mp4']
    hdf5_path=os.path.join('/home/wu_tian_ci/GAFL/data/hdf5/all_cls_rel/shift',name+'.hdf5')
    img_basepath='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'
    device=device
    model,preprocess=clip.load('ViT-B/16',device=device,jit=True)
    json_name=json.load(open(os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel/shift',name+'.json'),'r'))

    bbx_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel/shift',name+'_bbx.json')
    mask_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel/shift',name+'_mask.json')
    bbx_json=json.load(open(bbx_path,'r'))
    mask_json=json.load(open(mask_path,'r'))
    print(name,'keys',len(json_name.keys()))
    all_keys=json_name.keys()
    file_keys=[item for item in all_keys if 'label' not in item]
    label_keys=[i for i in all_keys if 'label' in i]
    assert len(file_keys) == len(label_keys)
    hf=h5py.File(hdf5_path, "a", swmr=True)
    FLAG_TENSOR=torch.tensor([False]).to(device)
    with torch.no_grad():
        for key in tqdm(file_keys):
            frames_ids=json_name[key]
            masks_list=mask_json[key]
            bbxes_list=bbx_json[key]
            grp=hf.create_group(key)
            feature_list=[]
            tmp_lists=[]
            mask_lists=[]
            frameid_list=[]
            for idx,(frame_id,mask_list,bbx_list) in enumerate(zip(frames_ids,masks_list,bbxes_list),start=1):

                img=Image.open(os.path.join(img_basepath,key,frame_id))
                # tmp_list=process_img_all(img,bbx_list,mask_list,preprocess)

                try:
                    tmp_list=process_img_all(img,bbx_list,mask_list,preprocess,device)
                except:
                    print(key,frame_id,bbx_list,len(mask_list),len(bbx_list))
                    # breakpoint()
                tmp_lists.append(tmp_list)
                mask_lists.append([1]+mask_list)
                frameid_list.append(frame_id)
                if idx%32==0 or (idx==len(frames_ids)):
                    tmp_list_device=torch.cat(tmp_lists,dim=0).to(device)
                    tmp_feature=model.encode_image(tmp_list_device).to(device)
                    mask_list_device=~torch.tensor(mask_lists,dtype=torch.bool,device=device).reshape(-1)
                    tmp_feature[mask_list_device]=torch.zeros_like(tmp_feature[mask_list_device],device=device)
                    assert mask_list_device[0]==FLAG_TENSOR
                    feature_list=tmp_feature.cpu().detach().numpy()
                    for ids,frameid in enumerate(frameid_list,start=0):
                        grp.create_dataset(frameid, data=feature_list[ids*11:(ids+1)*11])
                    tmp_lists=[]
                    mask_lists=[]
                    frameid_list=[]

        hf.close()


def trans_all(csv_name,file_path):
    except_list=['R4SJJ.mp4',
                 'FC2SK.mp4',
                 'C10FA.mp4',
                 'X2LBW.mp4',
                 'OZIJ7.mp4',
                 'LKH9A.mp4']
    except_name=','.join(except_list)
    file_dict=get_dict()
    zero_dict=get_frame_dict()

    anno_basepath='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'
    if csv_name=='test':
        csv_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_test.csv'
    else:
        csv_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_train.csv'
    df=pd.read_csv(csv_path)
    json_savepath=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all',file_path+'.json')
    # video length
    lengths=df['length']
    # class
    actions=df['actions']
    # video id
    ids=df['id']

    pattern2 = r'c\d{3}'



    '''
    video_id:[frame_id1,...,frames_idn]
    '''
    frames_dict={}
    '''
    video_id:[[[x1,y1,x2,y2],...,[x1,y1,x1,y1]],
    [[x1,y1,x2,y2],...,[x1,y1,x1,y1]]](10 bbxes for every frames)
    '''
    bbx_dict={}
    '''
    video_id:[[0,0,1,...,1],...,[0,0,1,1,...,0]](10 objs)
    '''
    mask_dict={}


    for _,l,a in tqdm(zip(ids,lengths,actions)):
        if _ in except_name:
            print(_)
            continue
        if file_dict.get(_) is None:
            continue
        try:
            all_classes=re.findall(pattern2,a)
        except:
            continue

        all_classes=[x[1:] for x in all_classes]

        video_id=_+'.mp4'
        filter_id=os.path.join(anno_basepath,video_id)
        try:
            filter_frames=os.listdir(filter_id)
        except:
            continue
        int_filter=natsorted(filter_frames)
        frames_list=[]
        for frame_id in int_filter:
            key=video_id+'/'+frame_id
            if zero_dict.get(key) is not None:
                continue
            frames_list.append(frame_id)
        frames_dict[video_id]=frames_list
        frames_dict[video_id+'_label']=all_classes
    json.dump(frames_dict,open(json_savepath,'w'))

# trans all from the 'trans' jsonfile
def trans_all_json():
    except_list=['R4SJJ.mp4',
                 'FC2SK.mp4',
                 'C10FA.mp4',
                 'X2LBW.mp4',
                 'OZIJ7.mp4',
                 'LKH9A.mp4']
    except_name=','.join(except_list)
    file_dict=get_dict()
    zero_dict=get_frame_dict()
    trans_base='/home/wu_tian_ci/GAFL/data/trans'
    trans_train_paths=[os.path.join(trans_base,'train_'+str(i)+'.json') for i in range(1,6)]
    trans_test_paths=[os.path.join(trans_base,'test_'+str(i)+'.json') for i in range(1,6)]
    train_keys=[]
    test_keys=[]
    for test_path,train_path in zip(trans_train_paths,trans_test_paths):
        test_json=json.load(open(test_path,'r'))
        train_json=json.load(open(train_path,'r'))
        train_keys.append(train_json.keys())
        test_keys.append(test_json.keys())

    trans_json_path=[]
    json_save_base='GAFL/json_dataset/all/shift'

    anno_basepath='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'

    test_csv_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_test.csv'

    train_csv_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_train.csv'
    test_df=pd.read_csv(test_csv_path)
    train_df=pd.read_csv(train_csv_path)
    test_save_path=[os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all/shift','test'+str(i)+'.json') for i in range(1,6)]
    train_save_path=[os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all/shift','train'+str(i)+'.json') for i in range(1,6)]
    # video length
    test_lengths=test_df['length']
    # class
    test_actions=test_df['actions']
    # video id
    test_ids=test_df['id']



    train_lengths=train_df['length']
    # class
    train_actions=train_df['actions']
    # video id
    train_ids=train_df['id']

    pattern2 = r'c\d{3}'




    train_frames_dicts=[{} for i in range(5)]
    test_frames_dicts=[{} for i in range(5)]


    for _,l,a in tqdm(zip(test_ids,test_lengths,test_actions)):
        if _ in except_name:
            print(_)
            continue
        if file_dict.get(_) is None:
            continue
        try:
            all_classes=re.findall(pattern2,a)
        except:
            continue

        all_classes=[x[1:] for x in all_classes]

        video_id=_+'.mp4'
        filter_id=os.path.join(anno_basepath,video_id)
        try:
            filter_frames=os.listdir(filter_id)
        except:
            continue
        int_filter=natsorted(filter_frames)
        frames_list=[]
        for frame_id in int_filter:
            key=video_id+'/'+frame_id
            if zero_dict.get(key) is not None:
                continue
            frames_list.append(frame_id)
        for i in range(5):
            if _ in train_keys[i]:
                train_frames_dicts[i][video_id]=frames_list
                train_frames_dicts[i][video_id+'_label']=all_classes
            if _ in test_keys[i]:
                test_frames_dicts[i][video_id]=frames_list
                test_frames_dicts[i][video_id+'_label']=all_classes
    for _,l,a in tqdm(zip(train_ids,train_lengths,train_actions)):
        if _ in except_name:
            print(_)
            continue
        if file_dict.get(_) is None:
            continue
        try:
            all_classes=re.findall(pattern2,a)
        except:
            continue

        all_classes=[x[1:] for x in all_classes]

        video_id=_+'.mp4'
        filter_id=os.path.join(anno_basepath,video_id)
        try:
            filter_frames=os.listdir(filter_id)
        except:
            continue
        int_filter=natsorted(filter_frames)
        frames_list=[]
        for frame_id in int_filter:
            key=video_id+'/'+frame_id
            if zero_dict.get(key) is not None:
                continue
            frames_list.append(frame_id)
        for i in range(5):
            if _ in train_keys[i]:
                train_frames_dicts[i][video_id]=frames_list
                train_frames_dicts[i][video_id+'_label']=all_classes
            if _ in test_keys[i]:
                test_frames_dicts[i][video_id]=frames_list
                test_frames_dicts[i][video_id+'_label']=all_classes
    for i in range(5):
        json.dump(train_frames_dicts[i],open(train_save_path[i],'w'))
        json.dump(test_frames_dicts[i],open(test_save_path[i],'w'))
   

def generate_bbx_mask(name):
    json_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all',name+'.json')
    json_file=json.load(open(json_path,'r'))
    keys=json_file.keys()
    base_path='/home/wu_tian_ci/GAFL/json_dataset/all'
    object_bbox= pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
    )
    '''
    video_id:[[[x1,y1,x2,y2],...,[x1,y1,x1,y1]],
    [[x1,y1,x2,y2],...,[x1,y1,x1,y1]]](10 bbxes for every frames)
    '''
    bbx_dict={}
    '''
    video_id:[[0,0,1,...,1],...,[0,0,1,1,...,0]](10 objs)
    '''
    mask_dict={}

    cls_num=get_cls_num()
    for key in tqdm(keys):
        if 'label' in key:
            continue
        frame_ids=json_file[key]
        cls_list=[]
        for frame_id in frame_ids:
            vf_id=key+'/'+frame_id
            objs=object_bbox[vf_id]
            for obj in objs:
                if not obj['visible']:
                        continue
                cls_list.append(cls_num[obj['class'].replace('/','')])
            if person_bbox[vf_id]['bbox'].shape == (1,4):
                cls_list.append(cls_num['person'])
        cls_list=np.unique(cls_list)
        cls_list.sort()
        video_bbx_list=[]
        vidoe_mask_list=[]
        frames_len=len(frame_ids)
        for frame_id in frame_ids:
            vf_id=key+'/'+frame_id
            objs=object_bbox[vf_id]
            frame_bbx_dict={}
            mask_list=[]
            bbx_list=[]
            person_flag=False
            for obj in objs:
                if not obj['visible']:
                    continue
                frame_bbx_dict[cls_num[obj['class'].replace('/','')]]=obj['bbox']
            if person_bbox[vf_id]['bbox'].shape == (1,4):
                person_flag=True
                frame_bbx_dict[cls_num['person']]=person_bbox[vf_id]['bbox']
            for cls in cls_list:
                if frame_bbx_dict.get(cls) is None:
                    mask_list.append(0)
                    bbx_list.append([0.,0.,0.,0.])
                    continue
                bbx=frame_bbx_dict[cls]
                if type(bbx) == np.ndarray:
                    bbx=bbx.squeeze()
                    bbx=[float(bbx[0]),float(bbx[1]),float(bbx[2]),float(bbx[3])]
                else:
                    bbx=list(bbx)
                mask_list.append(1)
                bbx_list.append(bbx)
            if len(bbx_list) != len(mask_list):
                breakpoint()
            if len(bbx_list)>10 or len(mask_list)>10:
                print(key,frame_id)
            assert len(bbx_list) <= 10
            assert len(mask_list) <= 10
            bbx_list=bbx_list+[[0.,0.,0.,0.]]*(10-len(bbx_list))
            mask_list=mask_list+[0]*(10-len(mask_list))

            new_bbx=[(x[0],x[1],x[2]+x[0],x[3]+x[1]) for x in bbx_list]
            if person_flag: 
                new_bbx[0]=dp(bbx_list[0])
            video_bbx_list.append(new_bbx)
            vidoe_mask_list.append(mask_list)
        if frames_len!=len(video_bbx_list) or frames_len!=len(vidoe_mask_list):
            breakpoint()
        bbx_dict[key]=video_bbx_list
        mask_dict[key]=vidoe_mask_list
    mask_path=os.path.join(base_path,name+'_mask.json')
    bbx_path=os.path.join(base_path,name+'_bbx.json')
    json.dump(mask_dict,open(mask_path,'w'))
    json.dump(bbx_dict,open(bbx_path,'w'))

def extend_bbx(bbx,mask,bbx_size):
    if mask==0:
        return bbx
    # print(bbx)
    bbx[0]=max(0.,bbx[0]-10)
    bbx[1]=max(0.,bbx[1]-10)
    if bbx[0]>bbx_size[0]:
        bbx[0]=0.
    if bbx[1]>bbx_size[1]:
        bbx[1]=0.
    bbx[2]=min(bbx[2]+10,bbx_size[0])
    bbx[3]=min(bbx[3]+10,bbx_size[1])
    return bbx

def generate_bbx_mask_cls_rel(name):
    video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
    json_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all',name+'.json')
    json_file=json.load(open(json_path,'r'))
    keys=json_file.keys()
    base_path='/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel'
    object_bbox= pickle.load(
        open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
    )
    '''
    video_id:[[[x1,y1,x2,y2],...,[x1,y1,x1,y1]],
    [[x1,y1,x2,y2],...,[x1,y1,x1,y1]]](10 bbxes for every frames)
    '''
    bbx_dict={}
    '''
    video_id:[[0,0,1,...,1],...,[0,0,1,1,...,0]](10 objs)
    '''
    mask_dict={}

    '''
    video_id[[cls1,cls2,cls3,cls4...],[...]](10objs)
    '''
    obj_cls_dict={}
    '''
    video_id:[[rel1,rel2...],[]] 10 objs
    '''
    rel_dict={}
    cls_num=get_cls_num()
    people_flag,rel_count=False,0
    out_box_count=0
    for key in tqdm(keys):
        if 'label' in key:
            continue
        frame_ids=json_file[key]
        cls_list=[]
        people_flag_=False
        for frame_id in frame_ids:
            vf_id=key+'/'+frame_id
            objs=object_bbox[vf_id]
            for obj in objs:
                if not obj['visible']:
                        continue
                cls_list.append(cls_num[obj['class']])
            if person_bbox[vf_id]['bbox'].shape == (1,4):
                cls_list.append(cls_num['person'])
                people_flag_=True
        if not people_flag_:
            people_flag=True
        cls_list=np.unique(cls_list).tolist()
        cls_list.sort()
        cls_list.reverse()
        video_bbx_list=[]
        vidoe_mask_list=[]
        video_obj_cls_list=[]
        vidoe_rel_list=[]
        frames_len=len(frame_ids)
        for frame_id in frame_ids:
            vf_id=key+'/'+frame_id
            objs=object_bbox[vf_id]
            frame_bbx_dict={}
            frame_rel_dict={}
            mask_list=[]
            rel_list=[]
            bbx_list=[]
            obj_cls_list=[]
            person_flag=False
            for obj in objs:
                if not obj['visible']:
                    continue
                frame_bbx_dict[cls_num[obj['class']]]=obj['bbox']
                frame_rel_dict[cls_num[obj['class']]]=get_relation(obj['attention_relationship'],obj['spatial_relationship'],obj['contacting_relationship'])

            if person_bbox[vf_id]['bbox'].shape == (1,4):
                person_flag=True
                frame_bbx_dict[cls_num['person']]=person_bbox[vf_id]['bbox']
                frame_rel_dict[cls_num['person']]=[0]
            for cls in cls_list:
                if frame_bbx_dict.get(cls) is None:
                    mask_list.append(0)
                    bbx_list.append([0.,0.,0.,0.])
                    obj_cls_list.append(0)
                    rel_list.append([0])
                    continue
                bbx=frame_bbx_dict[cls]
                if type(bbx) == np.ndarray:
                    bbx=bbx.squeeze()
                    bbx=[float(bbx[0]),float(bbx[1]),float(bbx[2]),float(bbx[3])]
                else:
                    bbx=list(bbx)
                mask_list.append(1)
                bbx_list.append(bbx)
                obj_cls_list.append(cls)
                rel_list.append(frame_rel_dict[cls])
            if len(bbx_list) != len(mask_list):
                breakpoint()
            if len(bbx_list)>10 or len(mask_list)>10:
                print(key,frame_id)
            if not people_flag_:
                bbx_list.insert(0,[0.,0.,0.,0.])
                mask_list.insert(0,0)
                obj_cls_list.insert(0,0)
                rel_list.insert(0,[0])
            if len(bbx_list)>10:
                print(bbx_list,key,cls_list)
            assert len(bbx_list) <= 10
            assert len(mask_list) <= 10
            assert len(obj_cls_list) <= 10
            assert len(bbx_list)==len(rel_list)
            bbx_list=bbx_list+[[0.,0.,0.,0.]]*(10-len(bbx_list))
            mask_list=mask_list+[0]*(10-len(mask_list))
            obj_cls_list=obj_cls_list+[0]*(10-len(obj_cls_list))
            rel_list=rel_list+[[0]]*(10-len(rel_list))

            new_bbx=[[x[0],x[1],x[2]+x[0],x[3]+x[1]] for x in bbx_list]
            # for bx in new_bbx:
            #     if (bx[0]>bx[2]) or (bx[1]>bx[3]):
            #         breakpoint()
            # if key=='1K0SU.mp4':
            #     print(new_bbx)
            #     breakpoint()
            if person_flag: 
                new_bbx[0]=dp(bbx_list[0])
            
            video_size=video2size[key.split('.')[0]]
            new_bbx2=[extend_bbx(bx,mk,video_size) for bx,mk in zip(new_bbx,mask_list)]
            video_bbx_list.append(new_bbx2)
            vidoe_mask_list.append(mask_list)
            video_obj_cls_list.append(obj_cls_list)
            vidoe_rel_list.append(rel_list)
            for bx in new_bbx2:
                if (bx[0]>bx[2]) or (bx[1]>bx[3]):
                    print(key,frame_id,obj_cls_list,new_bbx2)
                    out_box_count+=1
        if frames_len!=len(video_bbx_list) or frames_len!=len(vidoe_mask_list):
            breakpoint()

        bbx_dict[key]=video_bbx_list
        mask_dict[key]=vidoe_mask_list
        obj_cls_dict[key]=video_obj_cls_list
        rel_dict[key]=vidoe_rel_list
    mask_path=os.path.join(base_path,name+'_mask.json')
    bbx_path=os.path.join(base_path,name+'_bbx.json')
    obj_cls_path=os.path.join(base_path,name+'_obj_cls.json')
    rel_path=os.path.join(base_path,name+'_rel.json')
    json.dump(mask_dict,open(mask_path,'w'))
    json.dump(bbx_dict,open(bbx_path,'w'))
    json.dump(obj_cls_dict,open(obj_cls_path,'w'))
    json.dump(rel_dict,open(rel_path,'w'))
    print(out_box_count)


def generate_bbx_mask_cls_rel_shift(name):
    video2size=json.load(open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/video2size/ag.json','r'))
    json_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/all/shift',name+'.json')
    json_file=json.load(open(json_path,'r'))
    keys=json_file.keys()
    base_path='/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel/shift'
    object_bbox= pickle.load(
        open('/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
    )
    '''
    video_id:[[[x1,y1,x2,y2],...,[x1,y1,x1,y1]],
    [[x1,y1,x2,y2],...,[x1,y1,x1,y1]]](10 bbxes for every frames)
    '''
    bbx_dict={}
    '''
    video_id:[[0,0,1,...,1],...,[0,0,1,1,...,0]](10 objs)
    '''
    mask_dict={}

    '''
    video_id[[cls1,cls2,cls3,cls4...],[...]](10objs)
    '''
    obj_cls_dict={}
    '''
    video_id:[[rel1,rel2...],[]] 10 objs
    '''
    rel_dict={}
    cls_num=get_cls_num()
    people_flag,rel_count=False,0
    out_box_count=0
    for key in tqdm(keys):
        if 'label' in key:
            continue
        frame_ids=json_file[key]
        cls_list=[]
        people_flag_=False
        for frame_id in frame_ids:
            vf_id=key+'/'+frame_id
            objs=object_bbox[vf_id]
            for obj in objs:
                if not obj['visible']:
                        continue
                cls_list.append(cls_num[obj['class']])
            if person_bbox[vf_id]['bbox'].shape == (1,4):
                cls_list.append(cls_num['person'])
                people_flag_=True
        if not people_flag_:
            people_flag=True
        cls_list=np.unique(cls_list).tolist()
        cls_list.sort()
        cls_list.reverse()
        video_bbx_list=[]
        vidoe_mask_list=[]
        video_obj_cls_list=[]
        vidoe_rel_list=[]
        frames_len=len(frame_ids)
        for frame_id in frame_ids:
            vf_id=key+'/'+frame_id
            objs=object_bbox[vf_id]
            frame_bbx_dict={}
            frame_rel_dict={}
            mask_list=[]
            rel_list=[]
            bbx_list=[]
            obj_cls_list=[]
            person_flag=False
            for obj in objs:
                if not obj['visible']:
                    continue
                frame_bbx_dict[cls_num[obj['class']]]=obj['bbox']
                frame_rel_dict[cls_num[obj['class']]]=get_relation(obj['attention_relationship'],obj['spatial_relationship'],obj['contacting_relationship'])

            if person_bbox[vf_id]['bbox'].shape == (1,4):
                person_flag=True
                frame_bbx_dict[cls_num['person']]=person_bbox[vf_id]['bbox']
                frame_rel_dict[cls_num['person']]=[0]
            for cls in cls_list:
                if frame_bbx_dict.get(cls) is None:
                    mask_list.append(0)
                    bbx_list.append([0.,0.,0.,0.])
                    obj_cls_list.append(0)
                    rel_list.append([0])
                    continue
                bbx=frame_bbx_dict[cls]
                if type(bbx) == np.ndarray:
                    bbx=bbx.squeeze()
                    bbx=[float(bbx[0]),float(bbx[1]),float(bbx[2]),float(bbx[3])]
                else:
                    bbx=list(bbx)
                mask_list.append(1)
                bbx_list.append(bbx)
                obj_cls_list.append(cls)
                rel_list.append(frame_rel_dict[cls])
            if len(bbx_list) != len(mask_list):
                breakpoint()
            if len(bbx_list)>10 or len(mask_list)>10:
                print(key,frame_id)
            if not people_flag_:
                bbx_list.insert(0,[0.,0.,0.,0.])
                mask_list.insert(0,0)
                obj_cls_list.insert(0,0)
                rel_list.insert(0,[0])
            if len(bbx_list)>10:
                print(bbx_list,key,cls_list)
            assert len(bbx_list) <= 10
            assert len(mask_list) <= 10
            assert len(obj_cls_list) <= 10
            assert len(bbx_list)==len(rel_list)
            bbx_list=bbx_list+[[0.,0.,0.,0.]]*(10-len(bbx_list))
            mask_list=mask_list+[0]*(10-len(mask_list))
            obj_cls_list=obj_cls_list+[0]*(10-len(obj_cls_list))
            rel_list=rel_list+[[0]]*(10-len(rel_list))

            new_bbx=[[x[0],x[1],x[2]+x[0],x[3]+x[1]] for x in bbx_list]
            # for bx in new_bbx:
            #     if (bx[0]>bx[2]) or (bx[1]>bx[3]):
            #         breakpoint()
            # if key=='1K0SU.mp4':
            #     print(new_bbx)
            #     breakpoint()
            if person_flag: 
                new_bbx[0]=dp(bbx_list[0])
            
            video_size=video2size[key.split('.')[0]]
            new_bbx2=[extend_bbx(bx,mk,video_size) for bx,mk in zip(new_bbx,mask_list)]
            video_bbx_list.append(new_bbx2)
            vidoe_mask_list.append(mask_list)
            video_obj_cls_list.append(obj_cls_list)
            vidoe_rel_list.append(rel_list)
            for bx in new_bbx2:
                if (bx[0]>bx[2]) or (bx[1]>bx[3]):
                    print(key,frame_id,obj_cls_list,new_bbx2)
                    out_box_count+=1
        if frames_len!=len(video_bbx_list) or frames_len!=len(vidoe_mask_list):
            breakpoint()

        bbx_dict[key]=video_bbx_list
        mask_dict[key]=vidoe_mask_list
        obj_cls_dict[key]=video_obj_cls_list
        rel_dict[key]=vidoe_rel_list
    mask_path=os.path.join(base_path,name+'_mask.json')
    bbx_path=os.path.join(base_path,name+'_bbx.json')
    obj_cls_path=os.path.join(base_path,name+'_obj_cls.json')
    rel_path=os.path.join(base_path,name+'_rel.json')
    json.dump(mask_dict,open(mask_path,'w'))
    json.dump(bbx_dict,open(bbx_path,'w'))
    json.dump(obj_cls_dict,open(obj_cls_path,'w'))
    json.dump(rel_dict,open(rel_path,'w'))
    print(out_box_count)


# mapping table 

'''
id:name
tokens:
private:
common:
'''


def get_sperate_labels(name,labels,full_list):
    labels=list(set(labels))
    token_list=[int(i) for i in labels]

    contra_set=set(full_list)-set(token_list)

    token_set=set(token_list)
    contra_list=list(contra_set)
    
    list_len=len(token_list)

    random_l=random.sample(contra_list,1)

    label_list=[{'id':name,'token':token_list,'private':[157],'common':token_list},
                {'id':name,'token':random_l,'common':[157],'private':token_list}]
    for i in range(1,list_len):
        t_list=random.sample(token_list,i)
        t_set=set(t_list)
        p_list=list(token_set-t_set)
        if i == list_len:
            p_list=[157]
        
        label_list.append({
            'id':name,
            'token':t_list,
            'private':p_list,
            'common':t_list
        })
    return label_list


def get_sperate_labels2(name,labels,full_list):
    labels=list(set(labels))
    token_list=[int(i) for i in labels]

    contra_set=set(full_list)-set(token_list)

    token_set=set(token_list)
    contra_list=list(contra_set)
    
    list_len=len(token_list)

    # random_l=random.sample(contra_list,1)
    label_list=[]
    for i in range(1,list_len+1):
        t_list=random.sample(token_list,i)
        t_set=set(t_list)
        p_list=list(token_set-t_set)
        if i==list_len:
            p_list=[157]
        # all right
        label_list.append({
            'id':name,
            'token':t_list,
            'private':p_list,
            'common':t_list
        })
        # all wrong
        t_list=random.sample(contra_list,i)
        label_list.append({
            'id':name,
            'token':t_list,
            'private':token_list[:],
            'common':[157]
        })
    return label_list

# flag true -> all right token 
def get_sperate_labels3(name,labels,full_list,flag=False):
    labels=list(set(labels))
    token_list=[int(i) for i in labels]

    contra_set=set(full_list)-set(token_list)

    token_set=set(token_list)
    contra_list=list(contra_set)
    
    list_len=len(token_list)

    # random_l=random.sample(contra_list,1)
    label_list=[]
    for i in range(1,list_len+1):
        t_list=random.sample(token_list,i)
        t_set=set(t_list)
        p_list=list(token_set-t_set)
        if i==list_len:
            p_list=[157]
        # all right
        if flag:
            label_list.append({
                'id':name,
                'token':t_list,
                'private':p_list,
                'common':t_list
            })
        # all wrong
        else:
            t_list=random.sample(contra_list,i)
            label_list.append({
                'id':name,
                'token':t_list,
                'private':token_list[:],
                'common':[157]
            })
    return label_list

# padding to 16 tokens
def get_sperate_labels4(name,labels,full_list):
    labels=list(set(labels))
    token_list=[int(i) for i in labels]

    contra_set=set(full_list)-set(token_list)

    token_set=set(token_list)
    contra_list=list(contra_set)
    
    list_len=len(token_list)
    max_labels=16

    label_list=[]
    for i in range(1,list_len+1):
        t_list=random.sample(token_list,i)
        
        t_list_con=random.sample(contra_list,max_labels-i)
        t_list_token=t_list+t_list_con
        t_set=set(t_list)
        p_list=list(token_set-t_set)
        if len(p_list)==0:
            p_list=[157]
        label_list.append({
            'id':name,
            'token':t_list_token,
            'private':p_list,
            'common':t_list
        })
    return label_list

def get_sperate_labels_expand(name,labels,full_list):
    labels=list(set(labels))
    token_list=[int(i) for i in labels]

    contra_set=set(full_list)-set(token_list)

    token_set=set(token_list)
    contra_list=list(contra_set)
    
    list_len=len(token_list)
    max_labels=16
    random_l=random.sample(contra_list,max_labels)
    e_token_list=token_list+random.sample(contra_list,max_labels-len(token_list))
    label_list=[{'id':name,'token':e_token_list,'private':[157],'common':token_list},
                {'id':name,'token':random_l,'common':[157],'private':token_list}]
    for i in range(1,list_len):
        t_list=random.sample(token_list,i)
        
        t_list_con=random.sample(contra_list,max_labels-i)
        t_list_token=t_list+t_list_con
        t_set=set(t_list)
        p_list=list(token_set-t_set)
        
        label_list.append({
            'id':name,
            'token':t_list_token,
            'private':p_list,
            'common':t_list
        })
    return label_list

def get_sperate_labels_constrain(name,labels,full_list,sample_tokens,padding):
    labels=list(labels)
    token_list=[int(i) for i in labels]
    # labels_len=len(labels)
    contra_set=set(full_list)-set(token_list)

    token_set=set(token_list)
    contra_list=list(contra_set)

    max_labels=16
    if sample_tokens==0:
        if padding:
            return {'id':name,'token':random.sample(contra_list,max_labels),'private':token_list,'common':[157]}
        else:
            return {'id':name,'token':random.sample(contra_list,1),'private':token_list,'common':[157]}
    if sample_tokens==16:
        return {'id':name,'token':token_list,'private':[157],'common':token_list}

    token_ans=random.sample(token_list,sample_tokens)
    token_=dp(token_ans)
    p_token=list(token_set-set(token_ans))
    if len(p_token)==0:
        p_token=[157]
    if padding:
        token_=token_+random.sample(contra_list,max_labels-sample_tokens)
    
    return {'id':name,'token':token_,'private':p_token,'common':token_ans}

def mapping_table(name,label_type=1):
    full_list=[i for i in range(157)]
    json_file=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
    keys=list(json_file.keys())
    if label_type==1:
        json_save_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping',name+'.json')
    else:
        json_save_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping2',name+'.json')
    # key2=list(json_file2.keys())
    json_keys=[item for item in keys if 'label' not in item]
    ans_list=[]
    len_a=[]
    for k in tqdm(json_keys):
        label_name=k+'_label'
        json_set=set(json_file[label_name])
        len_a.append(len(list(json_set)))
        if len(json_set)>16:
            continue
        if label_type==1:
            l=get_sperate_labels(k,json_file[label_name],full_list)
        else:
            l=get_sperate_labels2(k,json_file[label_name],full_list)
        ans_list.extend(l)
    print(name,':',len(ans_list),' ',len(keys),' ',np.mean(len_a),' ',np.sum(len_a),' ',len(len_a))
    json.dump(ans_list,open(json_save_path,'w'))

def mapping_table_test(name,label_type=1):
    full_list=[i for i in range(157)]
    json_file=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
    keys=list(json_file.keys())
    json_save_pathss=[[['' for j in range(i+1)] for i in range(16)] for i in range(3)]
    json_save_paths=[['' for j in range(i+1)] for i in range(16)]
    jsp_=''
    ans_list=[[[] for j in range(i+1)] for i in range(16)]
    ans_=[]
    if label_type==4:
        ans_list=[[[[] for j in range(i+1)] for i in range(16)],
                  [[[] for j in range(i+1)] for i in range(16)],
                  [[[] for j in range(i+1)] for i in range(16)]]
    if label_type==1:
        json_base_path='/home/wu_tian_ci/GAFL/json_dataset/mapping_test/type_1'
    elif label_type==2:
        json_base_path='/home/wu_tian_ci/GAFL/json_dataset/mapping_test/type_2'
    elif label_type==3:
        json_base_path='/home/wu_tian_ci/GAFL/json_dataset/mapping_test/type_3'
    elif label_type==4:
        json_base_paths=[
            '/home/wu_tian_ci/GAFL/json_dataset/mapping_test/type_1',
            '/home/wu_tian_ci/GAFL/json_dataset/mapping_test/type_2',
            '/home/wu_tian_ci/GAFL/json_dataset/mapping_test/type_3'
        ]
    elif label_type ==5:
        json_base_path='/home/wu_tian_ci/GAFL/json_dataset/mapping_test/type_4'
    elif label_type ==6:
        json_base_path='/home/wu_tian_ci/GAFL/json_dataset/mapping_test/type_5'
    else:
        raise NotImplementedError
    if label_type in [1,2,3,5]:
        for i in range(1,17):
            for j in range(1,i+1):
                c_dir=os.path.join(json_base_path,str(i),str(j))
                if not os.path.exists(c_dir):
                    os.makedirs(c_dir)
                json_save_paths[i-1][j-1]=os.path.join(c_dir,name+'.json')
    elif label_type in [6]:
        if not os.path.exists(json_base_path):
            os.makedirs(json_base_path)
        jsp_=os.path.join(json_base_path,name+'.json')
    elif label_type in [4]:
        for i in range(1,17):
            for j in range(1,i+1):
                    for k in range(3):
                        c_dir=os.path.join(json_base_paths[k],str(i),str(j))
                        if not os.path.exists(c_dir):
                            os.makedirs(c_dir)
                        json_save_pathss[k][i-1][j-1]=os.path.join(c_dir,name+'.json')
    
    json_keys=[item for item in keys if 'label' not in item]
    len_a=[]
    for k in tqdm(json_keys):
        label_name=k+'_label'
        json_set=set(json_file[label_name])
        len_a.append(len(list(json_set)))
        tokens_len=len(json_set)
        if len(json_set)>16:
            continue


        if label_type==1:
            l=get_sperate_labels3(k,json_file[label_name],full_list,True)
        elif label_type==2:
            l=get_sperate_labels3(k,json_file[label_name],full_list,False)
        elif label_type==3:
            l1=get_sperate_labels3(k,json_file[label_name],full_list,True)
            l2=get_sperate_labels3(k,json_file[label_name],full_list,False)
        elif label_type==4:
            l1=get_sperate_labels3(k,json_file[label_name],full_list,True)
            l2=get_sperate_labels3(k,json_file[label_name],full_list,False)
        elif label_type==5:
            l=get_sperate_labels4(k,json_file[label_name],full_list)
        elif label_type==6:
            l1=get_sperate_labels3(k,json_file[label_name],full_list,True)
            l2=get_sperate_labels3(k,json_file[label_name],full_list,False)
            l3=get_sperate_labels4(k,json_file[label_name],full_list)
        else:
            raise NotImplementedError
        # breakpoint()
        for i in range(tokens_len):
            if label_type in [1,2,5]:
                ans_list[tokens_len-1][i].append(l[i])
            elif label_type in [3]:
                ans_list[tokens_len-1][i].append(l1[i])
                ans_list[tokens_len-1][i].append(l2[i])
            elif label_type in [4]:
                ans_list[0][tokens_len-1][i].append(l1[i])
                ans_list[1][tokens_len-1][i].append(l2[i])
                ans_list[2][tokens_len-1][i].append(l1[i])
                ans_list[2][tokens_len-1][i].append(l2[i])
            elif label_type in [6]:
                ans_.append(l1[i])
                ans_.append(l2[i])
                ans_.append(l3[i])
    # print(name,':',len(ans_list),' ',len(keys),' ',np.mean(len_a),' ',np.sum(len_a),' ',len(len_a))
    if label_type in [1,2,3,5]:
        for i in range(16):
            for j in range(i+1):
                # breakpoint()
                json.dump(ans_list[i][j],open(json_save_paths[i][j],'w'))
    elif label_type in [4]:
        for i in range(16):
            for j in range(i+1):
                # breakpoint()
                json.dump(ans_list[0][i][j],open(json_save_pathss[0][i][j],'w'))
                json.dump(ans_list[1][i][j],open(json_save_pathss[1][i][j],'w'))
                json.dump(ans_list[2][i][j],open(json_save_pathss[2][i][j],'w'))
    elif label_type in [6]:
        json.dump(ans_,open(jsp_,'w'))

def tally_label(name):
    full_list=[i for i in range(157)]
    json_file=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
    keys=list(json_file.keys())
    json_save_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping',name+'.json')
    # key2=list(json_file2.keys())
    json_keys=[item for item in keys if 'label' not in item]
    ans_list=[]
    len_a=[]
    len_label=[0 for i in range(28)]
    for k in tqdm(json_keys):
        label_name=k+'_label'
        json_set=set(json_file[label_name])
        label_len=len(json_set)
        len_label[label_len-1]+=1
        len_a.append(label_len)
    for i in range(27):
        len_label[i+1]+=len_label[i]
    sum_len=sum(len_a)
    for i in range(28):
        print(round(len_label[i]/sum_len,2),end=' ')

def mapping_table_expand(name):
    full_list=[i for i in range(157)]
    json_file=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
    keys=list(json_file.keys())
    json_save_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping_expand2',name+'.json')
    # key2=list(json_file2.keys())
    json_keys=[item for item in keys if 'label' not in item]
    ans_list=[]
    len_a=[]
    for k in tqdm(json_keys):
        label_name=k+'_label'
        json_set=set(json_file[label_name])
        len_a.append(len(list(json_set)))
        l=get_sperate_labels_expand(k,json_file[label_name],full_list)
        ans_list.extend(l)
    print(name,':',len(ans_list),' ',len(keys),' ',np.mean(len_a),' ',np.sum(len_a),' ',len(len_a))
    json.dump(ans_list,open(json_save_path,'w'))

def mapping_table_expand2(name):
    full_list=[i for i in range(157)]
    json_file=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
    keys=list(json_file.keys())
    json_save_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping_expand2',name+'.json')
    # key2=list(json_file2.keys())
    json_keys=[item for item in keys if 'label' not in item]
    ans_list=[]
    len_a=[]
    for k in tqdm(json_keys):
        label_name=k+'_label'
        json_set=set(json_file[label_name])
        if len(json_set)>16:
            continue
        len_a.append(len(list(json_set)))
        l=get_sperate_labels_expand(k,json_file[label_name],full_list)
        l2=get_sperate_labels(k,json_file[label_name],full_list)
        ans_list.extend(l)
        ans_list.extend(l2)

    print(name,':',len(ans_list),' ',len(keys),' ',np.mean(len_a),' ',np.sum(len_a),' ',len(len_a))
    json.dump(ans_list,open(json_save_path,'w'))

def mapping_table_expand3(name):
    full_list=[i for i in range(157)]
    json_file=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
    keys=list(json_file.keys())
    json_save_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping_expand3',name+'.json')
    # key2=list(json_file2.keys())
    json_keys=[item for item in keys if 'label' not in item]
    ans_list=[]
    len_a=[]
    for k in tqdm(json_keys):
        label_name=k+'_label'
        json_set=set(json_file[label_name])
        if len(json_set)>16:
            continue
        len_a.append(len(list(json_set)))
        l=get_sperate_labels_expand(k,json_file[label_name],full_list)
        # l2=get_sperate_labels(k,json_file[label_name],full_list)
        ans_list.extend(l)
        # ans_list.extend(l2)

    print(name,':',len(ans_list),' ',len(keys),' ',np.mean(len_a),' ',np.sum(len_a),' ',len(len_a))
    json.dump(ans_list,open(json_save_path,'w'))

def mapping_table_seperate(name,max_token,min_token,padding):
    
    full_list=[i for i in range(157)]
    json_file=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
    keys=list(json_file.keys())
    types='type_2' if padding else 'type_1'
    json_save_paths=[os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping_seperate',types,name+str(i)+'.json') for i in range(min_token,max_token+1)]
    # key2=list(json_file2.keys())
    json_keys=[item for item in keys if 'label' not in item]
    ans_list=[[] for i in range(max_token-min_token+1)]

    for k in tqdm(json_keys):
        label_name=k+'_label'
        json_set=set(json_file[label_name])
        if len(json_set)>16:
            continue
        # len_a.append(len(list(json_set)))
        for tokens in range(min_token,max_token+1):
            if tokens > len(json_set):
                break
            l=get_sperate_labels_constrain(k,json_set,full_list,tokens,padding)
            ans_list[tokens-min_token].append(l)

    for i in range(max_token-min_token+1):
        json.dump(ans_list[i],open(json_save_paths[i],'w'))

def tally_duplicate(csv_name):  

    if csv_name=='test':
        csv_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_test.csv'
    else:
        csv_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_train.csv'
    df=pd.read_csv(csv_path)
    lengths=df['length']
    actions=df['actions']
    ids=df['id']
    pattern1=r'\b\d+\.\d+\b'
    pattern2 = r'c\d{3}'
    counts=0

    positive_count=0
    total_cls=0
    total_dup=0
    dup_dict={}
    dup_dict1={}
    dup_dict2={}
    for _,l,a in zip(ids,lengths,actions):
        try:
            all_numbers=re.findall(pattern1,a)
            all_classes=re.findall(pattern2,a)
        except:
            # print('error',l,a,_)
            continue
        float_number=[float(X) for X in all_numbers]
        positive_count+=1
        max_float=max(float_number)
        if max_float > float(l):
            counts+=1
            # print(max_float,' ',float(l))

        total_cls+=len(all_classes)
        
            
        def dup_fun(fn,skip_list=None):
            cls_d=[0 for i in range(len(all_classes))]
            dup_list=torch.zeros(len(cls_d),dtype=torch.float32)
            for i in range(0,len(fn),2):
                if skip_list is not None and i//2 in skip_list:
                    continue
                cls1_begin,cls1_end=fn[i],fn[i+1]
                for j in range(i+2,len(fn),2):
                    if skip_list is not None and j//2 in skip_list:
                        continue
                    if i==j:
                        continue
                    cls2_begin,cls2_end=fn[j],fn[j+1]
                    max_begin=max(cls1_begin,cls2_begin)
                    min_end=min(cls1_end,cls2_end)
                    dup=min_end-max_begin
                    if dup <=0:
                        continue
                    cls_d[i//2]=max(cls_d[i//2],math.ceil(dup/(cls1_end-cls1_begin)*100))
                    cls_d[j//2]=max(cls_d[j//2],math.ceil(dup/(cls2_end-cls2_begin)*100))
                    dup_list[i//2]+=dup/(cls1_end-cls1_begin)
                    dup_list[j//2]+=dup/(cls2_end-cls2_begin)
            return cls_d,dup_list


        cls_dupulicate,dup_ilst=dup_fun(float_number)
        _,top1_skip=torch.topk(dup_ilst,1)
        top1_skip=top1_skip.tolist()
        cls_dup1,dup_ilst=dup_fun(float_number,top1_skip)
        if len(dup_ilst)>1:
            _,top2_skip=torch.topk(dup_ilst,2)
            top2_skip=top2_skip.tolist()
            cls_dup2,dup_ilst=dup_fun(float_number,top2_skip)
        else:
            cls_dup2,dup_ilst=dup_fun(float_number,top1_skip)
        # for i in range(0,len(float_number),2):
        #     cls1_begin,cls1_end=float_number[i],float_number[i+1]
        #     for j in range(i*2,len(float_number),2):
        #         if i==j:
        #             continue
        #         cls2_begin,cls2_end=float_number[j],float_number[j+1]
        #         max_begin=max(cls1_begin,cls2_begin)
        #         min_end=min(cls1_end,cls2_end)
        #         dup=min_end-max_begin
        #         if dup <=0:
        #             continue
        #         cls_dupulicate[i//2]=max(cls_dupulicate[i//2],math.ceil(dup/(cls1_end-cls1_begin)*100))
        #         cls_dupulicate[j//2]=max(cls_dupulicate[j//2],math.ceil(dup/(cls2_end-cls2_begin)*100))


        for prob in cls_dupulicate:
            if prob==0:
                continue
            if dup_dict.get(prob) is None:
                dup_dict[prob]=1
            else:
                dup_dict[prob]+=1

        for prob in cls_dup1:
            if prob==0:
                continue  
            if dup_dict1.get(prob) is None:
                dup_dict1[prob]=1
            else:
                dup_dict1[prob]+=1 

        for prob in cls_dup2:
            if prob==0:
                continue
            if dup_dict2.get(prob) is None:
                dup_dict2[prob]=1
            else:
                dup_dict2[prob]+=1
    probs=[0 for i in range(10)]
    probs1=[0 for i in range(10)]
    probs2=[0 for i in range(10)]

    dup_value=0
    dup_value1=0
    dup_value2=0

    for key,value in dup_dict.items():
        dup_value+=value
        if key>=100:

            probs[-1]+=value
        else:
            probs[int(key/10.)]+=value
    probs=[x/total_cls for x in probs]

    for key,value in dup_dict1.items():
        dup_value1+=value
        if key>=100:
            probs1[-1]+=value
        else:
            probs1[int(key/10.)]+=value
    probs1=[x/total_cls for x in probs1]

    for key,value in dup_dict2.items():
        dup_value2+=value
        if key>=100:
            probs2[-1]+=value
        else:
            probs2[int(key/10.)]+=value
    probs2=[x/total_cls for x in probs2]

    print('range(%) \t all \t top 1\t top2')
    for i in range(10):
        print(i*10,'~',(i+1)*10,':\t',round(probs[i]*100,2),'\t',round(probs1[i]*100,2),'\t',round(probs2[i]*100,2))

    print('片段个数','\t',dup_value,'\t',dup_value1,'\t',dup_value2)

def cut_2():
    base_path='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/action_genome/frames'
    videos=os.listdir(base_path)
    object_bbox= pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/object_bbox_and_relationship.pkl',"rb")
    )
    person_bbox = pickle.load(
        open('/home/wu_tian_ci/GAFL/data/action_genome/person_bbox.pkl', "rb")
    )
    person_dir='/home/wu_tian_ci/GAFL/data/cut_png/person2'
    obj_dir='/home/wu_tian_ci/GAFL/data/cut_png/obj2'
    import random
    random.shuffle(videos)
    videos=videos[:10]
    for idx,video in enumerate(videos):
        pngs_dirs=os.listdir(os.path.join(base_path,video))
        random.shuffle(pngs_dirs)
        for png_name in pngs_dirs:
            objs=object_bbox[video+'/'+png_name]
            count=0
            img=Image.open(os.path.join(base_path,video,png_name))
            for iddx,obj in enumerate(objs):
                if obj['visible']:
                    obbx=obj['bbox']
                    f_bbx=[obbx[0],obbx[1],obbx[0]+obbx[2],obbx[1]+obbx[3]]
                    obj_crop=img.crop(tuple(f_bbx))
                    obj_crop.save(os.path.join(obj_dir,video.split('.')[0]+png_name.split('.')[0]+str(iddx)+obj['class'].replace('/','_')+'.png'))
                if person_bbox[video+'/'+png_name]['bbox'].shape==(1,4):
                    pbbx=person_bbox[video+'/'+png_name]['bbox'][0]
                    person_img=img.crop(tuple(pbbx))
                    person_img.save(os.path.join(person_dir,video.split('.')[0]+png_name.split('.')[0]+str(iddx)+'person.png'))
   
def label_compute(name):
    full_list=[i for i in range(157)]
    json_file=json.load(
            open(os.path.join("/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel",name+'.json'),'r')
        )
    keys=list(json_file.keys())
    json_save_path=os.path.join('/home/wu_tian_ci/GAFL/json_dataset/mapping',name+'.json')
    # key2=list(json_file2.keys())
    json_keys=[item for item in keys if 'label' not in item]
    ans_list=[]
    label_list=[0 for i in range(28)]
    for k in tqdm(json_keys):
        label_name=k+'_label'
        json_list=list(set(json_file[label_name]))
        len_j=len(json_list)
        label_list[len_j-1]+=1
    label_list2=0
    sum_=sum(label_list)
    label_list2=dp(label_list)
    for i in range(15,-1,-1):
        label_list2[i]+=label_list2[i+1]
    for i in range(16):
        print('i:',i+1,round(label_list2[i]/sum_,3),label_list2[i],end=',')
    # print('')
        if (i%5==0) and (i!=0):
            print('')
    print('')
    xx=[i for i in range(1,17)]
    y=label_list[:16]
    draw_list_multi([y],xx,'video_nums',['video num'])

        
import argparse
if __name__=="__main__":
    # print_box()
    parser = argparse.ArgumentParser(description="Packs PIL images as HDF5.")
    parser.add_argument(
        "--type",
        type=str,
        default="test",
        help="test.json or train.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="gpu device",
    )

    # # trans_('train','train')
    # # trans_('test','test')
    args = parser.parse_args()
    # cls_process(args.type)
    # print_ex()
    # cut_test()
    # print('train')
    # print_ex('train')
    # print('test')
    # print_ex('test')
    # cls_dict=get_cls_num()
    # print(len(cls_dict.keys()))
    # label_compute('train')
    # label_compute('test')
    # mapping_table('test')
    # mapping_table('train')


    # mapping_table_seperate('test',16,0,True)
    # mapping_table_seperate('test',16,0,False)

    # label_compute('test')
    # mapping_table('train',2)
    # mapping_table('test',2)

    # mapping_table_test('test',1)
    # mapping_table_test('test',2)
    set_seed()
    # mapping_table_test('test',6)
    # trans_all_json()

    # for i in range(1,6):
    #     generate_bbx_mask_cls_rel_shift('test'+str(i))
    #     generate_bbx_mask_cls_rel_shift('train'+str(i))
    for i in range(1,6):
        pkl_process_all_cls_rel_shift('test'+str(i),'cuda:1')
        pkl_process_all_cls_rel_shift('train'+str(i),'cuda:1')
    # tally_bbx_in_video()
    # generate_bbx_mask(args.type)
    # trans_all(args.type,args.type)
    # pkl_process_all(args.type,args.device)
    # tally_duplicate(args.type)
    # cut_2()

    # mapping_table_expand2('train')
    # mapping_table_expand2('test')
    
    # pkl_process_all_cls_rel('train','cuda:0')
    # pkl_process_all_cls_rel('test','cuda:0')

    # print_box()


