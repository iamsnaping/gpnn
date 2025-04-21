import cv2
import pandas as pd
from data_utils import (sample_appearance_indices)
import re
import json
import os
from tqdm import tqdm
'''

{video_id:{
token:frames
token:frames
}}

'''


'''
indices = sample_appearance_indices(
    self.sample_rate, len(frame_ids),self.train 
)
'''
def get_frame_dict(df):
    except_list=['R4SJJ.mp4',
                 'FC2SK.mp4',
                 'C10FA.mp4',
                 'X2LBW.mp4',
                 'OZIJ7.mp4',
                 'LKH9A.mp4']
    pattern1=r'\b\d+\.\d+\b'
    pattern2 = r'c\d{3}'
    lengths=df['length']
    actions=df['actions']
    ids=df['id']
    json_path='/home/wu_tian_ci/GAFL/json_dataset/all_cls_rel/test.json'
    vbp='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_480'
    js=json.load(open(json_path,'r'))
    allkeys=js.keys()
    keys=[item for item in allkeys if 'label' not in item]
    frames_dict={}
    for _,l,a in tqdm(zip(ids,lengths,actions)):
        key=_+'.mp4'
        if key not in keys:
            continue
        if key in except_list:
            continue
        try:
            all_numbers=re.findall(pattern1,a)
            all_classes=re.findall(pattern2,a)
        except:
            # print('error',l,a,_)
            continue
        frame_dict={}
        v_path=os.path.join(vbp,key)
        cap = cv2.VideoCapture(v_path)  # 如果是摄像头，用 `0` 或 `1` 代替路径

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        frame_list=js[key]
        frame_list=sorted(frame_list,key=lambda x:int(x[:-4]))
        float_number=[int(x[:-4]) for x in frame_list]
        float_number=sorted(float_number)
        indices=sample_appearance_indices(16,len(frame_list),False)
        try:
            float_number_selected=[float_number[i] for i in indices]
        except:
            breakpoint()
        assert len(float_number_selected)==16
        for i in range(0,len(all_numbers),2):
            token=int(all_classes[i//2][1:])
            try:
                begin_=float(all_numbers[i])*fps
                end_=float(all_numbers[i+1])*fps
            except:
                breakpoint()
            frames_index=[]
            for idss,f in enumerate(float_number_selected,start=0):
                if f>end_:
                    break
                if f>=begin_:
                    frames_index.append(idss)

            frame_dict[token]=frames_index
        frames_dict[key]=frame_dict
    return frames_dict


           

def main():

    bsp='/home/wu_tian_ci/GAFL/data/ioufile/test.json'
    test_csv='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_test.csv'
    train_csv='/home/wu_tian_ci/revisiting-spatial-temporal-layouts/data/Charades/Charades_v1_train.csv'
    df1=pd.read_csv(test_csv)
    df2=pd.read_csv(train_csv)
    dict1=get_frame_dict(df1)
    dict2=get_frame_dict(df2)
    dict1.update(dict2)
    json.dump(dict1,open(bsp,'w'))

if __name__=='__main__':
    main()