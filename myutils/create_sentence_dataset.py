import argparse
import csv
import json
import os
import pickle

from natsort import natsorted
from tqdm import tqdm
import sys






def get_dataset(dataset_num:int):

    dataset_num=dataset_num
    dataset_txt=open('/home/wtc/revisiting-spatial-temporal-layouts/data/dataset_partition.txt','r')
    dataset_txt=dataset_txt.readlines()
    train_txt=dataset_txt[2*dataset_num-1][:-1].split(',')
    test_txt=dataset_txt[2*dataset_num][:-1].split(',')
    return train_txt,test_txt


def create_video2size():
    
    object_bbox_and_relationship = pickle.load(
    open(
        '/home/wtc/revisiting-spatial-temporal-layouts/data/action_genome/object_bbox_and_relationship.pkl',
        "rb",
        )
    )
    person_bbox = pickle.load(open('/home/wtc/revisiting-spatial-temporal-layouts/data/action_genome/person_bbox.pkl', "rb"))
    frame_names = natsorted(list(object_bbox_and_relationship.keys()))
    # Generate a mapping from video id to the video frames
    videoid2videoframes = {}
    video2size={}
    for frame_name in tqdm(frame_names):

        # obtain video id and frame id
        video_id, frame_id = (e.split(".")[0] for e in os.path.split(frame_name))
        # collect objects
        if video_id not in videoid2videoframes:
            videoid2videoframes[video_id] = []
        frame_elems = {"frame_id": frame_id, "frame_objects": []}
        frame_not_none=0
        for frame_object in object_bbox_and_relationship[frame_name]:
            if not frame_object["visible"]:
                continue
            frame_not_none+=1
        # Prepare person object
        flag=False
        if frame_not_none==0:
            flag=True
        if person_bbox[frame_name]["bbox"].shape == (1, 4):
            frame_not_none+=1

        if frame_not_none!=0:
            video2size[frame_name.split('.')[0]]=[person_bbox[frame_name].get('bbox_size')[0],person_bbox[frame_name].get('bbox_size')[1]]
    json.dump(
        video2size,
        open('/home/wtc/revisiting-spatial-temporal-layouts/data/video2size/video2size.json', "w"),
    )

def create_dataset(args):

    if args.subdataset<6:
        train_txt,test_txt=get_dataset(args.subdataset)

    keys=json.load(open('/home/wtc/revisiting-spatial-temporal-layouts/data/video2size/ag.json')).keys()
    object_bbox_and_relationship = pickle.load(
        open(
            os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.action_genome_path, "object_bbox_and_relationship.pkl"),
            "rb",
        )
    )
    person_bbox = pickle.load(
        open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.action_genome_path, "person_bbox.pkl"), "rb")
    )
    frame_names = natsorted(list(object_bbox_and_relationship.keys()))
    # Generate a mapping from video id to the video frames
    videoid2videoframes = {}
    for frame_name in tqdm(frame_names):
        # if frame_name not in keys:
        #     continue
        # obtain video id and frame id
        video_id, frame_id = (e.split(".")[0] for e in os.path.split(frame_name))
        # collect objects
        if video_id not in videoid2videoframes:
            videoid2videoframes[video_id] = []
        frame_elems = {"frame_id": frame_id, "frame_objects": []}
        for frame_object in object_bbox_and_relationship[frame_name]:
            if not frame_object["visible"]:
                continue
            x1, y1 = frame_object["bbox"][:2]
            x2 = x1 + frame_object["bbox"][2]
            y2 = y1 + frame_object["bbox"][3]
            frame_elems["frame_objects"].append(
                {
                    "category": frame_object["class"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": 1.0,
                }
            )
        # Prepare person object
        if person_bbox[frame_name]["bbox"].shape == (1, 4):
            # Prepare box
            x1, y1, x2, y2 = person_bbox[frame_name]["bbox"][0]
            bbox = [float(e) for e in (x1, y1, x2, y2)]
            x1, y1, x2, y2 = bbox
            person_object = {
                "category": "person",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": person_bbox[frame_name]["bbox_score"].item(),
            }
            frame_elems["frame_objects"].append(person_object)
        # Add all frame objects
        videoid2videoframes[video_id].append(frame_elems)
    # Aggregate the actions
    # return
    videoid2actions = {}
    train_ids = set()
    val_ids = set()

    with open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.charades_path, "Charades_v1_train.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                videoid2actions[row["id"]] = [
                    action.split()[0] for action in row["actions"].split(";")
                ]
                if args.subdataset==6:
                    train_ids.add(row["id"])
                else:
                    scene=row["scene"]
                    for tx in train_txt:
                        if tx in scene:
                            train_ids.add(row["id"])
                            continue
                    for tx1 in test_txt:
                        if tx1 in scene:
                            val_ids.add(row["id"])
                            continue
            except IndexError:
                continue

    with open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.charades_path, "Charades_v1_test.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                videoid2actions[row["id"]] = [
                    action.split()[0] for action in row["actions"].split(";")
                ]
                if args.subdataset==6:
                    val_ids.add(row["id"])
                else:
                    scene=row["scene"]
                    for tx in train_txt:
                        if tx in scene:
                            train_ids.add(row["id"])
                            continue
                    for tx1 in test_txt:
                        if tx1 in scene:
                            val_ids.add(row["id"])
                            continue
            except IndexError:
                continue
    print("Packing and dumping datasets...")
    # Full dataset
    full_dataset = []
    print("keys length",len(train_ids),len(val_ids))
    for key in videoid2videoframes.keys():
        video_object = {
            "id": key,
            "frames": [],
            "actions": videoid2actions[key],
        }
        for frame in videoid2videoframes[key]:
            if len(frame["frame_objects"]) == 0:
                continue
            video_object["frames"].append(frame)
        if len(video_object.get('frames'))==0:
            continue
        full_dataset.append(video_object)
    if args.saveall==1:
        json.dump(
            full_dataset,
            open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path,'jsonfile', "full_dataset.json"), "w"),
        )
    # Training and validation dataset
    train_dataset = []
    val_dataset = []
    for element in full_dataset:
        if element["id"] in train_ids:
            train_dataset.append(element)
        elif element["id"] in val_ids:
            val_dataset.append(element)
    json.dump(
        train_dataset,
        open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path, 'jsonfile',"train_"+str(args.subdataset)+".json"), "w"),
    )
    json.dump(
        val_dataset,
        open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path, 'jsonfile',"test_"+str(args.subdataset)+".json"), "w"),
    )


def create_dataset_2(args):
    # bert_path='/home/wtc/revisiting-spatial-temporal-layouts/weight'
    # tokenizer=BertTokenizer.from_pretrained(bert_path,local_files_only=True)
    if args.subdataset<6:
        train_txt,test_txt=get_dataset(args.subdataset)
    # print(train_txt,test_txt)
    # Load Pickle files
    # spath='sentence.txt'
    # word_len=dict()
    # f=open(spath,'a')
    keys=json.load(open('/home/wtc/revisiting-spatial-temporal-layouts/data/video2size/ag.json')).keys()
    object_bbox_and_relationship = pickle.load(
        open(
            os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.action_genome_path, "object_bbox_and_relationship.pkl"),
            "rb",
        )
    )
    person_bbox = pickle.load(
        open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.action_genome_path, "person_bbox.pkl"), "rb")
    )
    frame_names = natsorted(list(object_bbox_and_relationship.keys()))
    # Generate a mapping from video id to the video frames
    videoid2videoframes = {}
    for frame_name in tqdm(frame_names):
        # if frame_name not in keys:
        #     continue
        # obtain video id and frame id
        video_id, frame_id = (e.split(".")[0] for e in os.path.split(frame_name))
        # collect objects
        if video_id not in videoid2videoframes:
            videoid2videoframes[video_id] = []
        frame_elems = {"frame_id": frame_id, "frame_objects": []}
        for frame_object in object_bbox_and_relationship[frame_name]:
            if not frame_object["visible"]:
                continue
            x1, y1 = frame_object["bbox"][:2]
            x2 = x1 + frame_object["bbox"][2]
            y2 = y1 + frame_object["bbox"][3]
            cls=frame_object.get('class')

            attention=frame_object.get('attention_relationship')

            spatial=frame_object.get('spatial_relationship')
               
            contacting=frame_object.get('contacting_relationship') 

            attention=[atten.replace('_','') for atten in attention]+['pad']*(1-len(attention))
            spatial=[atten.replace('_','') for atten in spatial]+['pad']*(5-len(spatial))
            contacting=[atten.replace('_','') for atten in contacting]+['pad']*(4-len(contacting))

            # describe
            sentence=sentence_generator(cls,attention,spatial,contacting)
            # # 
            # words=tokenizer(sentence)
            # wor_len=len(words.get('input_ids'))

            # if word_len.get(wor_len) is None:
            #     word_len[wor_len]=1
            # else:
            #     word_len[wor_len]+=1
            # f.write(sentence)
            # continue
            # print(cls,frame_object['class'])
            # breakpoint()
            frame_elems["frame_objects"].append(
                {
                    "category": frame_object["class"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": 1.0,
                    "sentence":sentence,
                    "atten":attention,
                    "spatial":spatial,
                    "contacting":contacting
                }
            )
        # Prepare person object
        if person_bbox[frame_name]["bbox"].shape == (1, 4):
            # Prepare box
            sentence='There are only people here, no other objects.'
            if len(frame_elems["frame_objects"])==0:
                sentence='there is only th'
            x1, y1, x2, y2 = person_bbox[frame_name]["bbox"][0]
            bbox = [float(e) for e in (x1, y1, x2, y2)]
            x1, y1, x2, y2 = bbox
            person_object = {
                "category": "person",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": person_bbox[frame_name]["bbox_score"].item(),
                "sentence":'',
                "atten":['pad']*1,
                "spatial":['pad']*5,
                "contacting":['pad']*4,

            }
            frame_elems["frame_objects"].append(person_object)
        # Add all frame objects
        videoid2videoframes[video_id].append(frame_elems)
    # Aggregate the actions
    # return
    videoid2actions = {}
    train_ids = set()
    val_ids = set()

    with open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.charades_path, "Charades_v1_train.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                videoid2actions[row["id"]] = [
                    action.split()[0] for action in row["actions"].split(";")
                ]
                if args.subdataset==6:
                    train_ids.add(row["id"])
                else:
                    scene=row["scene"]
                    for tx in train_txt:
                        if tx in scene:
                            train_ids.add(row["id"])
                            continue
                    for tx1 in test_txt:
                        if tx1 in scene:
                            val_ids.add(row["id"])
                            continue
            except IndexError:
                continue

    with open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.charades_path, "Charades_v1_test.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                videoid2actions[row["id"]] = [
                    action.split()[0] for action in row["actions"].split(";")
                ]
                if args.subdataset==6:
                    val_ids.add(row["id"])
                else:
                    scene=row["scene"]
                    for tx in train_txt:
                        if tx in scene:
                            train_ids.add(row["id"])
                            continue
                    for tx1 in test_txt:
                        if tx1 in scene:
                            val_ids.add(row["id"])
                            continue
            except IndexError:
                continue
    print("Packing and dumping datasets...")
    # Full dataset
    full_dataset = []
    print("keys length",len(train_ids),len(val_ids))
    for key in videoid2videoframes.keys():
        video_object = {
            "id": key,
            "frames": [],
            "actions": videoid2actions[key],
        }
        for frame in videoid2videoframes[key]:
            if len(frame["frame_objects"]) == 0:
                continue
            video_object["frames"].append(frame)
        if len(video_object.get('frames'))==0:
            continue
        full_dataset.append(video_object)
    if args.saveall==1:
        json.dump(
            full_dataset,
            open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path,'jsonfile', "full_dataset.json"), "w"),
        )
    # Training and validation dataset
    train_dataset = []
    val_dataset = []
    for element in full_dataset:
        if element["id"] in train_ids:
            train_dataset.append(element)
        elif element["id"] in val_ids:
            val_dataset.append(element)
    json.dump(
        train_dataset,
        open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path, 'jsonfile',"train_"+str(args.subdataset)+".json"), "w"),
    )
    json.dump(
        val_dataset,
        open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path, 'jsonfile',"test_"+str(args.subdataset)+".json"), "w"),
    )

    # train_csv=os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path, 'csvfile',"train_"+str(args.subdataset)+".csv")
    # test_csv=os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path, 'csvfile',"test_"+str(args.subdataset)+".csv")
    # with open(train_csv, mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     # 如果提供了表头，写入表头

    #     writer.writerow(['id'])
        # 写入数据
        # for train_key in train_ids:
        #     writer.writerow([train_key])
    # with open(test_csv, mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     # 如果提供了表头，写入表头

    #     writer.writerow(['id'])
    #     # 写入数据
    #     for test_key in val_ids:
    #         writer.writerow([test_key])


def main():
    parser = argparse.ArgumentParser(
        description="Creates a dataset from Action Genome and Charades."
    )
    parser.add_argument(
        "--action_genome_path",
        type=str,
        default="data/action_genome_v1.0",
        help="Path to the action genome directory.",
    )
    parser.add_argument(
        "--charades_path",
        type=str,
        default="data/Charades",
        help="Path to the Charades directory.",
    )
    parser.add_argument(
        "--save_datasets_path",
        type=str,
        default="data/action_genome/",
        help="Where to save the datasets.",
    )

    parser.add_argument(
        "--subdataset",
        type=int,
        default=1,
        help="subdataset",
    )
    parser.add_argument(
        "--saveall",
        type=int,
        default=0,
        help="save full dataset",
    )
    args = parser.parse_args()
    create_dataset(args)
    # tally(args)

def read_pkl():
    object_bbox_and_relationship = pickle.load(
        open(
            os.path.join('/home/wtc/revisiting-spatial-temporal-layouts/data/action_genome/object_bbox_and_relationship.pkl'),
            "rb",
        )
    )
    json_read=json.load(open('/home/wtc/revisiting-spatial-temporal-layouts/data/video2size/ag.json'))
    print(len(json_read.keys()))
    frame_name=natsorted(list(object_bbox_and_relationship.keys()))
    f=open('/home/wtc/revisiting-spatial-temporal-layouts/data/action_genome/relationship_classes.txt')
    n=f.readlines()
    print(n)
    new_n=[]
    sentence='/home/wtc/revisiting-spatial-temporal-layouts/src/sentence.txt'
    f=open(sentence,'a')
    for i in n:
        new_n.append(i[:-1])
    # print(frame_name)
    people=dict()
    for name in frame_name:
        objs=object_bbox_and_relationship.get(name)
        l=len(objs)
        # print(objs)
        # if tally_dict.get(l) is None:
        #     tally_dict[l]=1
        # else:
        #     tally_dict[l]+=1
        # if l==9:
        #     for obj in objs:
        #         if obj.get('visible') == False:
        #             continue
        #         print(obj.get('class'),end=' ')
        t=0
        p_flag=True
        for obj in objs:
            if obj.get('visible')==True:
                t+=1
        # if t==9:
        #     print(objs)
        #     print('')
        if people.get(t) is None:
            people[t]=1
        else:
            people[t]+=1


    # f.close()
    total1=0
    total2=0
    for key in sorted(people):
        print(key,people[key])
        if int(key)>0:
            total1+=people[key]
        total2+=int(key)*people[key]
    print(round(total2/(total1*9),5))
    # for key,value in people.items():
    #     print('123')
    #     print(key,value)

                



    print(len(frame_name))
    # person_bbox = pickle.load(
        # open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.action_genome_path, "person_bbox.pkl"), "rb")
    # )


def tally(args):
    max_len=0
    max_s=0
    max_c=0
    max_a=0
    max_obj=0
    if args.subdataset<6:
        train_txt,test_txt=get_dataset(args.subdataset)
    keys=json.load(open('/home/wtc/revisiting-spatial-temporal-layouts/data/video2size/ag.json')).keys()
    object_bbox_and_relationship = pickle.load(
        open(
            os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.action_genome_path, "object_bbox_and_relationship.pkl"),
            "rb",
        )
    )
    person_bbox = pickle.load(
        open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.action_genome_path, "person_bbox.pkl"), "rb")
    )
    frame_names = natsorted(list(object_bbox_and_relationship.keys()))
    # Generate a mapping from video id to the video frames
    videoid2videoframes = {}
    for frame_name in tqdm(frame_names):
        # if frame_name not in keys:
        #     continue
        # obtain video id and frame id
        video_id, frame_id = (e.split(".")[0] for e in os.path.split(frame_name))
        # collect objects
        if video_id not in videoid2videoframes:
            videoid2videoframes[video_id] = []
        obj_len=0
        frame_elems = {"frame_id": frame_id, "frame_objects": []}
        for frame_object in object_bbox_and_relationship[frame_name]:
            if not frame_object["visible"]:
                continue
            attention=frame_object.get('attention_relationship')
            spatial=frame_object.get('spatial_relationship')
            contacting=frame_object.get('contacting_relationship') 
            # print(contacting,len(contacting))
            atten_len=len(attention)
            spatial_len=len(spatial)
            cont_len=len(contacting)
            max_len=max(atten_len,spatial_len,cont_len,max_len)
            max_a=max(max_a,atten_len)
            max_s=max(max_s,spatial_len)
            max_c=max(max_c,cont_len)
    
        max_obj=max(max_obj,obj_len)
    print(max_len,max_a,max_s,max_c,max_obj)
           




def read_txt():
    base_path='/home/wtc/revisiting-spatial-temporal-layouts/data/action_genome/frame_list.txt'
    t=open(base_path,'r')
    names=t.readlines()
    total=dict()
    for name in names:
        name=name.split('.')[0]
        if total.get(name) is None:
            total[name]=1
        else:
            total[name]+=1
    values=total.values()
    print(max(values))
    print(min(values))
    num=dict()
    for value in values:
        if num.get(value) is None:
            num[value]=1
        else:
            num[value]+=1
    num_list=[]
    values=0
    for key,value in num.items():
        num_list.append([key,value])
        values+=value
    
    n=natsorted(num_list,key=lambda x:x[0],reverse=True)
    value=0
    for i in n:
        value+=i[1]
        print(i[0],i[1],value/values)
        

# def dataset():
#     train_csv=
#     test_csv=


if __name__ == "__main__":
    # f=open('/home/wtc/revisiting-spatial-temporal-layouts/data/data_save/test_1.csv','r')
    # reader=csv.DictReader(f)
    # keys=[]
    # for row in reader:
    #     keys.append(row['id'])
    # print(len(keys))
    # create_video2size()
    # main()
    # tally()
    read_pkl()
    # read_txt()
    # for i in range(1,6):
    #     print(get_dataset(i))
