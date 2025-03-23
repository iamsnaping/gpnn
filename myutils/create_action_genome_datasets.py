import argparse
import csv
import json
import os
import pickle

from natsort import natsorted
from tqdm import tqdm
import sys

def create_dataset(args):
    # Load Pickle files
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
    videoid2actions = {}
    train_ids = set()
    with open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.charades_path, "Charades_v1_train.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                videoid2actions[row["id"]] = [
                    action.split()[0] for action in row["actions"].split(";")
                ]
                train_ids.add(row["id"])
            except IndexError:
                print('222')
                continue
    val_ids = set()
    with open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.charades_path, "Charades_v1_test.csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                videoid2actions[row["id"]] = [
                    action.split()[0] for action in row["actions"].split(";")
                ]
                val_ids.add(row["id"])
            except IndexError:
                print('111')
                continue
    print("Packing and dumping datasets...")
    # Full dataset
    full_dataset = []
    for key in videoid2videoframes.keys():
        video_object = {
            "id": key,
            "frames": [],
            "actions": videoid2actions[key],
        }
        print(videoid2actions[key])
        for frame in videoid2videoframes[key]:
            if len(frame["frame_objects"]) == 0:
                continue
            video_object["frames"].append(frame)
        full_dataset.append(video_object)
    json.dump(
        full_dataset,
        open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path, "full_dataset.json"), "w"),
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
        open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path, "train_dataset.json"), "w"),
    )
    json.dump(
        val_dataset,
        open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.save_datasets_path, "val_dataset.json"), "w"),
    )


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
    args = parser.parse_args()
    create_dataset(args)



def read_pkl():
    object_bbox_and_relationship=pickle.load(
        open(
            "/home/wtc/revisiting-spatial-temporal-layouts/data/action_genome/object_bbox_and_relationship.pkl",
            "rb",
        )
    )
    json_read=json.load(open('/home/wtc/revisiting-spatial-temporal-layouts/data/videoid2size.json'))
    print(len(json_read.keys()))
    frame_name=list(object_bbox_and_relationship.keys())
    f=open('/home/wtc/revisiting-spatial-temporal-layouts/data/action_genome/relationship_classes.txt')
    n=f.readlines()
    print(n)
    new_n=[]
    sentence='/home/wtc/revisiting-spatial-temporal-layouts/src/sentence.txt'
    f=open(sentence,'a')
    for i in n:
        new_n.append(i[:-1])
    for name in frame_name:
        objs=object_bbox_and_relationship.get(name)
        for obj in objs:
            if obj.get('visible')==True:
                cls=obj.get('class')
                attention=obj.get('attention_relationship')
            
                spatial=obj.get('spatial_relationship')
               
                contacting=obj.get('contacting_relationship') 

                sentence=sentence_generator(cls,attention,spatial,contacting)
                f.write(sentence+'\n')
    f.close()

                



    print(len(frame_name))
    print(object_bbox_and_relationship.get(frame_name[500]))
    # person_bbox = pickle.load(
        # open(os.path.join('/home/wtc/revisiting-spatial-temporal-layouts',args.action_genome_path, "person_bbox.pkl"), "rb")
    # )


if __name__ == "__main__":
    # main()
    read_pkl()
