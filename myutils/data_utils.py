from typing import List, Tuple
# import threading
import ffmpeg
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter, RandomCrop
from torchvision.transforms import functional as TF
import os
import torch.optim as optim
import math

def load_video(in_filepath: str):
    """Loads a video from a filepath."""
    print('in_file_path',in_filepath)
    probe = ffmpeg.probe(in_filepath)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    out, _ = (
        ffmpeg.input(in_filepath)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        # https://github.com/kkroening/ffmpeg-python/issues/68#issuecomment-443752014
        .global_args("-loglevel", "error")
        .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    return video


def sample_train_layout_indices(coord_nr_frames: int, nr_video_frames: int):
    # https://github.com/joaanna/something_else/blob/master/code/data_utils/data_loader_frames.py#L135
    average_duration = nr_video_frames * 1.0 / coord_nr_frames
    if average_duration > 0:
        offsets = np.multiply(
            list(range(coord_nr_frames)), average_duration
        ) + np.random.uniform(0, average_duration, size=coord_nr_frames)
        offsets = np.floor(offsets)
    elif nr_video_frames > coord_nr_frames:
        offsets = np.sort(np.random.randint(nr_video_frames, size=coord_nr_frames))
    else:
        offsets = np.arange(nr_video_frames)
    offsets = list(map(int, list(offsets)))
    return offsets


def get_test_layout_indices(coord_nr_frames: int, nr_video_frames: int):
    # https://github.com/joaanna/something_else/blob/master/code/data_utils/data_loader_frames.py#L148
    if nr_video_frames > coord_nr_frames:
        tick = nr_video_frames * 1.0 / coord_nr_frames
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(coord_nr_frames)])
    else:
        offsets = np.arange(nr_video_frames)
    offsets = list(map(int, list(offsets)))
    return offsets


def sample_appearance_indices(
    coord_nr_frames: int, nr_video_frames: int, train: bool, sample_rate=2
):
    # https://github.com/joaanna/something_else/blob/master/code/data_utils/data_loader_frames.py#L157
    d = coord_nr_frames * sample_rate  # 16 * 2
    if nr_video_frames > d:
        if train:
            # random sample
            offset = np.random.randint(0, nr_video_frames - d)
        else:
            # center crop
            offset = (nr_video_frames - d) // 2
        frame_list = list(range(offset, offset + d, sample_rate))
    else:
        # Temporal Augmentation
        if train:  # train
            if nr_video_frames - 2 < coord_nr_frames:
                # less frames than needed
                pos = np.linspace(0, nr_video_frames - 2, coord_nr_frames)
            else:  # take one
                pos = np.sort(
                    np.random.choice(
                        list(range(nr_video_frames - 2)), coord_nr_frames, replace=False
                    )
                )
        else:
            pos = np.linspace(0, nr_video_frames - 2, coord_nr_frames)
        frame_list = [round(p) for p in pos]
    # Without max(x, 0) bug when nr_video_frames = 1
    frame_list = [int(max(x, 0)) for x in frame_list]

    return frame_list


def pad_sequence(sequences: List[torch.Tensor], pad_tensor: torch.Tensor):
    num_trailing_dims = len(sequences[0].size()[1:])
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + (1,) * num_trailing_dims
    out_tensor = pad_tensor.repeat(out_dims)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor

    return out_tensor


class IdentityTransform:
    def __call__(self, image: Image):
        return image


class VideoColorJitter:
    # Adapted from: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py#L1140
    def __init__(self):
        (
            self.fn_idx,
            self.brightness_factor,
            self.contrast_factor,
            self.saturation_factor,
            self.hue_factor,
        ) = ColorJitter.get_params(
            brightness=(0.75, 1.25),
            contrast=(0.75, 1.25),
            saturation=(0.75, 1.25),
            hue=(-0.1, 0.1),
        )

    def __call__(self, img: Image):
        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness_factor is not None:
                img = TF.adjust_brightness(img, self.brightness_factor)
            elif fn_id == 1 and self.contrast_factor is not None:
                img = TF.adjust_contrast(img, self.contrast_factor)
            elif fn_id == 2 and self.saturation_factor is not None:
                img = TF.adjust_saturation(img, self.saturation_factor)
            elif fn_id == 3 and self.hue_factor is not None:
                img = TF.adjust_hue(img, self.hue_factor)

        return img


class ResizeBoxes:
    # Resize boxes according to the shortest size of the image. Adapted from:
    # https://github.com/chainer/chainercv/blob/master/chainercv/transforms/bbox/resize_bbox.py
    def __call__(self, box: List[int], scale_factor: float):
        out_box = [e * scale_factor for e in box]

        return out_box


class CenterCropBoxes:
    # Adapted from:
    # https://github.com/chainer/chainercv/blob/master/chainercv/transforms/bbox/translate_bbox.py
    def __init__(self, dummy_image, spatial_size: int):
        self.left = (dummy_image.size[0] - spatial_size) // 2
        self.top = (dummy_image.size[1] - spatial_size) // 2
        self.height = spatial_size
        self.width = spatial_size

    def __call__(self, box: List[int]):
        out_box = [
            box[0] - self.left,
            box[1] - self.top,
            box[2] - self.left,
            box[3] - self.top,
        ]

        return out_box


class RandomCropBoxes:
    # Adapted from:
    # https://github.com/chainer/chainercv/blob/master/chainercv/transforms/bbox/translate_bbox.py
    def __init__(self, dummy_image, spatial_size):
        self.top, self.left, self.height, self.width = RandomCrop.get_params(
            dummy_image, (spatial_size, spatial_size)
        )

    def __call__(self, box: List[int]):
        out_box = [
            box[0] - self.left,
            box[1] - self.top,
            box[2] - self.left,
            box[3] - self.top,
        ]

        return out_box


def valid_box(box: List[int], frame_size: int):
    if box[0] >= frame_size and box[2] >= frame_size:
        return False
    if box[0] <= 0 and box[2] <= 0:
        return False
    if box[1] >= frame_size and box[3] >= frame_size:
        return False
    if box[1] <= 0 and box[3] <= 0:
        return False
    return True


def clamp_box(box: List[int], frame_size: int):
    out_box = [max(0, min(e, frame_size)) for e in box]
    return out_box


def fix_box(box: List[int], video_size: Tuple[int, int]):
    # Cast box elements to integers
    box = [max(0, int(b)) for b in box]
    # If x1 > x2 or y1 > y2 switch (Hack)
    if box[0] > box[2]:
        box[0], box[2] = box[2], box[0]
    if box[1] > box[3]:
        box[1], box[3] = box[3], box[1]
    # Clamp to max size (Hack)
    if box[0] >= video_size[1]:
        box[0] = video_size[1] - 1
    if box[1] >= video_size[0]:
        box[1] = video_size[0] - 1
    if box[2] >= video_size[1]:
        box[2] = video_size[1] - 1
    if box[3] >= video_size[0]:
        box[3] = video_size[0] - 1
    # Fix if equal (Hack)
    if box[0] == box[2] and box[0] == 0:
        box[2] = 1
    if box[1] == box[3] and box[1] == 0:
        box[3] = 1
    if box[0] == box[2]:
        box[0] -= 1
    if box[1] == box[3]:
        box[1] -= 1
    return box




def get_linear_schedule_with_warmup(
    optimizer: optim.Optimizer, num_warmup_steps: int, num_training_steps: int
):
    # https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def get_cosine_schedule_with_warmup(
	optimizer: optim.Optimizer, # 如 Adam、SGD
	num_warmup_steps: int, # 热身阶段的步数，在此阶段学习率从 0 线性增加到初始学习率
	num_training_steps: int, # 总的训练步数，用于计算余弦退火的进度
	num_cycles: float = 0.5, # 余弦调度的周期数，默认为 0.5
	last_epoch: int = -1, # 用于恢复训练时的最后一个epoch的索引，默认为 -1
):
	def lr_lambda(current_step):
		# Warmup
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# decadence
		progress = float(current_step - num_warmup_steps) / float(
			max(1, num_training_steps - num_warmup_steps)
		)
		return max(
			0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)) # max(0.0, ...) 确保学习率不会变为负值
		)

	return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def add_weight_decay_one(model, weight_decay: float,lr:float,tks_lr:float):
    # https://github.com/rwightman/pytorch-image-models/blob/48371a33b11fc30ca23ed8988619af08902b215b/timm/optim/optim_factory.py#L25
    decay = []
    no_decay = []
    tks_decay=[]
    tks_nodecay=[]
    skip_list = {}

    # if hasattr(model.module, "no_weight_decay"):
    if hasattr(model, "no_weight_decay"):
        # skip_list = model.module.no_weight_decay()
        skip_list = model.no_weight_decay()
    # for name, param in model.module.named_parameters():  
    for name, param in model.named_parameters():
        if not param.requires_grad :
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if 'tks' in name:
                tks_nodecay.append(param)
            else:
                no_decay.append(param)
        else:
            if 'tks' in name:
                tks_decay.append(param)
            else:
                decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0,"lr":lr},
        {"params": decay, "weight_decay": weight_decay,"lr":lr},
        {"params": tks_nodecay, "weight_decay": 0.0,"lr":tks_lr},
        {"params": tks_decay, "weight_decay": weight_decay,"lr":tks_lr},
    ]


def add_weight_decay(model, weight_decay: float):
    # https://github.com/rwightman/pytorch-image-models/blob/48371a33b11fc30ca23ed8988619af08902b215b/timm/optim/optim_factory.py#L25
    decay = []
    no_decay = []
    skip_list = {}
    # if hasattr(model.module, "no_weight_decay"):
    if hasattr(model, "no_weight_decay"):
        # skip_list = model.module.no_weight_decay()
        skip_list = model.no_weight_decay()
    # for name, param in model.module.named_parameters():  
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:

            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def save_checkpoint(epoch,model,acc,optimizer,scheduler,time_stamp,name):
    base_path=os.path.join('/home/wu_tian_ci/GAFL/recoder/checkpoint',name)
    t_base_path=os.path.join(base_path,time_stamp[:8],time_stamp[8:12])

    if not os.path.exists(t_base_path):
        os.makedirs(t_base_path)
    if type(acc) is not str:
        path_name=str(epoch)+"_"+str(acc)+'.pth'
    else:
        path_name=str(epoch)+"_"+acc+'.pth'
    unfroze_dict={param_name:param for param_name,param in model.named_parameters() if param.requires_grad}
    state={
        "optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),
        "model": unfroze_dict
    }  
    path_name=os.path.join(t_base_path,path_name)
    print(path_name)
    torch.save(state,path_name)

def getTimeStamp():
    import time
    import calendar
    now=time.gmtime()
    now=calendar.timegm(now)

    from  datetime import datetime
    timeFormat=datetime.fromtimestamp(now)
    ans=timeFormat.strftime('%Y%m%d%H%M%S')
    return ans


class MyEvaluatorActionGenome:
    def __init__(
        self, total_instances: int, total_classes: int
    ):
        self.total_instances = total_instances
        self.total_classes = total_classes
        self.reset()
        self.best_mean_average_precision = 0.0

    def reset(self):
        self.index = 0
        self.predictions = np.zeros((self.total_instances, self.total_classes))
        self.ground_truths = np.zeros((self.total_instances, self.total_classes))

    def process(self, logits, labels):
        # Action Genome only for STLT so far
        size = logits.shape[0]
        self.predictions[self.index : self.index + size] = (
            logits.cpu().sigmoid().numpy()
        )
        self.ground_truths[self.index : self.index + size] = labels.cpu().numpy()
        self.index += size

    def evaluate(self):
        mean_average_precision, _, _ = charades_map(
            self.predictions, self.ground_truths
        )

        return {"map": mean_average_precision}

    def is_best(self):
        metrics = self.evaluate()
        if metrics["map"] > self.best_mean_average_precision:
            self.best_mean_average_precision = metrics["map"]
            return True
        return False


def mAP(submission_array, gt_array):
    # https://github.com/gsig/charades-algorithms/blob/master/pytorch/utils/map.py
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float("nan"))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    w_ap = m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float)
    return m_ap, w_ap, m_aps

def charades_map(submission_array, gt_array):
    # https://github.com/gsig/charades-algorithms/blob/master/pytorch/utils/map.py
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return mAP(fix, gt_array)

from torch import nn as nn


        


