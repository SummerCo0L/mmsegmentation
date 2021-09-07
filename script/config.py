import os
import os.path as osp
from os import listdir
import glob
import cv2 as cv
import collections
import time 
import tqdm
from PIL import Image
from functools import partial
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer

import torchvision
import torchvision.transforms as transforms
import torch

from nibabel.testing import data_path
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

from script.utility import *

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


# Define the new Dataset class
classes = ('Background','Liver','Kidney','Speen','Pancrea')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123]]
img_dir = './data/img'
mask_dir = './data/mask'
@DATASETS.register_module()
class FlareDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', 
        split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

# Create configuration
from mmseg.apis import set_random_seed
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
import mmcv

cfg = Config.fromfile('configs/ocrnet/ocrnet_hr48_512x512_80k_ade20k.py')

# Since we use ony one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head[0].norm_cfg = cfg.norm_cfg
cfg.model.decode_head[1].norm_cfg = cfg.norm_cfg
# cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head[0].num_classes = 5
cfg.model.decode_head[1].num_classes = 5
cfg.model.decode_head[0].loss_decode=dict(\
                type='FocalLoss')
cfg.model.decode_head[1].loss_decode=dict(\
                type='FocalLoss')
# cfg.model.auxiliary_head.num_classes = 5

# Modify dataset type and path
cfg.dataset_type = 'FlareDataset'
cfg.data_root = './data'

cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu = 8

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (256, 256)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]



cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = mask_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = './data/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = mask_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = './data/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = mask_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = './data/val.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './weight'
# cfg.optimizer.lr = 0.0005
cfg.runner = dict(type='EpochBasedRunner', max_epochs=5)
cfg.log_config.interval = 100
cfg.evaluation.interval = 200
cfg.checkpoint_config.interval = 4000
# cfg.checkpoint_config.by_epoch=True

# cfg.resume_from = '/content/drive/MyDrive/Contest_Folder/FLARE2021/weight/logs/segmentation/hrnet_512x512/best.pth'

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}') 