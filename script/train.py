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
import numpy as np
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

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
import mmcv

from script.utility import *
from script.config import *


# Define the new Dataset class
classes = ('Background','Liver','Kidney','Speen','Pancrea')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123]]

@DATASETS.register_module()
class FlareDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', 
        split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

if __name__ == '__main__':
    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]
    
    # Build the detector
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    
    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())